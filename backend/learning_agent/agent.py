import inspect
from collections.abc import Callable
from typing import Any, TypedDict
from uuid import uuid4

from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .schemas import (
    AgentRequest,
    AgentResponse,
    PlannerDecision,
    ToolCallRecord,
    TraceEvent,
)


class ExecutorState(TypedDict, total=False):
    goal: str
    max_steps: int
    current_step: int
    observations: list[dict[str, Any]]
    decision: PlannerDecision | None
    tool_calls: list[ToolCallRecord]
    trace: list[TraceEvent]
    status: str
    final_answer: str
    next_actions: list[str]


class LearningAgent:
    """Run the guarded planner-executor loop as a persistent LangGraph."""

    def __init__(
        self,
        planner,
        tools: dict[str, BaseTool | Callable[[dict[str, Any]], Any]],
        checkpointer=None,
    ):
        self.planner = planner
        self.tools = tools
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(ExecutorState)
        builder.add_node("plan", self._plan_node)
        builder.add_node("execute_tool", self._tool_node)
        builder.add_node("max_steps", self._max_steps_node)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"execute_tool": "execute_tool", "end": END},
        )
        builder.add_conditional_edges(
            "execute_tool",
            self._route_after_tool,
            {"plan": "plan", "max_steps": "max_steps"},
        )
        builder.add_edge("max_steps", END)
        return builder.compile(checkpointer=self.checkpointer)

    async def _plan_node(self, state: ExecutorState) -> dict[str, Any]:
        step = state.get("current_step", 0) + 1
        trace = list(state.get("trace", []))
        try:
            decision = self.planner.decide(
                state["goal"],
                list(state.get("observations", [])),
                list(self.tools),
            )
            if inspect.isawaitable(decision):
                decision = await decision
        except Exception:
            trace.append(TraceEvent(
                step=step,
                phase="guardrail",
                detail="Planner failed to produce a valid structured decision.",
            ))
            return {
                "current_step": step,
                "decision": None,
                "status": "failed",
                "final_answer": "Agent规划失败，请稍后重试或将学习目标描述得更具体。",
                "trace": trace,
            }

        trace.append(TraceEvent(
            step=step,
            phase="plan",
            detail=decision.reasoning or f"Planner chose {decision.action}.",
        ))
        update: dict[str, Any] = {
            "current_step": step,
            "decision": decision,
            "trace": trace,
        }
        if decision.action == "finish":
            trace.append(TraceEvent(
                step=step,
                phase="finish",
                detail="Planner completed the learning goal.",
            ))
            update.update({
                "status": "completed",
                "final_answer": decision.final_answer,
                "next_actions": decision.next_actions,
                "trace": trace,
            })
        return update

    @staticmethod
    def _route_after_plan(state: ExecutorState) -> str:
        if state.get("status") in {"completed", "failed"}:
            return "end"
        return "execute_tool"

    async def _tool_node(self, state: ExecutorState) -> dict[str, Any]:
        decision = state.get("decision")
        if decision is None or decision.action != "tool":
            raise RuntimeError("Tool node requires a tool decision.")

        step = state["current_step"]
        observations = list(state.get("observations", []))
        tool_calls = list(state.get("tool_calls", []))
        trace = list(state.get("trace", []))
        tool_name = decision.tool or ""

        duplicate = any(
            item.get("tool") == tool_name
            and item.get("arguments") == decision.arguments
            and item.get("status") == "success"
            for item in observations
        )
        if duplicate:
            error = "已阻止相同参数的重复工具调用。"
            record = ToolCallRecord(
                step=step,
                tool=tool_name,
                arguments=decision.arguments,
                status="error",
                error=error,
            )
            trace.append(TraceEvent(step=step, phase="guardrail", detail=error))
        else:
            selected_tool = self.tools.get(tool_name)
            if selected_tool is None:
                error = f"Tool is not allowed: {decision.tool}"
                record = ToolCallRecord(
                    step=step,
                    tool=tool_name,
                    arguments=decision.arguments,
                    status="error",
                    error=error,
                )
                trace.append(TraceEvent(step=step, phase="guardrail", detail=error))
            else:
                try:
                    if isinstance(selected_tool, BaseTool):
                        observation = await selected_tool.ainvoke(decision.arguments)
                    else:
                        observation = selected_tool(decision.arguments)
                        if inspect.isawaitable(observation):
                            observation = await observation
                    record = ToolCallRecord(
                        step=step,
                        tool=tool_name,
                        arguments=decision.arguments,
                        status="success",
                        observation=observation,
                    )
                    trace.append(TraceEvent(
                        step=step,
                        phase="tool",
                        detail=f"Executed {tool_name} successfully.",
                    ))
                except Exception as exc:
                    record = ToolCallRecord(
                        step=step,
                        tool=tool_name,
                        arguments=decision.arguments,
                        status="error",
                        error=str(exc),
                    )
                    trace.append(TraceEvent(
                        step=step,
                        phase="tool",
                        detail=f"{tool_name} failed: {exc}",
                    ))

        tool_calls.append(record)
        observations.append(record.model_dump())
        return {
            "observations": observations[-20:],
            "tool_calls": tool_calls,
            "trace": trace,
        }

    @staticmethod
    def _route_after_tool(state: ExecutorState) -> str:
        if state["current_step"] >= state["max_steps"]:
            return "max_steps"
        return "plan"

    @staticmethod
    def _max_steps_node(state: ExecutorState) -> dict[str, Any]:
        trace = list(state.get("trace", []))
        trace.append(TraceEvent(
            step=state["max_steps"],
            phase="guardrail",
            detail="Stopped after reaching the maximum number of steps.",
        ))
        return {
            "status": "max_steps",
            "final_answer": "Agent达到最大执行步数，已安全停止。",
            "trace": trace,
        }

    async def run(self, request: AgentRequest) -> AgentResponse:
        session_id = request.session_id or str(uuid4())
        state = await self.graph.ainvoke(
            {
                "goal": request.goal,
                "max_steps": request.max_steps,
                "current_step": 0,
                "decision": None,
                "tool_calls": [],
                "trace": [],
                "status": "",
                "final_answer": "",
                "next_actions": [],
            },
            config={"configurable": {"thread_id": session_id}},
        )
        return AgentResponse(
            session_id=session_id,
            status=state["status"],
            final_answer=state["final_answer"],
            tool_calls=state.get("tool_calls", []),
            next_actions=state.get("next_actions", []),
            trace=state.get("trace", []),
        )
