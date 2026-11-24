import inspect
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from .schemas import (
    AgentRequest,
    AgentResponse,
    ToolCallRecord,
    TraceEvent,
)


class LearningAgent:
    """Run a guarded planner-executor loop behind one small interface."""

    def __init__(self, planner, tools: dict[str, Callable[[dict[str, Any]], Any]]):
        self.planner = planner
        self.tools = tools
        self._session_observations: dict[str, list[dict[str, Any]]] = {}

    async def run(self, request: AgentRequest) -> AgentResponse:
        session_id = request.session_id or str(uuid4())
        observations = list(self._session_observations.get(session_id, []))
        tool_calls: list[ToolCallRecord] = []
        trace: list[TraceEvent] = []

        for step in range(1, request.max_steps + 1):
            try:
                decision = self.planner.decide(
                    request.goal,
                    observations,
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
                return AgentResponse(
                    session_id=session_id,
                    status="failed",
                    final_answer="Agent规划失败，请稍后重试或将学习目标描述得更具体。",
                    tool_calls=tool_calls,
                    trace=trace,
                )

            trace.append(TraceEvent(
                step=step,
                phase="plan",
                detail=decision.reasoning or f"Planner chose {decision.action}.",
            ))

            if decision.action == "finish":
                trace.append(TraceEvent(
                    step=step,
                    phase="finish",
                    detail="Planner completed the learning goal.",
                ))
                return AgentResponse(
                    session_id=session_id,
                    status="completed",
                    final_answer=decision.final_answer,
                    tool_calls=tool_calls,
                    next_actions=decision.next_actions,
                    trace=trace,
                )

            duplicate = any(
                item.get("tool") == (decision.tool or "")
                and item.get("arguments") == decision.arguments
                and item.get("status") == "success"
                for item in observations
            )
            if duplicate:
                error = "已阻止相同参数的重复工具调用。"
                record = ToolCallRecord(
                    step=step,
                    tool=decision.tool or "",
                    arguments=decision.arguments,
                    status="error",
                    error=error,
                )
                tool_calls.append(record)
                observations.append(record.model_dump())
                self._session_observations[session_id] = observations[-20:]
                trace.append(TraceEvent(step=step, phase="guardrail", detail=error))
                continue

            tool = self.tools.get(decision.tool or "")
            if tool is None:
                error = f"Tool is not allowed: {decision.tool}"
                record = ToolCallRecord(
                    step=step,
                    tool=decision.tool or "",
                    arguments=decision.arguments,
                    status="error",
                    error=error,
                )
                tool_calls.append(record)
                observations.append(record.model_dump())
                trace.append(TraceEvent(step=step, phase="guardrail", detail=error))
                continue

            try:
                observation = tool(decision.arguments)
                if inspect.isawaitable(observation):
                    observation = await observation
                record = ToolCallRecord(
                    step=step,
                    tool=decision.tool or "",
                    arguments=decision.arguments,
                    status="success",
                    observation=observation,
                )
                trace.append(TraceEvent(
                    step=step,
                    phase="tool",
                    detail=f"Executed {decision.tool} successfully.",
                ))
            except Exception as exc:
                record = ToolCallRecord(
                    step=step,
                    tool=decision.tool or "",
                    arguments=decision.arguments,
                    status="error",
                    error=str(exc),
                )
                trace.append(TraceEvent(
                    step=step,
                    phase="tool",
                    detail=f"{decision.tool} failed: {exc}",
                ))

            tool_calls.append(record)
            observations.append(record.model_dump())
            self._session_observations[session_id] = observations[-20:]

        trace.append(TraceEvent(
            step=request.max_steps,
            phase="guardrail",
            detail="Stopped after reaching the maximum number of steps.",
        ))
        return AgentResponse(
            session_id=session_id,
            status="max_steps",
            final_answer="Agent达到最大执行步数，已安全停止。",
            tool_calls=tool_calls,
            trace=trace,
        )
