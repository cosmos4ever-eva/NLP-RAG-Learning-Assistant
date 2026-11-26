import inspect
import json
import re
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .model_provider import call_agent_model
from .schemas import AgentRequest, AgentResponse, ReviewDecision, TraceEvent


class ModelReflectionAgent:
    """Review an executor result with a separate, structured critic role."""

    def __init__(self, model_call=call_agent_model):
        self.model_call = model_call

    def review(self, goal: str, result: AgentResponse) -> ReviewDecision:
        deterministic = self._deterministic_review(result)
        if deterministic is not None:
            return deterministic
        try:
            raw = self.model_call(self._build_prompt(goal, result))
            decision = ReviewDecision.model_validate(self._extract_json(raw))
        except Exception:
            return ReviewDecision(
                verdict="revise",
                goal_completion=0.0,
                grounding=0.0,
                tool_reliability=0.0,
                issues=["评测Agent暂不可用"],
                feedback="本次结果尚未完成独立评测，请稍后重新验收。",
                review_available=False,
            )
        failed_dimensions = []
        if decision.goal_completion < 0.8:
            failed_dimensions.append("目标完成度不足")
        if decision.grounding < 0.6:
            failed_dimensions.append("课程依据不足")
        if decision.tool_reliability < 0.8:
            failed_dimensions.append("工具执行可靠性不足")
        if decision.verdict == "pass" and failed_dimensions:
            return decision.model_copy(update={
                "verdict": "revise",
                "issues": [*decision.issues, *failed_dimensions],
                "feedback": "请优先修正：" + "、".join(failed_dimensions) + "。",
            })
        return decision

    @staticmethod
    def _deterministic_review(result: AgentResponse) -> ReviewDecision | None:
        if result.status != "completed":
            return ReviewDecision(
                verdict="revise",
                goal_completion=0.0,
                grounding=0.0,
                tool_reliability=0.0,
                issues=[f"执行Agent状态为{result.status}"],
                feedback="执行Agent未正常完成，请根据原始目标重新规划并执行。",
            )
        if not result.final_answer.strip():
            return ReviewDecision(
                verdict="revise",
                goal_completion=0.0,
                grounding=0.0,
                tool_reliability=1.0,
                issues=["最终回答为空"],
                feedback="请生成完整的最终回答。",
            )
        failed_tools = [item.tool for item in result.tool_calls if item.status == "error"]
        if failed_tools:
            names = "、".join(failed_tools)
            return ReviewDecision(
                verdict="revise",
                goal_completion=0.5,
                grounding=0.5,
                tool_reliability=0.0,
                issues=[f"工具调用失败：{names}"],
                feedback=f"请重新规划并修复或替代失败工具：{names}。",
            )
        return None

    @staticmethod
    def _extract_json(raw: str) -> dict:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        candidate = fenced.group(1) if fenced else None
        if candidate is None:
            start = raw.find("{")
            end = raw.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("评测Agent没有返回JSON结果。")
            candidate = raw[start:end + 1]
        return json.loads(candidate)

    @staticmethod
    def _build_prompt(goal: str, result: AgentResponse) -> str:
        tool_evidence = json.dumps(
            [item.model_dump(mode="json") for item in result.tool_calls],
            ensure_ascii=False,
            default=str,
        )[:12000]
        return f"""
你是NLP课程学习系统中独立的评测/反思Agent。请审查执行Agent是否真正完成用户目标，
不要继续回答用户问题，也不要因为文风流畅就判定通过。

原始目标：
{goal}

执行状态：{result.status}

最终回答：
{result.final_answer}

工具调用与Observation：
{tool_evidence}

请分别评估：
1. goal_completion：是否完成原始目标中的全部子任务。
2. grounding：知识性结论是否得到课程来源、图谱关系或工具结果支持。
3. tool_reliability：只评价实际工具调用是否成功；所有调用均为success时应为1.0。
   如果认为缺少某个必要工具，应扣goal_completion而不是tool_reliability。

只有在目标基本完整、依据充分且无关键工具错误时才能返回pass；否则返回revise。
严格输出一个JSON对象：
{{
  "verdict": "pass或revise",
  "goal_completion": 0.0,
  "grounding": 0.0,
  "tool_reliability": 0.0,
  "issues": ["具体问题"],
  "feedback": "供执行Agent直接使用的具体修改要求"
}}
""".strip()


class ReflectionState(TypedDict, total=False):
    original_request: AgentRequest
    current_request: AgentRequest
    original_goal: str
    attempt: int
    base_session_id: str | None
    result: AgentResponse
    review: ReviewDecision
    review_history: list[ReviewDecision]
    combined_trace: list[TraceEvent]
    combined_tool_calls: list
    final_response: AgentResponse


class ReflectiveLearningAgent:
    """Run executor and reviewer agents as a conditional LangGraph workflow."""

    def __init__(self, executor, reviewer, checkpointer=None):
        self.executor = executor
        self.reviewer = reviewer
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(ReflectionState)
        builder.add_node("execute", self._execute_node)
        builder.add_node("review", self._review_node)
        builder.add_node("revise", self._revision_node)
        builder.add_edge(START, "execute")
        builder.add_edge("execute", "review")
        builder.add_conditional_edges(
            "review",
            self._route_after_review,
            {"revise": "revise", "end": END},
        )
        builder.add_edge("revise", "execute")
        return builder.compile(checkpointer=self.checkpointer)

    async def _execute_node(self, state: ReflectionState) -> dict[str, Any]:
        attempt = state.get("attempt", 0) + 1
        result = await self.executor.run(state["current_request"])
        base_session_id = state.get("base_session_id") or result.session_id
        combined_tool_calls = list(state.get("combined_tool_calls", []))
        combined_tool_calls.extend([
            item.model_copy(update={"attempt": attempt})
            for item in result.tool_calls
        ])
        combined_trace = list(state.get("combined_trace", []))
        for event in result.trace:
            combined_trace.append(event.model_copy(update={
                "step": len(combined_trace) + 1,
            }))
        return {
            "attempt": attempt,
            "result": result,
            "base_session_id": base_session_id,
            "combined_tool_calls": combined_tool_calls,
            "combined_trace": combined_trace,
        }

    async def _review_node(self, state: ReflectionState) -> dict[str, Any]:
        review = self.reviewer.review(state["original_goal"], state["result"])
        if inspect.isawaitable(review):
            review = await review
        review = review.model_copy(update={"attempt": state["attempt"]})
        review_history = [*state.get("review_history", []), review]
        combined_trace = list(state.get("combined_trace", []))
        combined_trace.append(TraceEvent(
            step=len(combined_trace) + 1,
            phase="review",
            detail=f"Reflection agent verdict: {review.verdict}.",
        ))
        update: dict[str, Any] = {
            "review": review,
            "review_history": review_history,
            "combined_trace": combined_trace,
        }

        request = state["original_request"]
        if not review.review_available:
            status = "review_failed"
        elif review.verdict == "pass":
            status = state["result"].status
        elif state["attempt"] > request.max_reflections:
            status = "needs_revision"
        else:
            return update

        update["final_response"] = state["result"].model_copy(update={
            "session_id": state["base_session_id"],
            "status": status,
            "review": review,
            "review_history": review_history,
            "attempt_count": state["attempt"],
            "tool_calls": state.get("combined_tool_calls", []),
            "trace": combined_trace,
        })
        return update

    @staticmethod
    def _route_after_review(state: ReflectionState) -> str:
        return "end" if state.get("final_response") is not None else "revise"

    @staticmethod
    def _revision_node(state: ReflectionState) -> dict[str, Any]:
        combined_trace = list(state.get("combined_trace", []))
        combined_trace.append(TraceEvent(
            step=len(combined_trace) + 1,
            phase="revision",
            detail="Executor will retry using reflection feedback.",
        ))
        review = state["review"]
        issue_text = "；".join(review.issues) or "未具体列出"
        revised_goal = (
            f"原始学习目标：{state['original_goal']}\n"
            f"上一次结果的问题：{issue_text}\n"
            f"评测Agent的修改要求：{review.feedback}\n"
            "请重新完成原始目标，修正上述问题，不要在最终回答中讨论评测过程。"
        )
        original_request = state["original_request"]
        return {
            "current_request": AgentRequest(
                goal=revised_goal,
                session_id=f"{state['base_session_id']}:reflection:{state['attempt']}",
                max_steps=original_request.max_steps,
                max_reflections=0,
            ),
            "combined_trace": combined_trace,
        }

    async def run(self, request: AgentRequest) -> AgentResponse:
        workflow_id = request.session_id or str(uuid4())
        state = await self.graph.ainvoke(
            {
                "original_request": request,
                "current_request": request,
                "original_goal": request.goal,
                "attempt": 0,
                "base_session_id": request.session_id,
                "review_history": [],
                "combined_trace": [],
                "combined_tool_calls": [],
                "final_response": None,
            },
            config={"configurable": {"thread_id": workflow_id}},
        )
        return state["final_response"]
