from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    goal: str = Field(min_length=1)
    session_id: str | None = None
    max_steps: int = Field(default=4, ge=1, le=8)
    max_reflections: int = Field(default=1, ge=0, le=2)


class PlannerDecision(BaseModel):
    action: Literal["tool", "finish"]
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    final_answer: str = ""
    next_actions: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    step: int
    attempt: int = 1
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "error"]
    observation: Any = None
    error: str = ""


class TraceEvent(BaseModel):
    step: int
    phase: Literal["plan", "tool", "finish", "guardrail", "review", "revision"]
    detail: str


class ReviewDecision(BaseModel):
    verdict: Literal["pass", "revise"]
    goal_completion: float = Field(ge=0.0, le=1.0)
    grounding: float = Field(ge=0.0, le=1.0)
    tool_reliability: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    feedback: str = ""
    attempt: int = 1
    review_available: bool = True


class AgentResponse(BaseModel):
    session_id: str
    status: Literal[
        "completed", "max_steps", "failed", "needs_revision", "review_failed"
    ]
    final_answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    review: ReviewDecision | None = None
    review_history: list[ReviewDecision] = Field(default_factory=list)
    attempt_count: int = 1
