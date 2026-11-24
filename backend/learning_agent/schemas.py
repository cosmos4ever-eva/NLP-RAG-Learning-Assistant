from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    goal: str = Field(min_length=1)
    session_id: str | None = None
    max_steps: int = Field(default=4, ge=1, le=8)


class PlannerDecision(BaseModel):
    action: Literal["tool", "finish"]
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    final_answer: str = ""
    next_actions: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    step: int
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "error"]
    observation: Any = None
    error: str = ""


class TraceEvent(BaseModel):
    step: int
    phase: Literal["plan", "tool", "finish", "guardrail"]
    detail: str


class AgentResponse(BaseModel):
    session_id: str
    status: Literal["completed", "max_steps", "failed"]
    final_answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
