from fastapi import APIRouter, Depends

from .agent import LearningAgent
from .planner import ModelPlanner
from .schemas import AgentRequest, AgentResponse
from .tools import build_default_tools


router = APIRouter()
_agent = LearningAgent(planner=ModelPlanner(), tools=build_default_tools())


def get_agent() -> LearningAgent:
    return _agent


@router.post("/run", response_model=AgentResponse)
async def run_agent(
    request: AgentRequest,
    agent: LearningAgent = Depends(get_agent),
) -> AgentResponse:
    return await agent.run(request)
