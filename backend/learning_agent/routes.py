from fastapi import APIRouter, Depends

from .agent import LearningAgent
from .planner import ModelPlanner
from .reflection import ModelReflectionAgent, ReflectiveLearningAgent
from .schemas import AgentRequest, AgentResponse
from .tools import build_default_tools


router = APIRouter()
_executor = LearningAgent(planner=ModelPlanner(), tools=build_default_tools())
_agent = ReflectiveLearningAgent(
    executor=_executor,
    reviewer=ModelReflectionAgent(),
)


def get_agent() -> ReflectiveLearningAgent:
    return _agent


@router.post("/run", response_model=AgentResponse)
async def run_agent(
    request: AgentRequest,
    agent: LearningAgent = Depends(get_agent),
) -> AgentResponse:
    return await agent.run(request)
