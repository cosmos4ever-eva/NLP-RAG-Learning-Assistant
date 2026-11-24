from collections.abc import Callable
from typing import Any

from concept_compare import CompareRequest, compare_concepts
from graph_rag.pipeline import answer_question
from learning_analytics import (
    TodoCreateRequest,
    add_todo,
    get_learning_report,
    record_question_entities,
)
from learning_path import LearningPathRequest, get_learning_path
from quiz_gen import generate_quiz
from shared.schemas import QuizRequest


async def course_qa(arguments: dict[str, Any]) -> dict[str, Any]:
    question = str(arguments.get("question", "")).strip()
    if not question:
        raise ValueError("course_qa需要question参数。")
    record_question_entities(question)
    return answer_question(question, top_k=3, depth=1)


async def concept_compare(arguments: dict[str, Any]) -> dict[str, Any]:
    request = CompareRequest.model_validate(arguments)
    return (await compare_concepts(request)).model_dump()


async def learning_path(arguments: dict[str, Any]) -> dict[str, Any]:
    request = LearningPathRequest.model_validate(arguments)
    return (await get_learning_path(request)).model_dump()


async def generate_quiz_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    request = QuizRequest.model_validate(arguments)
    return (await generate_quiz(request)).model_dump()


async def learning_report(arguments: dict[str, Any]) -> dict[str, Any]:
    return (await get_learning_report()).model_dump()


async def create_todo(arguments: dict[str, Any]) -> dict[str, Any]:
    request = TodoCreateRequest.model_validate(arguments)
    return await add_todo(request)


def build_default_tools() -> dict[str, Callable[[dict[str, Any]], Any]]:
    """Return the allowlisted capabilities exposed to the planner."""
    return {
        "course_qa": course_qa,
        "concept_compare": concept_compare,
        "learning_path": learning_path,
        "generate_quiz": generate_quiz_tool,
        "learning_report": learning_report,
        "create_todo": create_todo,
    }
