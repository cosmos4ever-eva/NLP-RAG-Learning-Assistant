from typing import Any, Literal

from langchain.tools import tool
from langchain_core.tools import BaseTool

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


@tool("course_qa")
async def course_qa(question: str) -> dict[str, Any]:
    """基于课程证据和知识图谱回答一个NLP课程问题。"""
    question = question.strip()
    if not question:
        raise ValueError("course_qa需要question参数。")
    record_question_entities(question)
    return answer_question(question, top_k=3, depth=1)


@tool("concept_compare")
async def concept_compare(concept_a: str, concept_b: str) -> dict[str, Any]:
    """对两个易混或相关的NLP概念进行有证据的比较。"""
    request = CompareRequest(concept_a=concept_a, concept_b=concept_b)
    return (await compare_concepts(request)).model_dump()


@tool("learning_path")
async def learning_path(target_entity: str) -> dict[str, Any]:
    """根据课程知识图谱生成目标概念的前序学习路径。"""
    request = LearningPathRequest(target_entity=target_entity)
    return (await get_learning_path(request)).model_dump()


@tool("generate_quiz")
async def generate_quiz_tool(
    concept: str,
    num_questions: int = 3,
    difficulty: Literal["easy", "medium", "hard"] = "medium",
) -> dict[str, Any]:
    """围绕指定课程概念生成有证据支持的练习题。"""
    request = QuizRequest(
        concept=concept,
        num_questions=num_questions,
        difficulty=difficulty,
    )
    return (await generate_quiz(request)).model_dump()


@tool("learning_report")
async def learning_report() -> dict[str, Any]:
    """根据问答、错题和知识覆盖度生成学习诊断报告。"""
    return (await get_learning_report()).model_dump()


@tool("create_todo")
async def create_todo(content: str, source: str = "agent") -> dict[str, Any]:
    """把明确的下一步学习行动写入学习Todo列表。"""
    request = TodoCreateRequest(content=content, source=source)
    return await add_todo(request)


def build_default_tools() -> dict[str, BaseTool]:
    """Return the six allowlisted LangChain tools exposed to the planner."""
    tools = [
        course_qa,
        concept_compare,
        learning_path,
        generate_quiz_tool,
        learning_report,
        create_todo,
    ]
    return {item.name: item for item in tools}
