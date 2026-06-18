"""
Quiz generation and question recommendation module.

Flow:
concept -> retrieve course chunks -> query knowledge graph -> build prompt
-> call ECNUgpt -> return structured quiz items or recommended questions.
"""
import json
import re
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from graph_rag.pipeline import call_ecnu_gpt
from knowledge_graph.graph import get_kg
from shared.schemas import QuizItem, QuizRequest, QuizResponse
from vector_search.retriever import get_retriever

router = APIRouter()


class RecommendRequest(BaseModel):
    concept: str
    num_questions: int = Field(default=5, ge=1, le=10)


class RecommendedQuestion(BaseModel):
    id: int
    question: str
    reason: str
    difficulty: str = "medium"
    related_concepts: list[str] = []


class RecommendResponse(BaseModel):
    concept: str
    questions: list[RecommendedQuestion]
    source_chunk_ids: list[str]
    related_triples: list[dict]


def _search_course_chunks(query: str, top_k: int = 5):
    retriever = get_retriever()
    try:
        return retriever.search(query, top_k=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _get_related_triples(concept: str, depth: int = 1) -> list[dict]:
    kg = get_kg()
    triples = []
    seen = set()

    candidate_entities = []
    for entity in kg.get_all_entities():
        name = entity.get("name", "")
        entity_id = entity.get("id", "")
        if name == concept or entity_id == concept or concept in name or name in concept:
            candidate_entities.append(name)

    if not candidate_entities and kg.get_entity(concept):
        candidate_entities.append(concept)

    for entity in candidate_entities[:3]:
        for triple in kg.get_neighbors(entity, depth=depth):
            key = (triple["head"], triple["relation"], triple["tail"])
            if key in seen:
                continue
            seen.add(key)
            triples.append(triple)

    return triples


def _format_chunks(search_results) -> str:
    if not search_results:
        return "未检索到相关课程资料。"

    lines = []
    for result in search_results:
        lines.append(
            f"[来源: {result.chunk.chunk_id} | 相似度: {result.similarity}]\n"
            f"{result.chunk.text}"
        )
    return "\n\n".join(lines)


def _format_triples(triples: list[dict]) -> str:
    if not triples:
        return "未检索到相关知识图谱关系。"

    return "\n".join(
        f"{triple['head']} -{triple['relation']}-> {triple['tail']}"
        for triple in triples
    )


def _extract_json_object(text: str) -> dict:
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"模型没有返回合法 JSON，请重试或减少题目数量。原始返回: {text[:500]}",
        ) from exc


def _build_quiz_prompt(req: QuizRequest, search_results, triples: list[dict]) -> str:
    chunk_context = _format_chunks(search_results)
    kg_context = _format_triples(triples)
    question_types = ", ".join(req.question_types)

    return f"""
你是一个 NLP 课程智能教学助手。请根据课程资料和知识图谱，为指定知识点生成练习题。

知识点：
{req.concept}

题目数量：
{req.num_questions}

题型范围：
{question_types}

课程资料：
{chunk_context}

知识图谱关系：
{kg_context}

请严格返回 JSON，不要输出 Markdown，不要输出解释性前后缀。
JSON 格式如下：
{{
  "questions": [
    {{
      "id": 1,
      "type": "choice",
      "question": "题干",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "正确答案",
      "explanation": "解析",
      "source_chunk_ids": ["来源chunk_id"]
    }}
  ]
}}

要求：
1. 题目必须围绕知识点和课程资料，不要编造无依据内容。
2. choice 题必须提供 4 个选项。
3. fill 题 options 设为 null。
4. short_answer 题 options 设为 null。
5. explanation 要说明为什么答案正确。
6. source_chunk_ids 必须来自上方课程资料的来源 id；如果没有资料来源，使用空列表。
""".strip()


def _build_recommend_prompt(req: RecommendRequest, search_results, triples: list[dict]) -> str:
    chunk_context = _format_chunks(search_results)
    kg_context = _format_triples(triples)

    return f"""
你是一个 NLP 课程智能教学助手。请根据课程资料和知识图谱，为学生推荐后续可以追问的问题。

当前知识点：
{req.concept}

推荐问题数量：
{req.num_questions}

课程资料：
{chunk_context}

知识图谱关系：
{kg_context}

请严格返回 JSON，不要输出 Markdown，不要输出解释性前后缀。
JSON 格式如下：
{{
  "questions": [
    {{
      "id": 1,
      "question": "推荐追问的问题",
      "reason": "为什么这个问题值得继续学习",
      "difficulty": "easy/medium/hard",
      "related_concepts": ["相关概念1", "相关概念2"]
    }}
  ]
}}

要求：
1. 推荐问题要适合学生循序渐进学习。
2. 覆盖定义、对比、原理、应用、易混点等不同角度。
3. 不要推荐脱离课程资料和知识图谱的问题。
""".strip()


@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(req: QuizRequest):
    """Generate structured quiz questions for a concept."""
    search_results = _search_course_chunks(req.concept, top_k=5)
    triples = _get_related_triples(req.concept, depth=1)
    prompt = _build_quiz_prompt(req, search_results, triples)
    raw_answer = call_ecnu_gpt(prompt)
    data = _extract_json_object(raw_answer)

    questions = []
    for item in data.get("questions", [])[:req.num_questions]:
        questions.append(
            QuizItem(
                id=item.get("id", len(questions) + 1),
                type=item.get("type", "short_answer"),
                question=item.get("question", ""),
                options=item.get("options"),
                answer=item.get("answer", ""),
                explanation=item.get("explanation", ""),
                source_chunk_ids=item.get("source_chunk_ids", []),
            )
        )

    return QuizResponse(concept=req.concept, questions=questions)


@router.post("/recommend", response_model=RecommendResponse)
async def recommend_questions(req: RecommendRequest):
    """Recommend follow-up questions for a concept."""
    search_results = _search_course_chunks(req.concept, top_k=5)
    triples = _get_related_triples(req.concept, depth=1)
    prompt = _build_recommend_prompt(req, search_results, triples)
    raw_answer = call_ecnu_gpt(prompt)
    data = _extract_json_object(raw_answer)

    questions = []
    for item in data.get("questions", [])[:req.num_questions]:
        questions.append(
            RecommendedQuestion(
                id=item.get("id", len(questions) + 1),
                question=item.get("question", ""),
                reason=item.get("reason", ""),
                difficulty=item.get("difficulty", "medium"),
                related_concepts=item.get("related_concepts", []),
            )
        )

    return RecommendResponse(
        concept=req.concept,
        questions=questions,
        source_chunk_ids=[result.chunk.chunk_id for result in search_results],
        related_triples=triples,
    )
