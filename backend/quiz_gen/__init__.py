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
from learning_analytics import record_concepts, record_mistake
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


class QuizEvaluationRequest(BaseModel):
    concept: str
    type: str = "short_answer"
    question: str
    options: Optional[list[str]] = None
    answer: str
    explanation: str = ""
    difficulty: str = "medium"


class QuizEvaluationResponse(BaseModel):
    concept: str
    overall_score: float
    clarity_score: float
    correctness_score: float
    difficulty_match_score: float
    evidence_score: float
    score_comments: dict[str, str] = Field(default_factory=dict)
    problems: list[str]
    suggestions: list[str]


class QuizOptimizationRequest(BaseModel):
    concept: str
    type: str = "short_answer"
    question: str
    options: Optional[list[str]] = None
    answer: str
    explanation: str = ""
    difficulty: str = "medium"
    problems: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    score_comments: dict[str, str] = Field(default_factory=dict)


class QuizOptimizationResponse(BaseModel):
    concept: str
    optimized_question: QuizItem
    optimization_notes: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str]
    related_triples: list[dict]


class QuizSubmitRequest(BaseModel):
    concept: str
    question: str
    student_answer: str
    reference_answer: str
    explanation: str = ""


class QuizSubmitResponse(BaseModel):
    concept: str
    score: float
    is_correct: bool
    feedback: str
    wrong_concepts: list[str]
    suggested_review: list[str]
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


def _filter_review_concepts(concepts: list[str]) -> list[str]:
    kg = get_kg()
    entity_names = {entity["name"] for entity in kg.get_all_entities()}
    entity_ids = {entity["id"]: entity["name"] for entity in kg.get_all_entities()}

    filtered = []
    for concept in concepts:
        if not concept:
            continue
        concept = str(concept).strip()
        if concept in entity_names:
            filtered.append(concept)
        elif concept in entity_ids:
            filtered.append(entity_ids[concept])

    return list(dict.fromkeys(filtered))


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


def _as_score(value, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _score_comments(data: dict) -> dict[str, str]:
    comments = data.get("score_comments")
    if not isinstance(comments, dict):
        comments = {}

    fallback = {
        "overall_score": "综合来看，题目基本符合自动生成练习题的质量要求。",
        "clarity_score": "题干和必要信息表达清楚，学生能够理解作答要求。",
        "correctness_score": "参考答案和解析与课程资料及知识图谱基本一致。",
        "difficulty_match_score": "题目难度与目标难度基本匹配。",
        "evidence_score": "题目内容能够从课程资料或知识图谱中找到支撑。",
    }
    return {
        key: str(comments.get(key) or fallback[key])
        for key in fallback
    }


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

难度要求：
{req.difficulty}

难度说明：
easy：考查基本定义、概念识别和简单事实。
medium：考查概念关系、方法区别和原理理解。
hard：考查应用场景、推理分析、综合比较和易混点辨析。

题型说明：
choice：单选题，options 提供 4 个选项，answer 写唯一正确选项。
multiple_choice：多选题，options 提供 4 个选项，answer 写所有正确选项。
true_false：判断题，options 为 ["正确", "错误"]，answer 写正确或错误。
fill：填空题，options 设为 null。
short_answer：简答题，options 设为 null。
compare：概念对比题，要求学生比较两个相关概念，options 设为 null。
application：应用分析题，给出简短场景，要求学生选择或解释合适方法，options 可为 null 或 4 个选项。

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
3. multiple_choice 题必须提供 4 个选项，answer 可以包含多个正确选项。
4. true_false 题 options 固定为 ["正确", "错误"]。
5. fill、short_answer、compare 题 options 设为 null。
6. application 题可根据需要提供 options，也可以设为 null。
7. explanation 要说明为什么答案正确。
8. source_chunk_ids 必须来自上方课程资料的来源 id；如果没有资料来源，使用空列表。
9. 请严格按照指定难度生成题目，不要明显过难或过易。
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


def _build_evaluation_prompt(req: QuizEvaluationRequest, search_results, triples: list[dict]) -> str:
    chunk_context = _format_chunks(search_results)
    kg_context = _format_triples(triples)

    return f"""
你是一个 NLP 课程助教。请根据课程资料和知识图谱，对一道自动生成的题目做质量自检。

知识点：
{req.concept}

目标难度：
{req.difficulty}

题型：
{req.type}

题目：
{req.question}

选项：
{json.dumps(req.options, ensure_ascii=False) if req.options else "无"}

参考答案：
{req.answer}

解析：
{req.explanation}

课程资料：
{chunk_context}

知识图谱关系：
{kg_context}

评估规则补充：
1. 只有 choice 和 multiple_choice 题需要检查是否提供 A/B/C/D 四个选项。
2. true_false 题只需要判断是否提供“正确/错误”或等价判断选项。
3. fill、short_answer、compare 题本来就可以没有选项，不要因为没有 A/B/C/D 而扣分。
4. application 题可以有选项，也可以没有选项，应根据题干要求判断。
5. JSON 的键名必须完全使用下方英文键名，不要改成中文键名或嵌套结构。

请严格返回 JSON，不要输出 Markdown，不要输出解释性前后缀。
JSON 格式如下：
{{
  "overall_score": 0.0,
  "clarity_score": 0.0,
  "correctness_score": 0.0,
  "difficulty_match_score": 0.0,
  "evidence_score": 0.0,
  "score_comments": {{
    "overall_score": "对总体质量的一句话评价",
    "clarity_score": "对题干清晰度的一句话评价",
    "correctness_score": "对答案正确性的一句话评价",
    "difficulty_match_score": "对难度匹配度的一句话评价",
    "evidence_score": "对依据充分性的一句话评价"
  }},
  "problems": ["问题1"],
  "suggestions": ["修改建议1"]
}}

评分说明：
1. 所有分数范围为 0 到 1。
2. clarity_score 评价题干和必要选项是否清楚、完整、无歧义。
3. correctness_score 评价答案和解析是否与课程资料、知识图谱一致。
4. difficulty_match_score 评价题目是否符合 easy/medium/hard 的目标难度。
5. evidence_score 评价题目是否有课程资料或知识图谱依据，引用越充分分数越高。
6. score_comments 必须为五个指标分别提供一句简短、具体、适合学生理解的评价。
7. problems 和 suggestions 如果没有内容，返回空列表。
""".strip()


def _build_optimization_prompt(req: QuizOptimizationRequest, search_results, triples: list[dict]) -> str:
    chunk_context = _format_chunks(search_results)
    kg_context = _format_triples(triples)

    return f"""
你是一个 NLP 课程助教。请根据“题目质量自检结果”自动优化这道题。

优化目标：
1. 保留原题考查的核心知识点、题型和目标难度。
2. 修复自检发现的问题，例如题干表述不清、选项不完整、答案不准确、解析不充分、依据不足等。
3. 优化后的题目必须能从课程资料或知识图谱中找到依据。
4. 如果原题已经较好，也要做轻量润色，让题干、答案和解析更清晰。

知识点：
{req.concept}

目标难度：
{req.difficulty}

题型：
{req.type}

原题：
{req.question}

原选项：
{json.dumps(req.options, ensure_ascii=False) if req.options else "无"}

原参考答案：
{req.answer}

原解析：
{req.explanation}

自检发现的问题：
{json.dumps(req.problems, ensure_ascii=False)}

自检修改建议：
{json.dumps(req.suggestions, ensure_ascii=False)}

自检评分说明：
{json.dumps(req.score_comments, ensure_ascii=False)}

课程资料：
{chunk_context}

知识图谱关系：
{kg_context}

请严格返回 JSON，不要输出 Markdown，不要输出解释性前后缀。JSON 格式如下：
{{
  "type": "{req.type}",
  "question": "优化后的题干",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": "优化后的参考答案",
  "explanation": "优化后的解析，说明答案为什么正确，并指出关键知识点",
  "source_chunk_ids": ["来源chunk_id"],
  "optimization_notes": ["本次优化做了什么"]
}}

格式要求：
1. choice 题必须提供 4 个选项，answer 写唯一正确选项。
2. multiple_choice 题必须提供 4 个选项，answer 写所有正确选项。
3. true_false 题 options 固定为 ["正确", "错误"]。
4. fill、short_answer、compare 题 options 设为 null。
5. application 题可根据题干需要提供 options，也可以设为 null。
6. source_chunk_ids 必须来自上方课程资料的来源 id；如果没有可靠来源，返回空列表。
7. optimization_notes 用 1 到 3 条短句说明本次自动优化修复了哪些问题。
""".strip()


def _build_submit_prompt(req: QuizSubmitRequest, search_results, triples: list[dict]) -> str:
    chunk_context = _format_chunks(search_results)
    kg_context = _format_triples(triples)

    return f"""
你是一个 NLP 课程助教。请根据参考答案、解析、课程资料和知识图谱，对学生作答进行语义判分。

知识点：
{req.concept}

题目：
{req.question}

学生答案：
{req.student_answer}

参考答案：
{req.reference_answer}

参考解析：
{req.explanation}

课程资料：
{chunk_context}

知识图谱关系：
{kg_context}

请严格返回 JSON，不要输出 Markdown，不要输出解释性前后缀。
JSON 格式如下：
{{
  "score": 0.0,
  "is_correct": false,
  "feedback": "给学生的具体反馈",
  "wrong_concepts": ["没有掌握或混淆的概念"],
  "suggested_review": ["建议复习的知识点"]
}}

评分要求：
1. score 范围为 0 到 1。
2. 如果学生答案表达不同但语义正确，应给较高分。
3. 如果答案只说到表面但缺少关键点，应给中等分并指出缺失。
4. feedback 要具体说明正确点、缺失点和改进建议。
5. wrong_concepts 和 suggested_review 应优先来自知识图谱关系或题目相关概念。
""".strip()


@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(req: QuizRequest):
    """Generate structured quiz questions for a concept."""
    record_concepts([req.concept])
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
    record_concepts([req.concept])
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


@router.post("/evaluate", response_model=QuizEvaluationResponse)
async def evaluate_quiz(req: QuizEvaluationRequest):
    """Evaluate generated quiz quality with course and KG evidence."""
    search_results = _search_course_chunks(req.concept, top_k=5)
    triples = _get_related_triples(req.concept, depth=1)
    prompt = _build_evaluation_prompt(req, search_results, triples)
    raw_answer = call_ecnu_gpt(prompt)
    data = _extract_json_object(raw_answer)

    return QuizEvaluationResponse(
        concept=req.concept,
        overall_score=_as_score(data.get("overall_score")),
        clarity_score=_as_score(data.get("clarity_score")),
        correctness_score=_as_score(data.get("correctness_score")),
        difficulty_match_score=_as_score(data.get("difficulty_match_score")),
        evidence_score=_as_score(data.get("evidence_score")),
        score_comments=_score_comments(data),
        problems=data.get("problems", []),
        suggestions=data.get("suggestions", []),
    )


@router.post("/optimize", response_model=QuizOptimizationResponse)
async def optimize_quiz(req: QuizOptimizationRequest):
    """Optimize one generated quiz item using quality-check feedback."""
    search_results = _search_course_chunks(req.concept, top_k=5)
    triples = _get_related_triples(req.concept, depth=1)
    prompt = _build_optimization_prompt(req, search_results, triples)
    raw_answer = call_ecnu_gpt(prompt)
    data = _extract_json_object(raw_answer)

    question_type = data.get("type", req.type)
    options = data.get("options")
    if question_type in {"fill", "short_answer", "compare"}:
        options = None
    elif question_type == "true_false":
        options = options or ["正确", "错误"]

    optimized_question = QuizItem(
        id=1,
        type=question_type,
        question=data.get("question", req.question),
        options=options,
        answer=data.get("answer", req.answer),
        explanation=data.get("explanation", req.explanation),
        source_chunk_ids=data.get("source_chunk_ids", []),
    )

    return QuizOptimizationResponse(
        concept=req.concept,
        optimized_question=optimized_question,
        optimization_notes=data.get("optimization_notes", []),
        source_chunk_ids=[result.chunk.chunk_id for result in search_results],
        related_triples=triples,
    )


@router.post("/submit", response_model=QuizSubmitResponse)
async def submit_quiz(req: QuizSubmitRequest):
    """Score a student's answer and recommend review concepts."""
    search_results = _search_course_chunks(req.concept, top_k=5)
    triples = _get_related_triples(req.concept, depth=1)
    prompt = _build_submit_prompt(req, search_results, triples)
    raw_answer = call_ecnu_gpt(prompt)
    data = _extract_json_object(raw_answer)

    suggested_review = _filter_review_concepts(data.get("suggested_review", []))
    if not suggested_review:
        suggested_review = _filter_review_concepts([
            triple["tail"]
            for triple in triples
            if triple.get("relation") in {"先修知识", "相关概念", "包含"}
        ])[:5]

    score = float(data.get("score", 0.0))
    is_correct = bool(data.get("is_correct", score >= 0.6))

    if score < 0.8 or not is_correct:
        record_mistake(
            concept=req.concept,
            question=req.question,
            student_answer=req.student_answer,
            reference_answer=req.reference_answer,
            score=score,
            feedback=data.get("feedback", ""),
            wrong_concepts=data.get("wrong_concepts", []),
            suggested_review=suggested_review,
        )

    return QuizSubmitResponse(
        concept=req.concept,
        score=score,
        is_correct=is_correct,
        feedback=data.get("feedback", ""),
        wrong_concepts=data.get("wrong_concepts", []),
        suggested_review=suggested_review,
        source_chunk_ids=[result.chunk.chunk_id for result in search_results],
        related_triples=triples,
    )
