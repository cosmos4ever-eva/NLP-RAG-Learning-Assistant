"""
Learning analytics module.

Tracks asked concepts in memory and reports knowledge coverage.
"""
from fastapi import APIRouter
from pydantic import BaseModel

from knowledge_graph.graph import get_kg

router = APIRouter()

_asked_entities: list[str] = []
_mistake_records: list[dict] = []


class RecordQuestionRequest(BaseModel):
    question: str


class CoverageResponse(BaseModel):
    asked_entities: list[str]
    missing_entities: list[str]
    recommended_entities: list[str]
    coverage_rate: float


class LearningReportResponse(BaseModel):
    coverage_rate: float
    asked_count: int
    total_count: int
    mastered_concepts: list[str]
    weak_concepts: list[str]
    activity_profile: dict[str, int]
    ability_profile: list[dict]
    weak_concept_reasons: list[dict]
    recommended_next_steps: list[str]
    recommendation_basis: str
    summary: str


class RecommendedConcept(BaseModel):
    concept: str
    reason: str
    source: str


class RecommendedConceptsResponse(BaseModel):
    concepts: list[RecommendedConcept]


class MistakeRecordRequest(BaseModel):
    concept: str
    question: str
    student_answer: str
    reference_answer: str
    score: float
    feedback: str = ""
    wrong_concepts: list[str] = []
    suggested_review: list[str] = []


class MistakeBookResponse(BaseModel):
    count: int
    records: list[dict]


def extract_entities_from_question(question: str) -> list[str]:
    kg = get_kg()
    matched = []

    for entity in kg.get_all_entities():
        name = entity.get("name", "")
        entity_id = entity.get("id", "")

        if name and name in question:
            matched.append(name)
        elif entity_id and entity_id in question:
            matched.append(name)

    return matched


def record_question_entities(question: str) -> list[str]:
    entities = extract_entities_from_question(question)
    _asked_entities.extend(entities)
    return entities


def record_concepts(concepts: list[str]) -> list[str]:
    kg = get_kg()
    entity_names = {entity["name"] for entity in kg.get_all_entities()}
    entity_ids = {entity["id"]: entity["name"] for entity in kg.get_all_entities()}
    recorded = []

    for concept in concepts:
        if concept in entity_names:
            recorded.append(concept)
        elif concept in entity_ids:
            recorded.append(entity_ids[concept])

    _asked_entities.extend(recorded)
    return recorded


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys([item for item in items if item]))


def _related_concepts(concept: str, limit: int = 5) -> list[str]:
    kg = get_kg()
    related = []
    for triple in kg.get_neighbors(concept, depth=1):
        if triple.get("head") == concept:
            related.append(triple.get("tail", ""))
        elif triple.get("tail") == concept:
            related.append(triple.get("head", ""))
    return _unique(related)[:limit]


def record_mistake(
    concept: str,
    question: str,
    student_answer: str,
    reference_answer: str,
    score: float,
    feedback: str = "",
    wrong_concepts: list[str] | None = None,
    suggested_review: list[str] | None = None,
) -> dict:
    wrong_concepts = wrong_concepts or []
    suggested_review = suggested_review or []
    record_concepts([concept, *wrong_concepts, *suggested_review])
    record = {
        "id": len(_mistake_records) + 1,
        "concept": concept,
        "question": question,
        "student_answer": student_answer,
        "reference_answer": reference_answer,
        "score": round(float(score), 2),
        "feedback": feedback,
        "wrong_concepts": wrong_concepts,
        "suggested_review": suggested_review,
    }
    _mistake_records.append(record)
    return record


def recommend_concepts(limit: int = 8) -> list[RecommendedConcept]:
    kg = get_kg()
    all_entities = [entity["name"] for entity in kg.get_all_entities()]
    asked = _unique(_asked_entities)
    mistake_concepts = _unique(
        [
            concept
            for record in _mistake_records
            for concept in [
                record.get("concept", ""),
                *record.get("wrong_concepts", []),
                *record.get("suggested_review", []),
            ]
        ]
    )

    recommendations: list[RecommendedConcept] = []
    seen: set[str] = set()

    for concept in mistake_concepts:
        if concept not in seen:
            seen.add(concept)
            recommendations.append(
                RecommendedConcept(
                    concept=concept,
                    reason="该知识点出现在错题记录或建议复习中，建议优先回顾。",
                    source="mistake_book",
                )
            )
        for related in _related_concepts(concept, limit=3):
            if related not in seen:
                seen.add(related)
                recommendations.append(
                    RecommendedConcept(
                        concept=related,
                        reason=f"它与错题知识点「{concept}」存在图谱关联，适合一起复习。",
                        source="knowledge_graph",
                    )
                )
        if len(recommendations) >= limit:
            return recommendations[:limit]

    for concept in reversed(asked):
        for related in _related_concepts(concept, limit=3):
            if related not in seen:
                seen.add(related)
                recommendations.append(
                    RecommendedConcept(
                        concept=related,
                        reason=f"你最近关注过「{concept}」，该知识点与之相关，适合继续追问。",
                        source="question_history",
                    )
                )
            if len(recommendations) >= limit:
                return recommendations[:limit]

    for concept in all_entities:
        if concept not in asked and concept not in seen:
            seen.add(concept)
            recommendations.append(
                RecommendedConcept(
                    concept=concept,
                    reason="该知识点尚未覆盖，可作为下一步学习内容。",
                    source="coverage",
                )
            )
        if len(recommendations) >= limit:
            break

    return recommendations[:limit]


@router.post("/record")
async def record_question(req: RecordQuestionRequest):
    """Record concepts mentioned in a student question."""
    return {"recorded_entities": record_question_entities(req.question)}


@router.post("/mistakes")
async def add_mistake(req: MistakeRecordRequest):
    """Add a weakly answered question to the mistake book."""
    return record_mistake(
        concept=req.concept,
        question=req.question,
        student_answer=req.student_answer,
        reference_answer=req.reference_answer,
        score=req.score,
        feedback=req.feedback,
        wrong_concepts=req.wrong_concepts,
        suggested_review=req.suggested_review,
    )


@router.get("/mistakes", response_model=MistakeBookResponse)
async def get_mistakes():
    """Return the in-memory mistake book."""
    return MistakeBookResponse(count=len(_mistake_records), records=_mistake_records)


@router.get("/recommendations", response_model=RecommendedConceptsResponse)
async def get_recommendations(limit: int = 8):
    """Recommend concepts from question history, mistake book, and KG relations."""
    return RecommendedConceptsResponse(concepts=recommend_concepts(limit=limit))


@router.get("/coverage", response_model=CoverageResponse)
async def get_coverage():
    """Return simple knowledge coverage based on question history."""
    kg = get_kg()
    all_entities = [entity["name"] for entity in kg.get_all_entities()]
    asked_entities = sorted(set(_asked_entities))
    missing_entities = [entity for entity in all_entities if entity not in asked_entities]
    coverage_rate = len(asked_entities) / len(all_entities) if all_entities else 0.0

    return CoverageResponse(
        asked_entities=asked_entities,
        missing_entities=missing_entities,
        recommended_entities=missing_entities[:5],
        coverage_rate=round(coverage_rate, 4),
    )


@router.get("/report", response_model=LearningReportResponse)
async def get_learning_report():
    """Return a readable learning report based on concept coverage."""
    kg = get_kg()
    all_entities = [entity["name"] for entity in kg.get_all_entities()]
    mastered_concepts = sorted(set(_asked_entities))
    weak_concepts = [entity for entity in all_entities if entity not in mastered_concepts]
    coverage_rate = len(mastered_concepts) / len(all_entities) if all_entities else 0.0

    recommended_next_steps = []
    for concept in weak_concepts[:5]:
        prereqs = kg.get_prerequisites(concept)
        if prereqs:
            recommended_next_steps.append(
                f"学习「{concept}」前，建议先复习：{'、'.join(prereqs[:3])}。"
            )
        else:
            recommended_next_steps.append(f"建议继续学习未覆盖知识点：「{concept}」。")

    summary = (
        f"当前已覆盖 {len(mastered_concepts)}/{len(all_entities)} 个知识点，"
        f"覆盖率为 {coverage_rate:.2%}。"
    )

    if weak_concepts:
        summary += " 系统建议优先补充未覆盖知识点，并结合自动出题进行自测。"
    else:
        summary += " 已覆盖全部知识点，可以进入综合复习和难题训练。"

    return LearningReportResponse(
        coverage_rate=round(coverage_rate, 4),
        asked_count=len(mastered_concepts),
        total_count=len(all_entities),
        mastered_concepts=mastered_concepts,
        weak_concepts=weak_concepts[:10],
        recommended_next_steps=recommended_next_steps,
        summary=summary,
    )
