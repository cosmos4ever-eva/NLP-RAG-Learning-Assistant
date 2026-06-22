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
    weak_entities: list[str]
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
    weak_entities = _unique(
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
    missing_entities = [entity for entity in all_entities if entity not in asked_entities]
    coverage_rate = len(asked_entities) / len(all_entities) if all_entities else 0.0

    return CoverageResponse(
        asked_entities=asked_entities,
        missing_entities=missing_entities,
        weak_entities=weak_entities,
        recommended_entities=missing_entities[:5],
        coverage_rate=round(coverage_rate, 4),
    )


@router.get("/report", response_model=LearningReportResponse)
async def get_learning_report():
    """Return a diagnostic learning report, not just coverage statistics."""
    kg = get_kg()
    all_entities = [entity["name"] for entity in kg.get_all_entities()]
    mastered_concepts = sorted(set(_asked_entities))
    missing_concepts = [entity for entity in all_entities if entity not in mastered_concepts]
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
    weak_concepts = _unique([*mistake_concepts, *missing_concepts])
    coverage_rate = len(mastered_concepts) / len(all_entities) if all_entities else 0.0

    average_mistake_score = (
        sum(float(record.get("score", 0)) for record in _mistake_records) / len(_mistake_records)
        if _mistake_records
        else 1.0
    )

    if coverage_rate >= 0.7:
        coverage_level = "覆盖较充分"
    elif coverage_rate >= 0.35:
        coverage_level = "覆盖正在建立"
    else:
        coverage_level = "覆盖仍需扩展"

    if not _mistake_records:
        answer_level = "暂无错题记录，可继续通过自动出题积累诊断数据"
    elif average_mistake_score >= 0.8:
        answer_level = "答题稳定性较好，错题主要用于查漏补缺"
    elif average_mistake_score >= 0.6:
        answer_level = "存在部分理解不稳，建议结合解析复习相关概念"
    else:
        answer_level = "错题得分偏低，建议优先回到前序概念补基础"

    activity_profile = {
        "已触达知识点": len(mastered_concepts),
        "未覆盖知识点": len(missing_concepts),
        "错题记录": len(_mistake_records),
        "错题关联概念": len(mistake_concepts),
    }

    ability_profile = [
        {
            "维度": "知识覆盖",
            "判断": coverage_level,
            "依据": f"已覆盖 {len(mastered_concepts)}/{len(all_entities)} 个知识点。",
        },
        {
            "维度": "答题稳定性",
            "判断": answer_level,
            "依据": f"当前错题记录 {len(_mistake_records)} 条，平均得分 {average_mistake_score:.2f}。",
        },
        {
            "维度": "复习优先级",
            "判断": "优先处理错题相关概念，再补未覆盖概念",
            "依据": "错题概念更能反映当前理解薄弱处，未覆盖概念用于扩展课程地图。",
        },
    ]

    weak_concept_reasons = []
    for concept in mistake_concepts[:6]:
        related_records = [
            record
            for record in _mistake_records
            if concept == record.get("concept")
            or concept in record.get("wrong_concepts", [])
            or concept in record.get("suggested_review", [])
        ]
        reason = "该知识点出现在错题或建议复习中，说明当前掌握还不稳定。"
        if related_records:
            scores = [float(record.get("score", 0)) for record in related_records]
            reason = f"该知识点关联 {len(related_records)} 条错题记录，最低得分 {min(scores):.2f}。"
        weak_concept_reasons.append(
            {
                "知识点": concept,
                "原因": reason,
                "建议": "先复习概念定义和前序关系，再用自动出题做针对性练习。",
            }
        )

    for concept in missing_concepts:
        if len(weak_concept_reasons) >= 10:
            break
        if concept in {item["知识点"] for item in weak_concept_reasons}:
            continue
        prereqs = kg.get_prerequisites(concept)
        suggestion = "建议作为下一步扩展学习内容。"
        if prereqs:
            suggestion = f"学习前可先复习：{'、'.join(prereqs[:3])}。"
        weak_concept_reasons.append(
            {
                "知识点": concept,
                "原因": "该知识点在知识图谱中存在，但学习记录中尚未覆盖。",
                "建议": suggestion,
            }
        )

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
        summary += " 报告会优先结合错题和薄弱概念给出复习建议，而不是只展示覆盖列表。"
    else:
        summary += " 已覆盖全部知识点，可以进入综合复习和难题训练。"

    recommendation_basis = (
        "推荐学习知识点会综合四类依据：错题本中的低分或错误概念、学习历史中最近关注的概念、"
        "知识图谱中的前序/相关/包含关系，以及尚未覆盖的课程知识点。系统会优先推荐错题相关和前序基础，"
        "再补充未覆盖但重要的概念。"
    )

    return LearningReportResponse(
        coverage_rate=round(coverage_rate, 4),
        asked_count=len(mastered_concepts),
        total_count=len(all_entities),
        mastered_concepts=mastered_concepts,
        weak_concepts=weak_concepts[:10],
        activity_profile=activity_profile,
        ability_profile=ability_profile,
        weak_concept_reasons=weak_concept_reasons,
        recommended_next_steps=recommended_next_steps,
        recommendation_basis=recommendation_basis,
        summary=summary,
    )
