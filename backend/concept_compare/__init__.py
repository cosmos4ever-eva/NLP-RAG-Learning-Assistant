"""
Concept comparison module.
"""
import re

from fastapi import APIRouter
from pydantic import BaseModel

from graph_rag.pipeline import (
    call_ecnu_gpt,
    format_kg_context,
    format_ppt_context,
    get_course_context,
    get_kg_triples,
)
from knowledge_graph.graph import get_kg

try:
    from learning_analytics import _asked_entities, record_concepts
except ImportError:
    _asked_entities = []
    record_concepts = None

router = APIRouter()


def _remove_inline_source_tags(text: str) -> str:
    """Keep the comparison readable; sources are shown in the table below."""
    text = re.sub(r"\s*\[来源[:：][^\]]+\]", "", text)
    text = re.sub(r"\s*【来源[:：][^】]+】", "", text)
    text = re.sub(r"\s*（来源[:：][^）]+）", "", text)
    return text.strip()


class CompareRequest(BaseModel):
    concept_a: str
    concept_b: str


class CompareResponse(BaseModel):
    concept_a: str
    concept_b: str
    answer: str
    source_chunk_ids: list[str]
    related_triples: list[dict]


class ComparePair(BaseModel):
    concept_a: str
    concept_b: str
    reason: str
    source: str


class CompareRecommendResponse(BaseModel):
    pairs: list[ComparePair]


def _add_pair(
    pairs: list[ComparePair],
    seen: set[tuple[str, str]],
    concept_a: str,
    concept_b: str,
    reason: str,
    source: str,
):
    if not concept_a or not concept_b or concept_a == concept_b:
        return

    key = tuple(sorted((concept_a, concept_b)))
    if key in seen:
        return

    seen.add(key)
    pairs.append(
        ComparePair(
            concept_a=concept_a,
            concept_b=concept_b,
            reason=reason,
            source=source,
        )
    )


def _recommend_pairs_from_kg(limit: int = 8) -> list[ComparePair]:
    kg = get_kg()
    triples = kg.get_all_triples()
    pairs: list[ComparePair] = []
    seen: set[tuple[str, str]] = set()

    related = {"相关概念", "容易混淆"}
    for triple in triples:
        relation = triple.get("relation", "")
        if relation in related:
            _add_pair(
                pairs,
                seen,
                triple.get("head", ""),
                triple.get("tail", ""),
                f"二者在知识图谱中存在“{relation}”关系，适合做概念辨析。",
                "knowledge_graph",
            )

    children_by_parent: dict[str, list[str]] = {}
    for triple in triples:
        if triple.get("relation") == "包含":
            children_by_parent.setdefault(triple.get("head", ""), []).append(triple.get("tail", ""))

    for parent, children in children_by_parent.items():
        unique_children = list(dict.fromkeys(children))
        for index, concept_a in enumerate(unique_children):
            for concept_b in unique_children[index + 1:]:
                _add_pair(
                    pairs,
                    seen,
                    concept_a,
                    concept_b,
                    f"二者同属于“{parent}”下的知识点，学生容易混淆它们的功能边界。",
                    "knowledge_graph",
                )
                if len(pairs) >= limit:
                    return pairs

    return pairs[:limit]


def _recommend_pairs_from_history(limit: int = 5) -> list[ComparePair]:
    kg = get_kg()
    pairs: list[ComparePair] = []
    seen: set[tuple[str, str]] = set()

    for concept in reversed(list(dict.fromkeys(_asked_entities))):
        for triple in kg.get_neighbors(concept, depth=1):
            relation = triple.get("relation", "")
            if relation not in {"相关概念", "容易混淆", "先修知识", "包含"}:
                continue

            other = triple["tail"] if triple.get("head") == concept else triple.get("head")
            _add_pair(
                pairs,
                seen,
                concept,
                other,
                f"你最近关注过“{concept}”，它和“{other}”存在“{relation}”关系，适合进一步辨析。",
                "history",
            )
            if len(pairs) >= limit:
                return pairs

    return pairs


@router.get("/recommend", response_model=CompareRecommendResponse)
async def recommend_compare_pairs(limit: int = 8):
    """Recommend confusing concept pairs from question history and the KG."""
    pairs: list[ComparePair] = []
    seen: set[tuple[str, str]] = set()

    for pair in _recommend_pairs_from_history(limit=limit):
        _add_pair(pairs, seen, pair.concept_a, pair.concept_b, pair.reason, pair.source)

    for pair in _recommend_pairs_from_kg(limit=limit):
        _add_pair(pairs, seen, pair.concept_a, pair.concept_b, pair.reason, pair.source)
        if len(pairs) >= limit:
            break

    return CompareRecommendResponse(pairs=pairs[:limit])


@router.post("/concepts", response_model=CompareResponse)
async def compare_concepts(req: CompareRequest):
    """Compare two confusing or related NLP concepts."""
    if record_concepts:
        record_concepts([req.concept_a, req.concept_b])
    query = f"{req.concept_a} 和 {req.concept_b} 的区别"
    chunks = get_course_context(query, top_k=5)
    triples = get_kg_triples(req.concept_a, depth=1) + get_kg_triples(req.concept_b, depth=1)

    prompt = f"""
你是一个 NLP 课程智能教学助手。请把“概念对比”做成辨析训练，而不是普通问答。

概念 A：
{req.concept_a}

概念 B：
{req.concept_b}

课程资料：
{format_ppt_context(chunks)}

知识图谱关系：
{format_kg_context(triples)}

请严格按以下结构输出：

### 一句话区分
用 1-2 句话说明二者最核心的区别。

### 对比表格
用 Markdown 表格对比：定义、核心思想、输入/输出或处理对象、适用场景、优点、局限。

### 为什么容易混淆
说明学生为什么容易把这两个概念混在一起。

### 判断规则
给出 2-4 条“看到什么关键词/场景就选哪个概念”的判断规则。

### 辨析小题
出 1 道简短应用判断题，并给出参考答案和一句解析。

要求：
- 优先依据课程资料和知识图谱。
- 不要在正文中使用 [来源: ...]、【来源: ...】 或类似行内来源标注。
- 课程资料来源会由系统在下方表格单独展示，正文只需要给出清晰对比。
- 输出要像教学辨析材料，重点突出“区别、易混原因、判断规则”，不要写成普通概念解释。
- 如果资料不足，请明确说明。
""".strip()

    answer = _remove_inline_source_tags(call_ecnu_gpt(prompt))

    return CompareResponse(
        concept_a=req.concept_a,
        concept_b=req.concept_b,
        answer=answer,
        source_chunk_ids=[result.chunk.chunk_id for result in chunks],
        related_triples=triples,
    )
