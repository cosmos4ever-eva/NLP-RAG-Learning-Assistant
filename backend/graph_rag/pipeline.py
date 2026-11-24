"""
Reusable Graph-RAG pipeline.

This module contains the real workflow used by the FastAPI route:
question -> course retrieval -> KG lookup -> prompt construction -> ECNUgpt.
"""
import os
import re

import requests
from fastapi import HTTPException

from knowledge_graph.graph import get_kg
from vector_search.retriever import get_retriever


ECNUGPT_BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
ECNUGPT_MODEL = os.getenv("ECNUGPT_MODEL", "ecnu-plus")


# 用户常用简称 -> 图谱中的规范实体名。别名只负责召回，不改变图谱事实。
ENTITY_ALIASES = {
    "文本预处理": ("预处理",),
    "分词/Tokenization": ("中文分词", "英文分词", "分词", "tokenization"),
    "分布式假设": ("distributional hypothesis",),
    "分布式表示": ("稠密表示", "稠密向量"),
    "One-hot编码": ("one-hot", "one hot", "独热编码", "独热"),
    "共现矩阵": ("词项-上下文矩阵", "词-上下文矩阵"),
    "点互信息(PMI)": ("ppmi", "pmi", "点互信息"),
}


def _contains_identifier(question: str, identifier: str) -> bool:
    """Case-insensitive identifier match without matching PMI inside unrelated words."""
    if not identifier:
        return False
    question_folded = question.casefold()
    identifier_folded = identifier.casefold()
    if re.fullmatch(r"[a-z0-9_-]+", identifier_folded):
        pattern = rf"(?<![a-z0-9_]){re.escape(identifier_folded)}(?![a-z0-9_])"
        return re.search(pattern, question_folded) is not None
    return identifier_folded in question_folded


def extract_entities(question: str, max_entities: int = 3) -> list[str]:
    kg = get_kg()
    matched = []

    for canonical_name, aliases in ENTITY_ALIASES.items():
        if any(alias.casefold() in question.casefold() for alias in aliases):
            matched.append(canonical_name)
            if len(matched) >= max_entities:
                return matched

    # 长名称优先，避免较短概念抢占 max_entities 名额。
    entities = sorted(
        kg.get_all_entities(),
        key=lambda entity: len(entity.get("name", "")),
        reverse=True,
    )
    for entity in entities:
        name = entity.get("name", "")
        entity_id = entity.get("id", "")

        if name in matched:
            continue
        if name and name.casefold() in question.casefold():
            matched.append(name)
        elif _contains_identifier(question, entity_id):
            matched.append(name)

        if len(matched) >= max_entities:
            break

    return matched


def extract_grounded_entities(
    question: str,
    context_texts: list[str] | None = None,
    max_entities: int = 4,
) -> list[str]:
    """Prefer explicit question entities, then fill gaps from retrieved course text."""
    matched = extract_entities(question, max_entities=min(3, max_entities))
    if not context_texts or len(matched) >= max_entities:
        return matched

    has_question_entities = bool(matched)
    related_to_question = set()
    if has_question_entities:
        kg = get_kg()
        for entity in matched:
            for triple in kg.get_neighbors(entity, depth=1):
                related_to_question.add(triple["head"])
                related_to_question.add(triple["tail"])

    context = "\n".join(text for text in context_texts if text)
    for entity in extract_entities(context, max_entities=max_entities):
        if entity not in matched and (not has_question_entities or entity in related_to_question):
            matched.append(entity)
        if len(matched) >= max_entities:
            break
    return matched


def get_kg_triples(
    question: str,
    depth: int = 1,
    context_texts: list[str] | None = None,
    max_triples: int = 15,
) -> list[dict]:
    kg = get_kg()
    triples = []
    seen = set()

    for entity in extract_grounded_entities(question, context_texts=context_texts):
        for triple in kg.get_neighbors(entity, depth=depth):
            key = (triple["head"], triple["relation"], triple["tail"])
            if key in seen:
                continue

            seen.add(key)
            triples.append(triple)
            if len(triples) >= max_triples:
                return triples

    return triples


def get_course_context(question: str, top_k: int = 3):
    retriever = get_retriever()
    try:
        return retriever.search_context(question, top_k=top_k, neighbor_window=2)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def format_ppt_context(search_results) -> str:
    if not search_results:
        return "未检索到相关课程资料。"

    lines = []
    for result in search_results:
        reason = "语义命中" if result.retrieval_reason == "semantic" else "相邻页上下文"
        lines.append(
            f"[来源: {result.chunk.chunk_id} | {reason} | 相似度: {result.similarity}]\n"
            f"{result.chunk.text}"
        )

    return "\n\n".join(lines)


def format_kg_context(triples: list[dict]) -> str:
    if not triples:
        return "未检索到相关知识图谱关系。"

    return "\n".join(
        f"{triple['head']} -{triple['relation']}-> {triple['tail']}"
        for triple in triples
    )


def build_prompt(
    question: str,
    search_results,
    kg_triples: list[dict],
    allow_model_knowledge: bool = False,
) -> str:
    ppt_context = format_ppt_context(search_results)
    kg_context = format_kg_context(kg_triples)

    if allow_model_knowledge:
        grounding_policy = (
            "课程证据不足时，可以使用模型已有知识回答，但必须明确标注哪些内容不来自课程资料。"
        )
    else:
        grounding_policy = (
            "只能依据下方课程资料和知识图谱回答。先判断证据是否直接支持问题；检索到片段不等于片段相关。"
            "如果问题包含多个部分，要逐个子问题判断证据：有证据的部分正常回答，"
            "只对缺少证据的部分说明资料不足，不要因局部缺证据而拒答整题。"
            "如果证据不足，请明确说明当前课程资料无法回答，并指出需要补充什么资料；不要继续给出课外公式、"
            "推荐、定量结果或看似合理的推断。"
        )

    return f"""
你是一个 NLP 课程智能教学助手。请结合课程资料和知识图谱回答学生问题。

学生问题：
{question}

课程资料检索结果：
{ppt_context}

知识图谱相关关系：
{kg_context}

回答要求：
1. {grounding_policy}
2. 如果课程资料和知识图谱信息不足，请明确说明，不要编造。
3. 使用适合学生理解的语言，解释清楚概念。
4. 回答要有条理，可以分点说明。
5. 如果涉及多个概念，请说明它们之间的关系。
6. 不要在正文句子后面插入 [课程资料: ...] 或 [知识图谱: ...] 这类证据标签。
7. 请按以下结构组织回答：
   一、简明回答：直接回答学生问题，用自然语言解释清楚。
   二、概念逻辑链条：把相关知识整理成“理论基础 -> 核心概念 -> 方法/模型 -> 训练技巧/应用”的学习顺序。
   三、学习建议：给出一个简短的后续学习建议。
8. “概念逻辑链条”部分不要在每一行后堆叠证据标签，只需要说明概念之间的逻辑关系。
9. 不要单独输出“依据说明”“课程资料”“知识图谱”小节；系统会在回答下方用表格和图谱可视化单独展示来源与关系。
10. 不要因为学生问题包含某个概念，就假定检索片段已经覆盖该概念。
""".strip()


def evaluate_evidence(search_results, kg_triples: list[dict]) -> dict:
    """Score how well the answer is grounded by retrieved evidence."""
    source_count = len(search_results)
    triple_count = len(kg_triples)

    source_score = min(source_count / 3, 1.0) * 0.55
    graph_score = min(triple_count / 5, 1.0) * 0.45
    score = round(source_score + graph_score, 2)

    if score >= 0.75:
        level = "high"
    elif score >= 0.4:
        level = "medium"
    else:
        level = "low"

    summary = (
        f"检索到 {source_count} 个课程资料片段，"
        f"{triple_count} 条知识图谱关系，证据充分性为 {level}。"
    )

    return {
        "evidence_score": score,
        "evidence_level": level,
        "evidence_summary": summary,
    }


def call_ecnu_gpt(prompt: str) -> str:
    api_key = os.getenv("ECNUGPT_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="缺少环境变量 ECNUGPT_API_KEY，请先在启动后端的终端中设置 API Key。",
        )

    url = f"{ECNUGPT_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": ECNUGPT_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "你是华东师范大学大模型 ChatECNU，也是一个专业、耐心、清晰的 NLP 课程教学助手。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.3,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"ECNUgpt 请求失败: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"ECNUgpt 调用失败，状态码: {response.status_code}，返回: {response.text}",
        )

    data = response.json()
    return data["choices"][0]["message"]["content"]


def answer_question(question: str, top_k: int = 3, depth: int = 1) -> dict:
    search_results = get_course_context(question, top_k=top_k)
    kg_triples = get_kg_triples(
        question,
        depth=depth,
        context_texts=[result.chunk.text for result in search_results],
    )
    evidence = evaluate_evidence(search_results, kg_triples)
    prompt = build_prompt(question, search_results, kg_triples)
    answer = call_ecnu_gpt(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": [result.chunk.chunk_id for result in search_results],
        "related_triples": kg_triples,
        "prompt": prompt,
        **evidence,
    }
