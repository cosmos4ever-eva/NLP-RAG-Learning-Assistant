"""
Reusable Graph-RAG pipeline.

This module contains the real workflow used by the FastAPI route:
question -> course retrieval -> KG lookup -> prompt construction -> ECNUgpt.
"""
import os

import requests
from fastapi import HTTPException

from knowledge_graph.graph import get_kg
from vector_search.retriever import get_retriever


ECNUGPT_BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
ECNUGPT_MODEL = os.getenv("ECNUGPT_MODEL", "ecnu-plus")


def extract_entities(question: str, max_entities: int = 3) -> list[str]:
    kg = get_kg()
    matched = []

    for entity in kg.get_all_entities():
        name = entity.get("name", "")
        entity_id = entity.get("id", "")

        if name and name in question:
            matched.append(name)
        elif entity_id and entity_id in question:
            matched.append(name)

        if len(matched) >= max_entities:
            break

    return matched


def get_kg_triples(question: str, depth: int = 1) -> list[dict]:
    kg = get_kg()
    triples = []
    seen = set()

    for entity in extract_entities(question):
        for triple in kg.get_neighbors(entity, depth=depth):
            key = (triple["head"], triple["relation"], triple["tail"])
            if key in seen:
                continue

            seen.add(key)
            triples.append(triple)

    return triples


def get_course_context(question: str, top_k: int = 3):
    retriever = get_retriever()
    try:
        return retriever.search(question, top_k=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def format_ppt_context(search_results) -> str:
    if not search_results:
        return "未检索到相关课程资料。"

    lines = []
    for result in search_results:
        lines.append(
            f"[来源: {result.chunk.chunk_id} | 相似度: {result.similarity}]\n"
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


def build_prompt(question: str, search_results, kg_triples: list[dict]) -> str:
    ppt_context = format_ppt_context(search_results)
    kg_context = format_kg_context(kg_triples)

    return f"""
你是一个 NLP 课程智能教学助手。请结合课程资料和知识图谱回答学生问题。

学生问题：
{question}

课程资料检索结果：
{ppt_context}

知识图谱相关关系：
{kg_context}

回答要求：
1. 优先依据课程资料和知识图谱回答。
2. 如果课程资料和知识图谱信息不足，请明确说明，不要编造。
3. 使用适合学生理解的语言，解释清楚概念。
4. 回答要有条理，可以分点说明。
5. 如果涉及多个概念，请说明它们之间的关系。
6. 每个主要结论后都要标注依据。
7. 课程资料依据格式为：[课程资料: chunk_id]。
8. 知识图谱依据格式为：[知识图谱: 头实体-关系-尾实体]。
9. 如果某个判断没有依据，请不要写成确定结论。
10. 最后给出一个简短的学习建议。
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
    kg_triples = get_kg_triples(question, depth=depth)
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
