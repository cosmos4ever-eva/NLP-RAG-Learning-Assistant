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
    for index, result in enumerate(search_results, start=1):
        reason = "语义命中" if result.retrieval_reason == "semantic" else "相邻页上下文"
        lines.append(
            f"[S{index}] [来源: {result.chunk.chunk_id} | {reason} | 相似度: {result.similarity}]\n"
            f"{result.chunk.text}"
        )

    return "\n\n".join(lines)


def format_kg_context(triples: list[dict]) -> str:
    if not triples:
        return "未检索到相关知识图谱关系。"

    return "\n".join(
        f"[G{index}] {triple['head']} -{triple['relation']}-> {triple['tail']}"
        for index, triple in enumerate(triples, start=1)
    )


def build_evidence_items(search_results, max_excerpt_chars: int = 180) -> list[dict]:
    """Turn retrieval results into concise evidence cards for the product UI."""
    items = []
    for index, result in enumerate(search_results, start=1):
        excerpt = " ".join((result.chunk.text or "").split())
        if len(excerpt) > max_excerpt_chars:
            excerpt = excerpt[: max_excerpt_chars - 1].rstrip() + "…"
        items.append(
            {
                "evidence_id": f"S{index}",
                "chunk_id": result.chunk.chunk_id,
                "source": result.chunk.source,
                "page": result.chunk.page,
                "excerpt": excerpt,
                "retrieval_reason": result.retrieval_reason,
                "similarity": float(result.similarity),
            }
        )
    return items


def select_cited_evidence(
    answer: str,
    evidence_items: list[dict],
    triples: list[dict],
    max_items: int = 4,
) -> dict[str, list[dict]]:
    """Return only evidence explicitly cited in the generated answer."""
    cited_ids = []
    for evidence_id in re.findall(r"\b([SG]\d+)\b", answer or "", flags=re.IGNORECASE):
        normalized = evidence_id.upper()
        if normalized not in cited_ids:
            cited_ids.append(normalized)

    course_by_id = {
        str(item.get("evidence_id", "")).upper(): item
        for item in evidence_items
    }
    graph_by_id = {
        f"G{index}": {**triple, "evidence_id": f"G{index}"}
        for index, triple in enumerate(triples, start=1)
    }
    selected = [
        (evidence_id, course_by_id.get(evidence_id) or graph_by_id.get(evidence_id))
        for evidence_id in cited_ids
    ]
    selected = [(evidence_id, item) for evidence_id, item in selected if item][:max_items]

    return {
        "course": [item for evidence_id, item in selected if evidence_id.startswith("S")],
        "graph": [item for evidence_id, item in selected if evidence_id.startswith("G")],
    }


def normalize_answer_markdown(answer: str) -> str:
    """Keep generated section titles compact even when the model writes inline."""
    pattern = re.compile(
        r"(?m)^###\s*(结论|关键辨析|易错点)\s*[：:]\s*(\S.*)$"
    )
    return pattern.sub(lambda match: f"### {match.group(1)}\n{match.group(2)}", answer).strip()


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
6. 关键结论末尾必须标出直接支持它的证据编号，例如 [S1] 或 [G2]；只能使用上文实际存在的编号。
   全文最多使用 4 个不同的证据编号；只引用直接支持结论的证据，不要为了增加数量而堆叠编号。
7. 请按以下紧凑结构组织回答：
   ### 结论：标题必须单独占一行，下一行用不超过 120 个汉字直接回答问题。
   ### 关键辨析：标题必须单独占一行，下一行开始列最多 3 个要点；每点最多 2 句。
   ### 易错点：标题必须单独占一行，下一行最多 1 个要点；没有明显易错点时省略本节。
8. 不要重复同一结论，不要输出泛泛的“学习建议”，除非学生明确询问学习路径。
9. 不要单独复述完整课程资料或知识图谱；系统会在回答下方展示与证据编号对应的短摘录和关系。
10. 不要因为学生问题包含某个概念，就假定检索片段已经覆盖该概念。证据不足的子问题要简短指出缺口。
11. 如果问题涉及任务、模型与求解/解码方法，必须明确写出“任务 → 模型 → 算法”链条，且每一环都要有证据支持。
12. 对算法过程或概念差异，在证据允许时补充 1 个不超过 60 个汉字的具体小例子；没有直接证据时不要自行编造例子。
""".strip()


def evaluate_evidence(search_results, kg_triples: list[dict]) -> dict:
    """Estimate retrieval support without treating neighbor-page volume as proof."""
    source_count = len(search_results)
    triple_count = len(kg_triples)
    semantic_results = [
        result for result in search_results
        if result.retrieval_reason == "semantic"
    ]
    best_similarity = max(
        (float(result.similarity) for result in semantic_results),
        default=0.0,
    )

    source_score = min(best_similarity / 0.5, 1.0) * 0.65
    graph_score = min(triple_count / 5, 1.0) * 0.35
    score = round(source_score + graph_score, 2)

    # High support requires a strong course-material anchor. Graph volume and
    # automatically expanded neighbor pages cannot produce a high label alone.
    if best_similarity >= 0.3 and score >= 0.65:
        level = "high"
    elif best_similarity >= 0.15 or triple_count > 0:
        level = "medium"
    else:
        level = "low"

    summary = (
        f"{len(semantic_results)} 个语义锚点（最高相似度 {best_similarity:.2f}），"
        f"扩展后共 {source_count} 个课程片段、{triple_count} 条图谱关系；"
        f"检索支撑等级为 {level}。"
    )

    return {
        "evidence_score": score,
        "evidence_level": level,
        "evidence_summary": summary,
    }


def resolve_generation_provider(provider_name: str | None = None) -> dict[str, str]:
    requested = (provider_name or os.getenv("LEARNING_MODEL_PROVIDER", "")).strip().lower()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    ecnu_key = os.getenv("ECNUGPT_API_KEY", "").strip()
    if not requested:
        requested = "deepseek" if deepseek_key else "ecnu"

    if requested == "deepseek":
        if not deepseek_key:
            raise HTTPException(status_code=500, detail="缺少环境变量DEEPSEEK_API_KEY。")
        return {
            "name": "deepseek",
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            "api_key": deepseek_key,
        }

    if requested == "ecnu":
        if not ecnu_key:
            raise HTTPException(status_code=500, detail="缺少环境变量ECNUGPT_API_KEY。")
        return {
            "name": "ecnu",
            "base_url": ECNUGPT_BASE_URL,
            "model": os.getenv("ECNUGPT_MODEL", ECNUGPT_MODEL),
            "api_key": ecnu_key,
        }

    raise HTTPException(status_code=500, detail=f"不支持的生成模型供应商：{requested}")


def call_ecnu_gpt(prompt: str, provider_name: str | None = None) -> str:
    """Call the configured learning model; name retained for module compatibility."""
    provider = resolve_generation_provider(provider_name)
    url = f"{provider['base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {provider['api_key']}",
    }
    payload = {
        "model": provider["model"],
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业、耐心、清晰的NLP课程教学助手。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    if provider["name"] == "deepseek" and provider["model"] == "deepseek-flash":
        payload["thinking"] = {"type": "disabled"}
    else:
        payload["temperature"] = 0.3

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"{provider['name']}模型请求失败: {exc}",
        ) from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"{provider['name']}模型调用失败，状态码: {response.status_code}，"
                f"返回: {response.text}"
            ),
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
    answer = normalize_answer_markdown(call_ecnu_gpt(prompt))
    evidence_items = build_evidence_items(search_results)
    cited_evidence = select_cited_evidence(answer, evidence_items, kg_triples)

    return {
        "question": question,
        "answer": answer,
        "sources": [result.chunk.chunk_id for result in search_results],
        "evidence_items": evidence_items,
        "cited_evidence_items": cited_evidence["course"],
        "cited_triples": cited_evidence["graph"],
        "related_triples": kg_triples,
        "prompt": prompt,
        **evidence,
    }
