from kg_query import KnowledgeGraph


def extract_keywords(question):
    """
    先用最简单的方法：手动/规则提取关键词。
    后面可以换成 jieba、HanLP、LLM 抽取。
    """
    candidates = ["苹果", "水果", "计算机", "人工智能", "机器学习", "知识图谱"]

    keywords = []
    for word in candidates:
        if word in question:
            keywords.append(word)

    return keywords


def build_kg_context(question, topk=5):
    kg = KnowledgeGraph()
    keywords = extract_keywords(question)

    all_facts = []

    for keyword in keywords:
        facts = kg.query_as_text(keyword, topk=topk)
        if facts:
            all_facts.append(f"和「{keyword}」相关的知识：\n{facts}")

    return "\n\n".join(all_facts)


def build_prompt(question):
    kg_context = build_kg_context(question)

    prompt = f"""
你是一个智能教学助手。请结合知识图谱中的相关知识回答用户问题。

用户问题：
{question}

知识图谱相关知识：
{kg_context}

请给出清晰、准确、适合学生理解的回答。
"""
    return prompt


if __name__ == "__main__":
    question = "苹果是什么？它可以用来做什么？"

    prompt = build_prompt(question)
    print(prompt)