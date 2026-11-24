def answer_question(question,retrieve,graph_lookup,generate):
    sources=retrieve(question); triples=graph_lookup(question)
    if not sources and not triples:
        return {"status":"insufficient_evidence","answer":"当前课程证据不足，无法回答；需要补充直接讲解该概念的课程页面。","sources":[],"triples":[]}
    return {"status":"answered","answer":generate(question,sources,triples),"sources":sources,"triples":triples}
