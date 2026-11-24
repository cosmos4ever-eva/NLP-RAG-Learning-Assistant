def answer_question(question, retrieve, graph_lookup, generate):
    sources = retrieve(question)
    triples = graph_lookup(question)
    answer = generate(question, sources, triples)
    return {"status": "answered", "answer": answer, "sources": sources, "triples": triples}
