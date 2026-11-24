def make_part(question, status, answer=None, sources=None):
    return {"question": question, "status": status, "answer": answer, "sources": list(sources or [])}
