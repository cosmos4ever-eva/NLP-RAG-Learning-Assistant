def build_response(answer, results, triples, evidence_level):
    return {"answer": answer, "sources": [item["chunk_id"] for item in results], "triples": triples}
def render_response(payload):
    lines=[payload["answer"]]
    for source in payload["sources"]:
        lines.append(f"{source['source']} p.{source['page']} ({source['retrieval_reason']})")
    return "\n".join(lines)
