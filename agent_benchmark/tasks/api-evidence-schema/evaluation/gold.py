def build_response(answer,results,triples,evidence_level):
    sources=[]
    for item in results:
        sources.append({"chunk_id":item["chunk_id"],"source":item["source"],"page":item["page"],"retrieval_reason":item.get("retrieval_reason","semantic")})
    return {"status":"ok","answer":answer,"sources":sources,"related_triples":triples,"evidence_level":evidence_level}
def render_response(payload):
    if payload.get("status")=="error": return f"Error: {payload.get('error','unknown error')}"
    lines=[payload.get("answer","")]
    for source in payload.get("sources",[]): lines.append(f"{source['source']} p.{source['page']} ({source['retrieval_reason']})")
    return "\n".join(lines)
