import re
def _tokens(text): return set(re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]{2,}",text.lower()))
def search_context(chunks,query,top_k=2,neighbor_window=1,max_context=6):
    clauses=[x.strip() for x in re.split(r"[？?；;。]+",query) if x.strip()]
    anchors=[]
    for clause in clauses:
        terms=_tokens(clause); ranked=sorted(chunks,key=lambda c:len(terms&_tokens(c["text"])),reverse=True)
        candidate=next((c for c in ranked if len(terms&_tokens(c["text"]))>0 and c["id"] not in {a["id"] for a in anchors}),None)
        if candidate: anchors.append(candidate)
        if len(anchors)>=top_k: break
    results=[{**c,"retrieval_reason":"semantic"} for c in anchors]
    seen={c["id"] for c in anchors}
    for anchor in anchors:
        for c in chunks:
            if len(results)>=max_context: break
            if c["id"] not in seen and c["source"]==anchor["source"] and 0<abs(c["page"]-anchor["page"])<=neighbor_window:
                results.append({**c,"retrieval_reason":"neighbor"}); seen.add(c["id"])
    return results
