import re
def _tokens(text): return set(re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]{2,}", text.lower()))
def search_context(chunks, query, top_k=2, neighbor_window=1, max_context=6):
    terms=_tokens(query)
    ranked=sorted(chunks,key=lambda c:len(terms & _tokens(c["text"])),reverse=True)
    return [{**chunk,"retrieval_reason":"semantic"} for chunk in ranked[:top_k]]
