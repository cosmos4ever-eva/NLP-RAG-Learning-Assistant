import re
def tokens(text): return re.findall(r"[a-z]+|[\u4e00-\u9fff]{2,}",text.lower())
def search_context(chunks,query,top_k=3,neighbor_window=1,max_context=8):
    terms=set(tokens(query)); ranked=sorted(chunks,key=lambda c:len(terms&set(tokens(c["text"]))),reverse=True)
    return [{**c,"retrieval_reason":"semantic"} for c in ranked[:top_k]]
