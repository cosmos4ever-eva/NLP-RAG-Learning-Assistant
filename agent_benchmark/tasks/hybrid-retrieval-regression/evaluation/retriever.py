import math,re
from config import STOPWORDS,EXPANSIONS,BROAD_TERMS
def tokens(text):
    raw=re.findall(r"[a-z]+|[\u4e00-\u9fff]{2,}",text.lower())
    return [x[:-1] if x.isascii() and x.endswith("s") and len(x)>3 and x not in {"does"} else x for x in raw]
def query_terms(text):
    terms=[x for x in tokens(text) if x not in STOPWORDS]; low=text.lower()
    for key,values in EXPANSIONS.items():
        if key in low: terms.extend(y for value in values for y in tokens(value))
    for specific,broad in BROAD_TERMS.items():
        if specific in terms: terms=[x for x in terms if x not in broad]
    return list(dict.fromkeys(terms))
def search_context(chunks,query,top_k=3,neighbor_window=1,max_context=8):
    docs=[set(tokens(c["text"])) for c in chunks]; df={t:sum(t in d for d in docs) for d in docs for t in d}
    def rank(text):
        q=query_terms(text); scores=[]
        for i,doc in enumerate(docs):
            matched=[t for t in q if t in doc]; rarity=sum(math.log((len(docs)+1)/(df.get(t,0)+1))+1 for t in matched); coverage=len(matched)/max(len(q),1); scores.append((coverage+rarity/max(len(q),1),i))
        return [chunks[i] for score,i in sorted(scores,reverse=True) if score>0]
    clauses=[x.strip() for x in re.split(r"[?？;；。]+",query) if query_terms(x)]; anchors=[]
    for clause in clauses:
        candidate=next((c for c in rank(clause) if c["id"] not in {a["id"] for a in anchors}),None)
        if candidate: anchors.append(candidate)
        if len(anchors)>=top_k: break
    for c in rank(query):
        if len(anchors)>=top_k: break
        if c["id"] not in {a["id"] for a in anchors}: anchors.append(c)
    out=[{**c,"retrieval_reason":"semantic"} for c in anchors]; seen={c["id"] for c in anchors}
    for a in anchors:
        for c in chunks:
            if len(out)>=max_context: break
            if c["id"] not in seen and c["source"]==a["source"] and 0<abs(c["page"]-a["page"])<=neighbor_window: out.append({**c,"retrieval_reason":"neighbor"}); seen.add(c["id"])
    return out
