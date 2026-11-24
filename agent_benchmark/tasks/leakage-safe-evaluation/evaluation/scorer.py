import hashlib
SYSTEMS=("llm_only","tfidf_rag","graph_rag")
def score_answers(cases,answers,judge,apply_overrides=False):
    expected={(c["id"],s) for c in cases for s in SYSTEMS}; actual=[(a.get("id"),a.get("system")) for a in answers]
    if len(actual)!=len(set(actual)) or set(actual)!=expected: raise ValueError("incomplete_or_invalid_answer_matrix")
    by_id={c["id"]:c for c in cases}; rows=[]
    for answer in answers:
        case=by_id[answer["id"]]; scores=dict(judge(case,answer)); adjudication="model_assisted_first_pass"
        override=case.get("legacy_override") or {}; digest=hashlib.sha256(str(answer.get("answer","")).encode()).hexdigest()
        if apply_overrides and override.get("answer_sha256")==digest:
            scores.update({k:v for k,v in override.items() if k in ("correctness","groundedness","relation","boundary")}); adjudication="matching_answer_override"
        row={**answer,**scores,"adjudication":adjudication}; row["total"]=sum(int(row[k]) for k in ("correctness","groundedness","relation","boundary")); rows.append(row)
    return rows
