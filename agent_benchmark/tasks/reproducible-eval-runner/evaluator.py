import json
from pathlib import Path
def evaluate(inputs):
    data=json.loads(Path(inputs["artifacts"]["submission"]).read_text(encoding="utf-8")); src=data.get("files",{}).get("runner.py"); checks={"source_present":isinstance(src,str),"public_tests_passed":bool(data.get("public_tests",{}).get("ok"))}
    if isinstance(src,str):
        try:
            ns={}; exec(compile(src,"runner.py","exec"),ns); cases=[{"id":"b"},{"id":"a"}]; calls=[]
            result=ns["run_cases"](cases,lambda c,s:calls.append((c["id"],s)) or "ok",retrieval_only=False)
            rows=result["rows"]; manifest=result["manifest"]
            expected=[(cid,s) for cid in ("b","a") for s in ("llm_only","tfidf_rag","graph_rag")]
            checks["three_system_matrix"]=[(r["id"],r["system"]) for r in rows]==expected and len(calls)==6
            blocked=[]; retrieval=ns["run_cases"](cases,lambda *a:blocked.append(True),retrieval_only=True)
            checks["retrieval_only_zero_calls"]=not blocked and all(r["answer"] is None for r in retrieval["rows"])
            checks["manifest_complete"]=manifest.get("systems")==["llm_only","tfidf_rag","graph_rag"] and manifest.get("case_count")==2 and manifest.get("generation_requested") is True
            checks["input_not_mutated"]=cases==[{"id":"b"},{"id":"a"}]
        except Exception: checks.update(three_system_matrix=False,retrieval_only_zero_calls=False,manifest_complete=False,input_not_mutated=False)
    passed=all(checks.values()); return {"status":"PASS" if passed else "FAIL","score":1.0 if passed else 0.0,"metrics":checks,"checks":checks}
