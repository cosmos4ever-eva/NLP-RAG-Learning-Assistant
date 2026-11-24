import json
from pathlib import Path
def evaluate(inputs):
    data=json.loads(Path(inputs["artifacts"]["submission"]).read_text(encoding="utf-8")); src=data.get("files",{}).get("api.py"); checks={"source_present":isinstance(src,str),"public_tests_passed":bool(data.get("public_tests",{}).get("ok"))}
    if isinstance(src,str):
        try:
            ns={}; exec(compile(src,"api.py","exec"),ns); result={"chunk_id":"c1","source":"L6","page":9}
            payload=ns["build_response"]("answer",[result],[{"head":"a","relation":"r","tail":"b"}],"high")
            checks["top_level_schema"]={"status","answer","sources","related_triples","evidence_level"}<=set(payload)
            checks["source_schema"]={"chunk_id","source","page","retrieval_reason"}<=set(payload["sources"][0]) and payload["sources"][0]["retrieval_reason"]=="semantic"
            checks["frontend_renders"]=isinstance(ns["render_response"](payload),str)
            checks["frontend_error_safe"]="down" in ns["render_response"]({"status":"error","error":"backend down"}).lower()
        except Exception: checks.update(top_level_schema=False,source_schema=False,frontend_renders=False,frontend_error_safe=False)
    passed=all(checks.values()); return {"status":"PASS" if passed else "FAIL","score":1.0 if passed else 0.0,"metrics":checks,"checks":checks}
