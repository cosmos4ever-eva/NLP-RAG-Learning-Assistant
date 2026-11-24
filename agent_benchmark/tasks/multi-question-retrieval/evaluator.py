import json
from pathlib import Path
def evaluate(inputs):
    data=json.loads(Path(inputs["artifacts"]["submission"]).read_text(encoding="utf-8")); src=data.get("files",{}).get("retriever.py"); checks={"source_present":isinstance(src,str),"public_tests_passed":bool(data.get("public_tests",{}).get("ok"))}
    if isinstance(src,str):
        try:
            ns={}; exec(compile(src,"retriever.py","exec"),ns)
            chunks=[{"id":"a1","source":"A","page":1,"text":"alpha alpha 定义"},{"id":"a2","source":"A","page":2,"text":"alpha 补充"},{"id":"b10","source":"B","page":10,"text":"beta 原因"},{"id":"b11","source":"B","page":11,"text":"beta 细节"}]
            got=ns["search_context"](chunks,"alpha是什么？beta为什么重要？",top_k=2,neighbor_window=1,max_context=6); ids=[x["id"] for x in got]
            checks["each_clause_has_anchor"]="a1" in ids and "b10" in ids and next(x for x in got if x["id"]=="a1")["retrieval_reason"]=="semantic" and next(x for x in got if x["id"]=="b10")["retrieval_reason"]=="semantic"
            checks["neighbors_added"]="a2" in ids and "b11" in ids
            checks["unique_and_bounded"]=len(ids)==len(set(ids)) and len(ids)<=6
        except Exception: checks.update(each_clause_has_anchor=False,neighbors_added=False,unique_and_bounded=False)
    passed=all(checks.values()); return {"status":"PASS" if passed else "FAIL","score":1.0 if passed else 0.0,"metrics":checks,"checks":checks}
