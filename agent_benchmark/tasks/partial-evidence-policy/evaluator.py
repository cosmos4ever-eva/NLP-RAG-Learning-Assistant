import importlib.util,json,sys,tempfile
from pathlib import Path
def _load(files):
    tmp=tempfile.TemporaryDirectory(); root=Path(tmp.name)
    for name,source in files.items(): (root/name).write_text(source,encoding="utf-8")
    sys.path.insert(0,str(root)); sys.modules.pop("schemas",None)
    spec=importlib.util.spec_from_file_location("candidate_partial_pipeline",root/"pipeline.py"); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return tmp,mod
def evaluate(inputs):
    data=json.loads(Path(inputs["artifacts"]["submission"]).read_text(encoding="utf-8")); files=data.get("files",{}); checks={"files_present":all(isinstance(files.get(x),str) for x in ("pipeline.py","schemas.py")),"public_tests_passed":bool(data.get("public_tests",{}).get("ok"))}
    if checks["files_present"]:
        try:
            tmp,mod=_load({x:files[x] for x in ("pipeline.py","schemas.py")}); questions=["known","unknown","known2"]; original=list(questions); retrieval=[]; generated=[]
            def retrieve(q): retrieval.append(q); return [q+"-page"] if q.startswith("known") else []
            def generate(q,s): generated.append((q,list(s))); return "answer:"+q
            mixed=mod.answer_parts(questions,retrieve,generate); parts=mixed.get("parts",[])
            checks["partial_status_and_order"]=mixed.get("status")=="partial" and [p.get("question") for p in parts]==questions
            checks["per_part_contract"]=len(parts)==3 and parts[0].get("status")=="answered" and parts[1].get("status")=="insufficient_evidence" and parts[1].get("answer") is None and bool(parts[1].get("missing_material")) and all({"question","status","answer","sources","missing_material"}<=set(p) for p in parts)
            checks["call_boundaries"]=retrieval==questions and [x[0] for x in generated]==["known","known2"]
            checks["input_not_mutated"]=questions==original
            no_calls=[]; none=mod.answer_parts(["x"],lambda q:[],lambda *a:no_calls.append(True)); checks["none_supported_safe"]=none.get("status")=="insufficient_evidence" and not no_calls
            tmp.cleanup()
        except Exception: checks.update(partial_status_and_order=False,per_part_contract=False,call_boundaries=False,input_not_mutated=False,none_supported_safe=False)
    passed=all(checks.values()); return {"status":"PASS" if passed else "FAIL","score":1.0 if passed else 0.0,"metrics":checks,"checks":checks}
