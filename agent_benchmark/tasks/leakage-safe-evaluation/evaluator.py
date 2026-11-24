import hashlib,importlib.util,json,sys,tempfile
from pathlib import Path
def _load(files):
    tmp=tempfile.TemporaryDirectory(); root=Path(tmp.name)
    for n,s in files.items():(root/n).write_text(s,encoding="utf-8")
    mods={}
    for name in ("runner","scorer"):
        spec=importlib.util.spec_from_file_location("candidate_safe_"+name,root/(name+".py")); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); mods[name]=mod
    return tmp,mods
def evaluate(inputs):
    data=json.loads(Path(inputs["artifacts"]["submission"]).read_text(encoding="utf-8")); files=data.get("files",{}); checks={"files_present":all(isinstance(files.get(x),str) for x in ("runner.py","scorer.py")),"public_tests_passed":bool(data.get("public_tests",{}).get("ok"))}
    if checks["files_present"]:
        try:
            tmp,m=_load({x:files[x] for x in ("runner.py","scorer.py")})
        except Exception:
            m=None
        if m is not None:
            cases=[{"id":"d1","split":"dev","question":"q","expected_answer_points":["SECRET"]},{"id":"t1","split":"test","question":"sealed","expected_answer_points":["TEST_SECRET"]}]; original=json.loads(json.dumps(cases)); seen=[]
            try: result=m["runner"].run_experiment(cases,"dev",lambda case,system:seen.append((dict(case),system)) or "answer")
            except Exception: result={"manifest":{},"answers":[]}
            checks["split_sealed"]=result.get("manifest",{}).get("case_count")==1 and {a.get("id") for a in result.get("answers",[])}=={"d1"}
            checks["gold_not_leaked"]=len(seen)==3 and all(set(c)=={"id","question"} and "SECRET" not in str(c) for c,s in seen)
            checks["matrix_complete_ordered"]=[a.get("system") for a in result.get("answers",[])]==["llm_only","tfidf_rag","graph_rag"]
            checks["input_not_mutated"]=cases==original
            score_cases=[{"id":"d1","legacy_override":{"answer_sha256":hashlib.sha256(b"old").hexdigest(),"correctness":4}}]; answers=[{"id":"d1","system":s,"answer":"new"} for s in ("llm_only","tfidf_rag","graph_rag")]; base=lambda c,a:{"correctness":1,"groundedness":1,"relation":1,"boundary":1}
            try:
                scored=m["scorer"].score_answers(score_cases,answers,base,apply_overrides=True); checks["stale_override_ignored"]=all(r.get("correctness")==1 and r.get("adjudication")!="matching_answer_override" and r.get("total")==4 for r in scored)
            except Exception: checks["stale_override_ignored"]=False
            try:
                score_cases[0]["legacy_override"]={"answer_sha256":hashlib.sha256(b"new").hexdigest(),"correctness":4}; matched=m["scorer"].score_answers(score_cases,[dict(x) for x in answers],base,apply_overrides=True); checks["matching_override_applied"]=all(r.get("correctness")==4 and r.get("total")==7 and r.get("adjudication")=="matching_answer_override" for r in matched)
            except Exception: checks["matching_override_applied"]=False
            failures=[]
            for bad in (answers[:-1],answers+[dict(answers[0])],answers+[{"id":"d1","system":"unknown","answer":"x"}]):
                try:m["scorer"].score_answers(score_cases,bad,base)
                except ValueError: failures.append(True)
                except Exception: pass
            checks["invalid_matrices_rejected"]=len(failures)==3; tmp.cleanup()
        else: checks.update(split_sealed=False,gold_not_leaked=False,matrix_complete_ordered=False,input_not_mutated=False,stale_override_ignored=False,matching_override_applied=False,invalid_matrices_rejected=False)
    passed=all(checks.values()); return {"status":"PASS" if passed else "FAIL","score":1.0 if passed else 0.0,"metrics":checks,"checks":checks}
