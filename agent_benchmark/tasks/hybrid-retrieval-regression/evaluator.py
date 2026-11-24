import importlib.util,json,sys,tempfile
from pathlib import Path
def _load(files):
    tmp=tempfile.TemporaryDirectory(); root=Path(tmp.name)
    for n,s in files.items():(root/n).write_text(s,encoding="utf-8")
    sys.path.insert(0,str(root)); sys.modules.pop("config",None); spec=importlib.util.spec_from_file_location("candidate_hybrid",root/"retriever.py"); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return tmp,mod
def evaluate(inputs):
    data=json.loads(Path(inputs["artifacts"]["submission"]).read_text(encoding="utf-8")); files=data.get("files",{}); checks={"files_present":all(isinstance(files.get(x),str) for x in ("retriever.py","config.py")),"public_tests_passed":bool(data.get("public_tests",{}).get("ok"))}
    if checks["files_present"]:
        try:
            tmp,m=_load({x:files[x] for x in ("retriever.py","config.py")})
            chunks=[{"id":"dir","source":"L2","page":2,"text":"text preprocessing normalization stemming tokenization"},{"id":"norm","source":"L2","page":20,"text":"normalization common problems lowercase contraction hyphen"},{"id":"norm-neighbor","source":"L2","page":21,"text":"normalization worked examples"},{"id":"align","source":"L2","page":41,"text":"edit distance character alignment"},{"id":"trace","source":"L2","page":55,"text":"edit distance is not enough record source backtrace path alignment"},{"id":"dh","source":"L6","page":9,"text":"distributional hypothesis similar words appear in similar context"},{"id":"generic","source":"L4","page":1,"text":"what is classification context"},{"id":"alpha","source":"A","page":1,"text":"alpha definition"},{"id":"beta","source":"B","page":10,"text":"beta reason"}]
            def ids(q,**kw): return [x["id"] for x in m.search_context(chunks,q,**kw)]
            checks["specific_over_directory"]=ids("Why does text preprocessing need normalization and give a typical problem",top_k=1,neighbor_window=0)[0]=="norm"
            checks["synonym_expansion"]=ids("Why is edit distance insufficient for alignment and what information should the system save?",top_k=1,neighbor_window=0)[0]=="trace"
            checks["question_words_removed"]=ids("What is the distributional hypothesis and what does context mean?",top_k=1,neighbor_window=0)[0]=="dh"
            multi=m.search_context(chunks,"What is alpha? Why beta?",top_k=2,neighbor_window=0,max_context=2); checks["multi_clause_coverage"]={x["id"] for x in multi}=={"alpha","beta"} and all(x["retrieval_reason"]=="semantic" for x in multi)
            neighbor=m.search_context(chunks,"normalization typical problem",top_k=1,neighbor_window=1,max_context=2); checks["neighbor_and_budget"]=[x["id"] for x in neighbor]==["norm","norm-neighbor"]
            tmp.cleanup()
        except Exception: checks.update(specific_over_directory=False,synonym_expansion=False,question_words_removed=False,multi_clause_coverage=False,neighbor_and_budget=False)
    passed=all(checks.values()); return {"status":"PASS" if passed else "FAIL","score":1.0 if passed else 0.0,"metrics":checks,"checks":checks}
