SYSTEMS=["llm_only","tfidf_rag","graph_rag"]
def run_experiment(cases,split,generate):
    selected=[c for c in cases if c.get("split")==split]; answers=[]
    for case in selected:
        safe={"id":case["id"],"question":case["question"]}
        for system in SYSTEMS: answers.append({"id":case["id"],"system":system,"answer":generate(dict(safe),system)})
    return {"manifest":{"split":split,"case_count":len(selected),"systems":list(SYSTEMS)},"answers":answers}
