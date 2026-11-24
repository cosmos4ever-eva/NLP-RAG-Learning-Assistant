SYSTEMS=["llm_only","tfidf_rag","graph_rag"]
def run_cases(cases,generate,retrieval_only=False):
    rows=[]
    for case in cases:
        for system in SYSTEMS:
            rows.append({"id":case["id"],"system":system,"answer":None if retrieval_only else generate(case,system)})
    return {"manifest":{"systems":list(SYSTEMS),"case_count":len(cases),"generation_requested":not retrieval_only},"rows":rows}
