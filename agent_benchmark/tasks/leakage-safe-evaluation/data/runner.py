SYSTEMS=["graph_rag"]
def run_experiment(cases,split,generate):
    answers=[]
    for case in cases:
        for system in SYSTEMS: answers.append({"id":case["id"],"system":system,"answer":generate(case,system)})
    return {"manifest":{"split":split},"answers":answers}
