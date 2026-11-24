SYSTEMS = ["graph_rag"]
def run_cases(cases, generate, retrieval_only=False):
    rows=[]
    for case in cases:
        rows.append({"id":case["id"],"system":"graph_rag","answer":generate(case,"graph_rag")})
    return rows
