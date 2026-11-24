from schemas import make_part
def answer_parts(parts,retrieve,generate):
    evidence=[retrieve(question) for question in parts]
    if not all(evidence):
        return {"status":"insufficient_evidence","parts":[]}
    return {"status":"answered","parts":[make_part(q,"answered",generate(q,s),s) for q,s in zip(parts,evidence)]}
