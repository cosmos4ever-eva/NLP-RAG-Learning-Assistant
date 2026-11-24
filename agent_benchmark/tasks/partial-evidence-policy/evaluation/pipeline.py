from schemas import make_part
def answer_parts(parts,retrieve,generate):
    output=[]; answered=0
    for question in parts:
        sources=retrieve(question)
        if sources:
            output.append(make_part(question,"answered",generate(question,sources),sources,None)); answered+=1
        else: output.append(make_part(question,"insufficient_evidence",None,[],"direct course evidence for this subquestion"))
    status="answered" if answered==len(parts) else "insufficient_evidence" if answered==0 else "partial"
    return {"status":status,"parts":output}
