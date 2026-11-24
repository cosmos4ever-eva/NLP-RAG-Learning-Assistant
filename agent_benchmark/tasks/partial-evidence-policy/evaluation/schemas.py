def make_part(question,status,answer=None,sources=None,missing_material=None):
    return {"question":question,"status":status,"answer":answer,"sources":list(sources or []),"missing_material":missing_material}
