def score_answers(cases,answers,judge,apply_overrides=False):
    rows=[]
    for answer in answers:
        case=next(c for c in cases if c["id"]==answer["id"]); score=judge(case,answer)
        if apply_overrides and case.get("legacy_override"): score.update(case["legacy_override"])
        rows.append({**answer,**score,"adjudication":"scored"})
    return rows
