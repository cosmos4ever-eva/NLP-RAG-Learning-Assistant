import json
from pathlib import Path
def evaluate(inputs):
    data=json.loads(Path(inputs["artifacts"]["submission"]).read_text(encoding="utf-8")); src=data.get("files",{}).get("pipeline.py"); checks={"source_present":isinstance(src,str),"public_tests_passed":bool(data.get("public_tests",{}).get("ok"))}
    if isinstance(src,str):
        ns={}
        try:
            exec(compile(src,"pipeline.py","exec"),ns); called=[]
            def forbidden(*args): called.append(True); raise AssertionError("generator_called")
            abstain=ns["answer_question"]("写出LSTM公式",lambda q:[],lambda q:[],forbidden)
            checks["generator_not_called"]=not called
            message=" ".join(str(abstain.get(field,"")) for field in ("answer","message","explanation","material_needed")).lower()
            shortage=any(x in message for x in ("不足","无法","缺少","insufficient","cannot answer"))
            needed_marker=any(x in message for x in ("需要","补充","need","require"))
            material=any(x in message for x in ("课程","资料","材料","课件","course","lecture","reading"))
            needed=needed_marker and material
            checks["structured_abstention"]=abstain.get("status")=="insufficient_evidence" and shortage and needed
            normal=ns["answer_question"]("什么是词嵌入",lambda q:[{"page":1}],lambda q:[],lambda *a:"ok")
            checks["evidence_path_preserved"]=bool(normal.get("status")=="answered" and normal.get("answer")=="ok" and normal.get("sources"))
        except Exception:
            checks.update(generator_not_called=False,structured_abstention=False,evidence_path_preserved=False)
    passed=all(checks.values()); return {"status":"PASS" if passed else "FAIL","score":1.0 if passed else 0.0,"metrics":checks,"checks":checks}
