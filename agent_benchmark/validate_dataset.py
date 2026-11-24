"""Verify every broken baseline fails and every hidden reference passes."""
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TASKS={
    "graph-multiedge":["graph.py"],
    "evidence-abstention":["pipeline.py"],
    "multi-question-retrieval":["retriever.py"],
    "api-evidence-schema":["api.py"],
    "reproducible-eval-runner":["runner.py"],
    "partial-evidence-policy":["pipeline.py","schemas.py"],
    "hybrid-retrieval-regression":["retriever.py","config.py"],
    "leakage-safe-evaluation":["runner.py","scorer.py"],
}

def load_evaluator(path):
    spec=importlib.util.spec_from_file_location("task_evaluator_"+path.parent.name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def score(task,files):
    evaluator=load_evaluator(ROOT/"tasks"/task/"evaluator.py")
    with tempfile.TemporaryDirectory() as tmp:
        artifact=Path(tmp)/"submission.json"
        artifact.write_text(json.dumps({"files":files,"public_tests":{"ok":True}},ensure_ascii=False),encoding="utf-8")
        return evaluator.evaluate({"artifacts":{"submission":str(artifact)}})

def main():
    report={}
    for task,filenames in TASKS.items():
        task_dir=ROOT/"tasks"/task
        broken={name:(task_dir/"data"/name).read_text(encoding="utf-8") for name in filenames}
        gold={}
        for name in filenames:
            path=task_dir/"evaluation"/name
            if not path.exists() and len(filenames)==1: path=task_dir/"evaluation"/"gold.py"
            gold[name]=path.read_text(encoding="utf-8")
        broken_result=score(task,broken); gold_result=score(task,gold)
        report[task]={"broken":broken_result["status"],"gold":gold_result["status"]}
        if broken_result["status"]!="FAIL" or gold_result["status"]!="PASS": raise SystemExit(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
