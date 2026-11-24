"""Verify every broken baseline fails and every hidden reference passes."""
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TASKS={
    "graph-multiedge":"graph.py",
    "evidence-abstention":"pipeline.py",
    "multi-question-retrieval":"retriever.py",
    "api-evidence-schema":"api.py",
    "reproducible-eval-runner":"runner.py",
}

def load_evaluator(path):
    spec=importlib.util.spec_from_file_location("task_evaluator_"+path.parent.name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def score(task,filename,source):
    evaluator=load_evaluator(ROOT/"tasks"/task/"evaluator.py")
    with tempfile.TemporaryDirectory() as tmp:
        artifact=Path(tmp)/"submission.json"
        artifact.write_text(json.dumps({"files":{filename:source},"public_tests":{"ok":True}},ensure_ascii=False),encoding="utf-8")
        return evaluator.evaluate({"artifacts":{"submission":str(artifact)}})

def main():
    report={}
    for task,filename in TASKS.items():
        task_dir=ROOT/"tasks"/task
        broken=(task_dir/"data"/filename).read_text(encoding="utf-8")
        gold=(task_dir/"evaluation"/"gold.py").read_text(encoding="utf-8")
        broken_result=score(task,filename,broken); gold_result=score(task,filename,gold)
        report[task]={"broken":broken_result["status"],"gold":gold_result["status"]}
        if broken_result["status"]!="FAIL" or gold_result["status"]!="PASS": raise SystemExit(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
