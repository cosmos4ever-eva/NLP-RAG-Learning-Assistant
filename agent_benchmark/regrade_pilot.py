"""Re-run deterministic evaluators on saved pilot submissions without model calls."""
import argparse,importlib.util,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TASKS=("graph-multiedge","evidence-abstention","multi-question-retrieval","api-evidence-schema","reproducible-eval-runner","partial-evidence-policy","hybrid-retrieval-regression","leakage-safe-evaluation")
def load(task):
    path=ROOT/"tasks"/task/"evaluator.py"; spec=importlib.util.spec_from_file_location("regrade_"+task,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",required=True); a=p.parse_args(); run=Path(a.run_dir); old=json.loads((run/"summary.json").read_text(encoding="utf-8")); old_by={x["task"]:x for x in old["results"]}; results=[]
    for task in TASKS:
        if not (run/task/"submission.json").exists(): continue
        artifact=run/task/"submission.json"; result=load(task).evaluate({"artifacts":{"submission":str(artifact)}}); prior=old_by[task]
        metadata={key:prior[key] for key in ("elapsed_seconds","tool_calls","write_calls","agent_test_calls","correction_prompts","finish_reason","termination_reason","usage","failure_class") if key in prior}
        result.update(task=task,model=old["model"],**metadata); (run/task/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); results.append(result)
    checks_total=sum(len(r.get("checks",{})) for r in results); checks_passed=sum(sum(bool(value) for value in r.get("checks",{}).values()) for r in results)
    old.update(passed=sum(x["status"]=="PASS" for x in results),pass_rate=sum(x["status"]=="PASS" for x in results)/len(results),hidden_checks_passed=checks_passed,hidden_checks_total=checks_total,failure_classes={name:sum(r.get("failure_class")==name for r in results) for name in sorted({r.get("failure_class") for r in results if r.get("failure_class")})},results=results,regraded_without_model_calls=True); (run/"summary.json").write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(old,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
