"""Windows compatibility pilot. Not a replacement for official ageval Attempts."""
import argparse,csv,importlib.util,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
import requests

from shared.completion_policy import correction_prompt, decide_completion

ROOT=Path(__file__).resolve().parent
TASKS={"graph-multiedge":["graph.py"],"evidence-abstention":["pipeline.py"],"multi-question-retrieval":["retriever.py"],"api-evidence-schema":["api.py"],"reproducible-eval-runner":["runner.py"],"partial-evidence-policy":["pipeline.py","schemas.py"],"hybrid-retrieval-regression":["retriever.py","config.py"],"leakage-safe-evaluation":["runner.py","scorer.py"]}
TOOLS=[
 {"type":"function","function":{"name":"list_files","description":"List workspace files","parameters":{"type":"object","properties":{}}}},
 {"type":"function","function":{"name":"read_file","description":"Read a UTF-8 file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"write_file","description":"Replace a non-test Python implementation file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
 {"type":"function","function":{"name":"run_public_tests","description":"Run public unittest tests","parameters":{"type":"object","properties":{}}}},
]
def safe(root,raw):
    p=(root/raw).resolve()
    if p!=root and root not in p.parents: raise ValueError("outside_workspace")
    return p
def tool(root,name,args):
    try:
        if name=="list_files": return {"ok":True,"files":sorted(str(p.relative_to(root)).replace("\\","/") for p in root.rglob("*") if p.is_file())}
        if name=="read_file": return {"ok":True,"content":safe(root,args.get("path","")).read_text(encoding="utf-8")}
        if name=="write_file":
            p=safe(root,args.get("path",""))
            if p.suffix!=".py" or "tests" in p.parts: return {"ok":False,"error":"protected_path"}
            p.write_text(str(args.get("content","")),encoding="utf-8"); return {"ok":True}
        if name=="run_public_tests":
            p=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p","test_*.py"],cwd=root,capture_output=True,text=True,timeout=60)
            return {"ok":p.returncode==0,"returncode":p.returncode,"output":(p.stdout+p.stderr)[-3000:]}
        return {"ok":False,"error":"unknown_tool"}
    except Exception as exc: return {"ok":False,"error":f"{type(exc).__name__}:{exc}"}
def evaluator(task):
    path=ROOT/"tasks"/task/"evaluator.py"; spec=importlib.util.spec_from_file_location("eval_"+task,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def solve(task,filenames,model,key,out):
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp); shutil.copytree(ROOT/"tasks"/task/"data",work,dirs_exist_ok=True)
        instruction=(work/"instruction.md").read_text(encoding="utf-8")
        messages=[{"role":"system","content":"You are a coding agent. Be concise and action-oriented: inspect files, modify implementation code, run public tests, and do not modify tests. A task is not complete until you have written an implementation file and run the tests."},{"role":"user","content":instruction}]
        trajectory=[]; started=time.time(); final=""; tool_count=0; write_calls=0; test_calls=0; corrections=0
        usage={"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}; last_finish_reason=""; termination_reason="turn_budget_exhausted"
        for turn in range(12):
            payload={"model":model,"temperature":0.0,"max_tokens":4096,"thinking":{"type":"disabled"},"messages":messages,"tools":TOOLS,"tool_choice":"auto"}
            response=requests.post("https://api.deepseek.com/chat/completions",headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},json=payload,timeout=180)
            response.raise_for_status(); body=response.json(); choice=body["choices"][0]; msg=choice["message"]
            last_finish_reason=str(choice.get("finish_reason") or "")
            turn_usage=body.get("usage") if isinstance(body.get("usage"),dict) else {}
            for key_name in usage: usage[key_name]+=int(turn_usage.get(key_name) or 0)
            calls=msg.get("tool_calls") or []
            assistant_message={"role":"assistant","content":msg.get("content")}
            if calls: assistant_message["tool_calls"]=calls
            messages.append(assistant_message)
            trajectory.append({"event":"model","turn":turn+1,"finish_reason":last_finish_reason,"usage":turn_usage,"has_tool_calls":bool(calls)})
            if not calls:
                final=str(msg.get("content") or "")
                decision=decide_completion(finish_reason=last_finish_reason,write_calls=write_calls,test_calls=test_calls,turn=turn+1,max_turns=12)
                termination_reason=decision["reason"]
                trajectory.append({"event":"completion_guard","turn":turn+1,"action":decision["action"],"reason":decision["reason"]})
                if decision["action"]=="continue":
                    corrections+=1; messages.append({"role":"user","content":correction_prompt(decision["reason"])}); continue
                break
            for call in calls:
                fn=call["function"]; args=json.loads(fn.get("arguments") or "{}"); result=tool(work,fn["name"],args)
                tool_count+=1
                if fn["name"]=="write_file" and result.get("ok"): write_calls+=1
                if fn["name"]=="run_public_tests": test_calls+=1
                trajectory.append({"event":"tool","turn":turn+1,"tool":fn["name"],"args":args,"result":result})
                messages.append({"role":"tool","tool_call_id":call["id"],"content":json.dumps(result,ensure_ascii=False)})
        public=tool(work,"run_public_tests",{}); artifact=out/"submission.json"; out.mkdir(parents=True,exist_ok=True)
        artifact.write_text(json.dumps({"files":{name:(work/name).read_text(encoding="utf-8") for name in filenames},"public_tests":public,"final_text":final,"tool_calls":tool_count,"write_calls":write_calls,"agent_test_calls":test_calls,"correction_prompts":corrections,"finish_reason":last_finish_reason,"termination_reason":termination_reason,"usage":usage},ensure_ascii=False,indent=2),encoding="utf-8")
        result=evaluator(task).evaluate({"artifacts":{"submission":str(artifact)}})
        failure_class="passed" if result["status"]=="PASS" else "no_action" if write_calls==0 else "no_test_before_finish" if test_calls==0 else "public_test_failure" if not public.get("ok") else "hidden_check_failure"
        result.update(task=task,model=model,elapsed_seconds=round(time.time()-started,3),tool_calls=tool_count,write_calls=write_calls,agent_test_calls=test_calls,correction_prompts=corrections,finish_reason=last_finish_reason,termination_reason=termination_reason,usage=usage,failure_class=failure_class)
        (out/"pilot_trajectory.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in trajectory),encoding="utf-8"); (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        return result
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--output",required=True); p.add_argument("--tasks",help="Comma-separated task ids; default all"); a=p.parse_args(); key=os.getenv("DEEPSEEK_API_KEY","").strip()
    if not key: raise SystemExit("DEEPSEEK_API_KEY missing")
    selected=[x.strip() for x in a.tasks.split(",")] if a.tasks else list(TASKS)
    unknown=set(selected)-set(TASKS)
    if unknown: raise SystemExit("unknown tasks: "+", ".join(sorted(unknown)))
    output=Path(a.output); results=[]
    for task in selected:
        result=solve(task,TASKS[task],a.model,key,output/task); results.append(result); print(task,result["status"],flush=True)
    checks_total=sum(len(r.get("checks",{})) for r in results); checks_passed=sum(sum(bool(value) for value in r.get("checks",{}).values()) for r in results)
    summary={"model":a.model,"tasks":len(results),"passed":sum(r["status"]=="PASS" for r in results),"pass_rate":sum(r["status"]=="PASS" for r in results)/len(results),"hidden_checks_passed":checks_passed,"hidden_checks_total":checks_total,"failure_classes":{name:sum(r.get("failure_class")==name for r in results) for name in sorted({r.get("failure_class") for r in results})},"results":results,"official_ageval_attempt":False,"compatibility_note":"ageval 0.10.0 local runtime requires AF_UNIX on native Windows"}
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__": main()
