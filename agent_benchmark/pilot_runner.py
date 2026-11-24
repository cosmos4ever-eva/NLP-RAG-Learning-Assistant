"""Windows compatibility pilot. Not a replacement for official ageval Attempts."""
import argparse,csv,importlib.util,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
TASKS={"graph-multiedge":"graph.py","evidence-abstention":"pipeline.py","multi-question-retrieval":"retriever.py","api-evidence-schema":"api.py","reproducible-eval-runner":"runner.py"}
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
def solve(task,filename,model,key,out):
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp); shutil.copytree(ROOT/"tasks"/task/"data",work,dirs_exist_ok=True)
        instruction=(work/"instruction.md").read_text(encoding="utf-8")
        messages=[{"role":"system","content":"You are a coding agent. Inspect files, fix the implementation, run tests, and do not modify tests."},{"role":"user","content":instruction}]
        trajectory=[]; started=time.time(); final=""
        for turn in range(12):
            payload={"model":model,"temperature":0.0,"max_tokens":4096,"thinking":{"type":"disabled"},"messages":messages,"tools":TOOLS,"tool_choice":"auto"}
            response=requests.post("https://api.deepseek.com/chat/completions",headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},json=payload,timeout=180)
            response.raise_for_status(); msg=response.json()["choices"][0]["message"]; messages.append(msg)
            calls=msg.get("tool_calls") or []
            if not calls: final=str(msg.get("content") or ""); break
            for call in calls:
                fn=call["function"]; args=json.loads(fn.get("arguments") or "{}"); result=tool(work,fn["name"],args)
                trajectory.append({"turn":turn+1,"tool":fn["name"],"args":args,"result":result})
                messages.append({"role":"tool","tool_call_id":call["id"],"content":json.dumps(result,ensure_ascii=False)})
        public=tool(work,"run_public_tests",{}); artifact=out/"submission.json"; out.mkdir(parents=True,exist_ok=True)
        artifact.write_text(json.dumps({"files":{filename:(work/filename).read_text(encoding="utf-8")},"public_tests":public,"final_text":final,"tool_calls":len(trajectory)},ensure_ascii=False,indent=2),encoding="utf-8")
        result=evaluator(task).evaluate({"artifacts":{"submission":str(artifact)}}); result.update(task=task,model=model,elapsed_seconds=round(time.time()-started,3),tool_calls=len(trajectory))
        (out/"pilot_trajectory.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in trajectory),encoding="utf-8"); (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        return result
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--output",required=True); a=p.parse_args(); key=os.getenv("DEEPSEEK_API_KEY","").strip()
    if not key: raise SystemExit("DEEPSEEK_API_KEY missing")
    output=Path(a.output); results=[]
    for task,filename in TASKS.items():
        result=solve(task,filename,a.model,key,output/task); results.append(result); print(task,result["status"],flush=True)
    summary={"model":a.model,"tasks":len(results),"passed":sum(r["status"]=="PASS" for r in results),"pass_rate":sum(r["status"]=="PASS" for r in results)/len(results),"results":results,"official_ageval_attempt":False,"compatibility_note":"ageval 0.10.0 local runtime requires AF_UNIX on native Windows"}
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__": main()
