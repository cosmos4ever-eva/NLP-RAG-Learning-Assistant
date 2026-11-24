"""Small tool-using coding harness shared by the benchmark tasks."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ageval_sdk import Agent, RunContext, RunTerminal


TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "List workspace files.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 workspace file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Replace a UTF-8 workspace file with complete content.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_public_tests", "description": "Run the public unittest suite in the workspace.", "parameters": {"type": "object", "properties": {}}}},
]


def _safe_path(root: Path, raw: str) -> Path:
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise ValueError("path_outside_workspace")
    return path


def _native_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    calls = response.get("tool_calls") if isinstance(response.get("tool_calls"), list) else []
    return [item for item in calls if isinstance(item, dict) and item.get("name")]


def _tool_result(root: Path, name: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        if name == "list_files":
            return {"ok": True, "files": sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file())}
        if name == "read_file":
            path = _safe_path(root, str(args.get("path") or ""))
            return {"ok": True, "content": path.read_text(encoding="utf-8")}
        if name == "write_file":
            path = _safe_path(root, str(args.get("path") or ""))
            if path.suffix != ".py" or "tests" in path.parts:
                return {"ok": False, "error": "only_non_test_python_files_are_writable"}
            path.write_text(str(args.get("content") or ""), encoding="utf-8")
            return {"ok": True, "path": str(path.relative_to(root))}
        if name == "run_public_tests":
            proc = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            return {"ok": proc.returncode == 0, "returncode": proc.returncode, "output": (proc.stdout + proc.stderr)[-4000:]}
        return {"ok": False, "error": f"unknown_tool:{name}"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}"}


def _append_tool_messages(messages: list[dict[str, Any]], call: dict[str, Any], result: dict[str, Any], counter: int) -> None:
    call_id = str(call.get("id") or f"call_{counter}")
    name = str(call.get("name") or "")
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]})
    messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, ensure_ascii=False)})


async def run_coding_task(ctx: RunContext) -> RunTerminal:
    root = ctx.workspace_root.resolve()
    instruction_path = root / "instruction.md"
    if not instruction_path.is_file():
        return RunTerminal.failed("instruction_missing")
    instruction = instruction_path.read_text(encoding="utf-8")
    params = ctx.params if isinstance(ctx.params, dict) else {}
    files = params.get("submission_files") if isinstance(params.get("submission_files"), list) else []
    profile = str(params.get("active_profile") or "solver")
    messages: list[dict[str, Any]] = [{"role": "system", "content": "You are a coding agent. Inspect the workspace, edit only implementation Python files, run public tests, and finish when the task is solved."}]
    tool_count = 0
    final_text = ""
    agent = Agent(attempt_id=ctx.scope.attempt_id)
    async with agent.session(profile, max_turns=12) as session:
        for turn in range(12):
            prompt = instruction if turn == 0 else "Continue solving the task. Use tools as needed; when complete, briefly state completion without another tool call."
            response = await session.invoke(prompt, tools=TOOLS, messages=messages)
            if not response.get("ok"):
                return RunTerminal.failed(str(response.get("error") or "agent_invoke_failed"))
            calls = _native_tool_calls(response)
            if not calls:
                final_text = str(response.get("text") or response.get("content") or "")
                break
            call = calls[0]
            args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            result = _tool_result(root, str(call.get("name") or ""), args)
            tool_count += 1
            _append_tool_messages(messages, call, result, tool_count)

    submission = {}
    for raw in files:
        path = _safe_path(root, str(raw))
        submission[str(raw)] = path.read_text(encoding="utf-8") if path.is_file() else None
    public_result = _tool_result(root, "run_public_tests", {})
    ctx.publish_json("submission", {"files": submission, "tool_calls": tool_count, "public_tests": public_result, "final_text": final_text})
    return RunTerminal.completed("coding-task-finished")
