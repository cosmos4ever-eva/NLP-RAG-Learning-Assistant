import argparse
import asyncio
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

import sys

sys.path.insert(0, str(BACKEND))

from learning_agent.agent import LearningAgent
from learning_agent.planner import ModelPlanner
from learning_agent.reflection import ModelReflectionAgent, ReflectiveLearningAgent
from learning_agent.schemas import AgentRequest
from learning_agent.tools import build_default_tools


REFUSAL_MARKERS = (
    "资料不足",
    "证据不足",
    "未检索到",
    "无法依据",
    "无法回答",
    "未提供",
    "需要补充",
)


def load_tasks(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detected_refusal(answer: str) -> bool:
    return any(marker in answer for marker in REFUSAL_MARKERS)


def score_result(task: dict, result, latency_seconds: float) -> dict:
    called_tools = [item.tool for item in result.tool_calls]
    successful_tools = [item.tool for item in result.tool_calls if item.status == "success"]
    required_tools = task["required_tools"]
    required_hits = [tool for tool in required_tools if tool in successful_tools]
    required_recall = len(required_hits) / len(required_tools) if required_tools else 1.0
    unique_called = set(called_tools)
    tool_precision = (
        len(unique_called & set(required_tools)) / len(unique_called)
        if unique_called else 0.0
    )
    tool_success = all(item.status == "success" for item in result.tool_calls)
    refusal = detected_refusal(result.final_answer)
    boundary_ok = not task["expected_refusal"] or refusal
    passed = (
        result.status == "completed"
        and required_recall == 1.0
        and tool_success
        and boundary_ok
    )
    return {
        "id": task["id"],
        "category": task["category"],
        "status": result.status,
        "passed": passed,
        "required_tools": required_tools,
        "called_tools": called_tools,
        "required_tool_recall": round(required_recall, 4),
        "tool_precision": round(tool_precision, 4),
        "tool_success": tool_success,
        "expected_refusal": task["expected_refusal"],
        "detected_refusal": refusal,
        "attempt_count": result.attempt_count,
        "review_history": [item.model_dump(mode="json") for item in result.review_history],
        "latency_seconds": round(latency_seconds, 3),
        "final_answer": result.final_answer,
        "tool_calls": [item.model_dump(mode="json") for item in result.tool_calls],
        "trace": [item.model_dump(mode="json") for item in result.trace],
    }


def summarize(rows: list[dict]) -> dict:
    required_instances = sum(len(row["required_tools"]) for row in rows)
    required_hits = sum(
        row["required_tool_recall"] * len(row["required_tools"])
        for row in rows
    )
    triggered = [row for row in rows if row["attempt_count"] > 1]
    recovered = [row for row in triggered if row["status"] == "completed"]
    boundary = [row for row in rows if row["expected_refusal"]]
    latencies = [row["latency_seconds"] for row in rows]
    return {
        "task_count": len(rows),
        "task_pass_count": sum(row["passed"] for row in rows),
        "task_pass_rate": round(sum(row["passed"] for row in rows) / len(rows), 4),
        "completed_count": sum(row["status"] == "completed" for row in rows),
        "required_tool_recall": round(required_hits / required_instances, 4),
        "mean_tool_precision": round(statistics.mean(row["tool_precision"] for row in rows), 4),
        "tool_call_success_rate": round(
            sum(row["tool_success"] for row in rows) / len(rows), 4
        ),
        "first_pass_count": sum(
            row["status"] == "completed" and row["attempt_count"] == 1
            for row in rows
        ),
        "reflection_trigger_count": len(triggered),
        "reflection_recovery_count": len(recovered),
        "boundary_refusal_count": sum(row["detected_refusal"] for row in boundary),
        "boundary_count": len(boundary),
        "mean_latency_seconds": round(statistics.mean(latencies), 3),
        "median_latency_seconds": round(statistics.median(latencies), 3),
    }


async def run(tasks: list[dict]) -> list[dict]:
    rows = []
    for index, task in enumerate(tasks, start=1):
        executor = LearningAgent(planner=ModelPlanner(), tools=build_default_tools())
        agent = ReflectiveLearningAgent(
            executor=executor,
            reviewer=ModelReflectionAgent(),
        )
        started = time.perf_counter()
        result = await agent.run(AgentRequest(
            goal=task["goal"],
            session_id=f"agent-acceptance-{task['id'].lower()}",
            max_steps=5,
            max_reflections=1,
        ))
        latency = time.perf_counter() - started
        row = score_result(task, result, latency)
        rows.append(row)
        print(
            f"[{index}/{len(tasks)}] {task['id']} "
            f"status={row['status']} pass={row['passed']} "
            f"tools={row['called_tools']} attempts={row['attempt_count']} "
            f"latency={row['latency_seconds']}s",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "evaluation" / "agent_acceptance_v1.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    rows = asyncio.run(run(tasks))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.tasks.relative_to(ROOT)),
        "dataset_sha256": file_sha256(args.tasks),
        "protocol": {
            "max_steps": 5,
            "max_reflections": 1,
            "task_pass": "completed + all required tools succeeded + boundary refusal when required",
        },
        "summary": summarize(rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
