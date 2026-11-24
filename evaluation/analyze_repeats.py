"""Aggregate repeated dev runs without exposing prompts, answers, or course text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def aggregate(run_dirs: list[Path], system: str) -> dict:
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    cases = {
        item["id"]: item
        for item in _load_jsonl(ROOT / "evaluation" / "questions.jsonl")
        if item["split"] == "dev"
    }
    score_runs = []
    answer_runs = []
    run_summaries = []
    for run_dir in run_dirs:
        scores = {
            row["id"]: int(row["total"])
            for row in csv.DictReader(
                (run_dir / "annotations_model_scored.csv").open(encoding="utf-8-sig")
            )
            if row["system"] == system
        }
        answers = {
            item["id"]: item
            for item in _load_jsonl(run_dir / "answers.jsonl")
            if item["system"] == system
        }
        if set(scores) != set(cases) or set(answers) != set(cases):
            raise ValueError(f"{run_dir}: expected all dev case ids")
        answer_scores = [
            scores[case_id]
            for case_id, case in cases.items()
            if case["expected_behavior"] == "answer"
        ]
        run_summaries.append(
            {
                "run": run_dir.name,
                "answer_mean": sum(answer_scores) / len(answer_scores),
                "all_case_raw_judge_mean": sum(scores.values()) / len(scores),
                "total_tokens": sum((item.get("usage") or {}).get("total_tokens", 0) for item in answers.values()),
                "mean_latency_seconds": statistics.mean(item["latency_seconds"] for item in answers.values()),
            }
        )
        score_runs.append(scores)
        answer_runs.append(answers)

    unstable = []
    identical_answers = 0
    for case_id, case in cases.items():
        values = [scores[case_id] for scores in score_runs]
        hashes = {
            hashlib.sha256(answers[case_id]["answer"].encode("utf-8")).hexdigest()
            for answers in answer_runs
        }
        identical_answers += len(hashes) == 1
        if max(values) != min(values):
            unstable.append(
                {
                    "id": case_id,
                    "category": case["category"],
                    "scores": values,
                    "range": max(values) - min(values),
                }
            )

    run_means = [item["answer_mean"] for item in run_summaries]
    return {
        "split": "dev",
        "test_evaluated": False,
        "system": system,
        "runs": run_summaries,
        "answer_mean_across_runs": statistics.mean(run_means),
        "answer_mean_sample_sd": statistics.stdev(run_means) if len(run_means) > 1 else 0.0,
        "answer_mean_range": [min(run_means), max(run_means)],
        "identical_answers_across_all_runs": identical_answers,
        "cases": len(cases),
        "score_unstable_cases": len(unstable),
        "unstable": sorted(unstable, key=lambda item: (-item["range"], item["id"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--system", default="graph_rag")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Output already exists; choose a new path.")
    result = aggregate([Path(item) for item in args.runs], args.system)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
