"""Apply an explicit, run-specific review file to model-assisted scores."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORE_FIELDS = ("correctness", "groundedness", "relation", "boundary")


def answer_sha256(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--overrides", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    with (run_dir / "annotations_model_scored.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    overrides_data = json.loads(Path(args.overrides).read_text(encoding="utf-8"))
    overrides = {(item["id"], item["system"]): item for item in overrides_data}
    answers = {
        (item["id"], item["system"]): item["answer"]
        for item in map(
            json.loads,
            (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }

    for key, override in overrides.items():
        if key not in answers:
            raise SystemExit(f"Override target has no saved answer: {key}")
        actual_hash = answer_sha256(answers[key])
        if override.get("answer_sha256") != actual_hash:
            raise SystemExit(f"Stale adjudication override for {key}")

    for row in rows:
        override = overrides.get((row["id"], row["system"]))
        if override:
            for field in SCORE_FIELDS:
                row[field] = str(override[field])
            row["note"] = override["reason"]
            row["adjudication"] = override.get("reviewer", "current_run_manual_review")
        row["total"] = str(sum(int(row[field]) for field in SCORE_FIELDS))

    output = run_dir / "annotations_adjudicated.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    cases = {
        item["id"]: item
        for item in map(
            json.loads,
            (ROOT / "evaluation" / "questions.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    metrics = {"overrides_applied": len(overrides), "systems": {}}
    for system in sorted({row["system"] for row in rows}):
        system_rows = [row for row in rows if row["system"] == system]
        by_category = defaultdict(list)
        for row in system_rows:
            by_category[cases[row["id"]]["category"]].append(int(row["total"]))
        boundary_rows = [
            row for row in system_rows if cases[row["id"]]["expected_behavior"] == "abstain"
        ]
        metrics["systems"][system] = {
            "n": len(system_rows),
            "mean_total": sum(int(row["total"]) for row in system_rows) / len(system_rows),
            "mean_by_category": {
                category: sum(values) / len(values) for category, values in by_category.items()
            },
            "boundary_compliance": (
                sum(int(row["boundary"]) == 2 for row in boundary_rows) / len(boundary_rows)
                if boundary_rows
                else None
            ),
        }
    (run_dir / "metrics_adjudicated.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
