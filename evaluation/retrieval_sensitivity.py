"""Dev-only retrieval sensitivity analysis; never reads or scores test cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from parsing.parser import parse_all_course_files
from vector_search.retriever import VectorRetriever


CONFIGS = (
    ("top3_neighbors0", 3, 0),
    ("top3_neighbors2", 3, 2),
    ("top5_neighbors0", 5, 0),
    ("top5_neighbors2", 5, 2),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gold_pages(case: dict) -> set[tuple[str, int]]:
    return {
        (Path(evidence["source"]).stem.replace(" ", "_")[:50], page)
        for evidence in case["source_evidence"]
        for page in evidence["locators"]
    }


def summarize(rows: list[dict]) -> dict:
    evidence_rows = [row for row in rows if row["gold_pages"]]
    summary = {
        "evidence_cases": len(evidence_rows),
        "anchor_hits": sum(row["anchor_hit"] for row in evidence_rows),
        "context_hits": sum(row["context_hit"] for row in evidence_rows),
        "mean_gold_evidence_recall": (
            sum(row["gold_evidence_recall"] for row in evidence_rows) / len(evidence_rows)
            if evidence_rows
            else None
        ),
        "mean_context_chunks": (
            sum(row["context_chunks"] for row in rows) / len(rows) if rows else None
        ),
        "by_category": {},
    }
    for category in sorted({row["category"] for row in evidence_rows}):
        subset = [row for row in evidence_rows if row["category"] == category]
        summary["by_category"][category] = {
            "n": len(subset),
            "anchor_hits": sum(row["anchor_hit"] for row in subset),
            "context_hits": sum(row["context_hit"] for row in subset),
            "mean_gold_evidence_recall": sum(
                row["gold_evidence_recall"] for row in subset
            ) / len(subset),
        }
    return summary


def paired_delta(base_rows: list[dict], candidate_rows: list[dict]) -> dict:
    base = {row["id"]: row for row in base_rows}
    candidate = {row["id"]: row for row in candidate_rows}
    ids = sorted(set(base) & set(candidate))
    changed = []
    for case_id in ids:
        before = base[case_id]
        after = candidate[case_id]
        delta = after["gold_evidence_recall"] - before["gold_evidence_recall"]
        if delta or after["anchor_hit"] != before["anchor_hit"] or after["context_hit"] != before["context_hit"]:
            changed.append(
                {
                    "id": case_id,
                    "anchor_hit_before": before["anchor_hit"],
                    "anchor_hit_after": after["anchor_hit"],
                    "context_hit_before": before["context_hit"],
                    "context_hit_after": after["context_hit"],
                    "gold_evidence_recall_delta": delta,
                }
            )
    return {
        "cases_compared": len(ids),
        "improved_recall": sum(item["gold_evidence_recall_delta"] > 0 for item in changed),
        "regressed_recall": sum(item["gold_evidence_recall_delta"] < 0 for item in changed),
        "changed_cases": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Output already exists; choose a new path.")

    cases = [
        json.loads(line)
        for line in (ROOT / "evaluation" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = [case for case in cases if case["split"] == "dev"]
    materials = parse_all_course_files()
    chunks = [chunk for document in materials.values() for chunk in document["chunks"]]
    retriever = VectorRetriever()
    retriever._save = lambda _: None
    retriever.build_index(chunks)

    rows_by_config = {}
    summaries = {}
    for name, top_k, neighbor_window in CONFIGS:
        rows = []
        for case in cases:
            gold = _gold_pages(case)
            anchors = retriever.search(case["question"], top_k=top_k)
            context = retriever.search_context(
                case["question"],
                top_k=top_k,
                neighbor_window=neighbor_window,
                max_context_chunks=12,
            )
            anchor_pages = {(item.chunk.source, item.chunk.page) for item in anchors}
            context_pages = {(item.chunk.source, item.chunk.page) for item in context}
            rows.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "gold_pages": len(gold),
                    "anchor_hit": bool(gold & anchor_pages) if gold else None,
                    "context_hit": bool(gold & context_pages) if gold else None,
                    "gold_evidence_recall": len(gold & context_pages) / len(gold) if gold else 0.0,
                    "context_chunks": len(context),
                }
            )
        rows_by_config[name] = rows
        summaries[name] = summarize(rows)

    result = {
        "split": "dev",
        "test_evaluated": False,
        "cases": len(cases),
        "chunk_count": len(chunks),
        "max_context_chunks": 12,
        "dataset_sha256": _sha256(ROOT / "evaluation" / "questions.jsonl"),
        "retriever_sha256": _sha256(ROOT / "backend" / "vector_search" / "retriever.py"),
        "summaries": summaries,
        "comparisons": {
            "neighbors_top3": paired_delta(
                rows_by_config["top3_neighbors0"], rows_by_config["top3_neighbors2"]
            ),
            "top5_with_neighbors": paired_delta(
                rows_by_config["top3_neighbors2"], rows_by_config["top5_neighbors2"]
            ),
        },
        "rows": rows_by_config,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summaries": summaries, "comparisons": result["comparisons"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
