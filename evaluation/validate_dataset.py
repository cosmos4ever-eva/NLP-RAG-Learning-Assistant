"""Validate the structure and evidence references of questions.jsonl."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(__file__).with_name("questions.jsonl")
KG_PATH = ROOT / "backend" / "knowledge_graph" / "triples.json"

ALLOWED_SPLITS = {"dev", "test"}
ALLOWED_CATEGORIES = {
    "course_grounded",
    "cross_concept",
    "boundary",
}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
ALLOWED_BEHAVIORS = {"answer", "abstain"}
ALLOWED_SOURCES = {
    "CS-NLP-L2 Text Preprocession.pptx": ("slide", 60),
    "CS-NLP-L3-ngram Language Model.pdf": ("page", 87),
    "CS-NLP-L4-Text classification.pdf": ("page", 63),
    "CS-NLP-L5-Logistic Regression.pdf": ("page", 37),
    "CS-NLP-L6-Word embedding.pdf": ("page", 59),
    "CS-NLP-L7-Word2Vec.pdf": ("page", 43),
    "CS-NLP-L8-Sequence Models.pdf": ("page", 81),
}
REQUIRED_FIELDS = {
    "id",
    "split",
    "category",
    "difficulty",
    "question",
    "expected_behavior",
    "expected_answer_points",
    "source_evidence",
    "required_concepts",
    "required_relations",
    "forbidden_claims",
    "rationale",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"line {line_number}: invalid JSON: {exc}") from exc
    return rows


def relation_key(relation: dict) -> tuple[str, str, str]:
    return relation["head"], relation["relation"], relation["tail"]


def validate() -> None:
    rows = load_jsonl(DATASET_PATH)
    assert 40 <= len(rows) <= 60, f"expected 40-60 cases, got {len(rows)}"

    kg_data = json.loads(KG_PATH.read_text(encoding="utf-8-sig"))
    kg_relations = {relation_key(item) for item in kg_data["triples"]}
    kg_entities = {item["name"] for item in kg_data["entities"]}

    ids = set()
    for index, row in enumerate(rows, 1):
        missing = REQUIRED_FIELDS - row.keys()
        assert not missing, f"row {index}: missing fields {sorted(missing)}"
        assert row["id"] not in ids, f"duplicate id: {row['id']}"
        ids.add(row["id"])

        assert row["id"] == f"EVAL-{index:03d}", (
            f"row {index}: expected id EVAL-{index:03d}, got {row['id']}"
        )
        assert row["split"] in ALLOWED_SPLITS, f"{row['id']}: invalid split"
        assert row["category"] in ALLOWED_CATEGORIES, f"{row['id']}: invalid category"
        assert row["difficulty"] in ALLOWED_DIFFICULTIES, f"{row['id']}: invalid difficulty"
        assert row["expected_behavior"] in ALLOWED_BEHAVIORS, f"{row['id']}: invalid behavior"
        assert row["question"].strip(), f"{row['id']}: empty question"
        assert row["rationale"].strip(), f"{row['id']}: empty rationale"

        if row["expected_behavior"] == "answer":
            assert row["expected_answer_points"], f"{row['id']}: answer case has no gold points"
            assert row["source_evidence"], f"{row['id']}: answer case has no evidence"
        else:
            assert row["category"] == "boundary", f"{row['id']}: abstain must be boundary"
            assert not row["source_evidence"], f"{row['id']}: abstain case should have no evidence"
            assert row["forbidden_claims"], f"{row['id']}: abstain case needs forbidden claims"

        for evidence in row["source_evidence"]:
            source = evidence.get("source")
            assert source in ALLOWED_SOURCES, f"{row['id']}: unknown source {source}"
            expected_type, max_locator = ALLOWED_SOURCES[source]
            assert evidence.get("locator_type") == expected_type, (
                f"{row['id']}: {source} must use {expected_type}"
            )
            locators = evidence.get("locators", [])
            assert locators, f"{row['id']}: evidence has no locators"
            assert all(isinstance(value, int) and 1 <= value <= max_locator for value in locators), (
                f"{row['id']}: locator out of range for {source}"
            )
            assert evidence.get("supports", "").strip(), f"{row['id']}: evidence lacks summary"

        for concept in row["required_concepts"]:
            assert concept in kg_entities, f"{row['id']}: unknown KG concept {concept}"

        for relation in row["required_relations"]:
            key = relation_key(relation)
            assert key in kg_relations, f"{row['id']}: relation not in KG: {key}"

    counts = {
        "split": Counter(row["split"] for row in rows),
        "category": Counter(row["category"] for row in rows),
        "difficulty": Counter(row["difficulty"] for row in rows),
        "behavior": Counter(row["expected_behavior"] for row in rows),
    }
    assert counts["split"]["dev"] == counts["split"]["test"], "dev/test must be balanced"
    assert counts["behavior"]["abstain"] >= 6, "need at least six boundary cases"

    print(f"Validated {len(rows)} cases from {DATASET_PATH}")
    for name, counter in counts.items():
        print(f"{name}: {dict(sorted(counter.items()))}")


if __name__ == "__main__":
    validate()
