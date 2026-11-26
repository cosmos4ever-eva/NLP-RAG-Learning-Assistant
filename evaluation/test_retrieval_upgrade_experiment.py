import json
import tempfile
import unittest
from pathlib import Path

from evaluation.run_retrieval_upgrade_experiment import load_cases, summarize_rows


class RetrievalUpgradeExperimentTest(unittest.TestCase):
    def test_sealed_test_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.jsonl"
            path.write_text(
                json.dumps({"id": "T1", "split": "test"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "sealed"):
                load_cases(path, split="test", allow_sealed_test=False)

    def test_summary_separates_evidence_and_relation_denominators(self):
        rows = [
            {
                "id": "A",
                "gold_pages": 2,
                "anchor_hit": True,
                "context_hit": True,
                "gold_evidence_recall": 0.5,
                "reciprocal_rank": 1.0,
                "required_relations": 1,
                "relation_recall": 1.0,
                "latency_ms": 10.0,
                "retrieved_triples": 3,
            },
            {
                "id": "B",
                "gold_pages": 0,
                "anchor_hit": None,
                "context_hit": None,
                "gold_evidence_recall": None,
                "reciprocal_rank": None,
                "required_relations": 0,
                "relation_recall": None,
                "latency_ms": 20.0,
                "retrieved_triples": 0,
            },
        ]

        summary = summarize_rows(rows)

        self.assertEqual(1, summary["evidence_cases"])
        self.assertEqual(1.0, summary["evidence_hit_at_3"])
        self.assertEqual(0.5, summary["mean_gold_evidence_recall"])
        self.assertEqual(1, summary["relation_cases"])
        self.assertEqual(1.0, summary["mean_relation_recall"])
        self.assertEqual(15.0, summary["mean_latency_ms"])


if __name__ == "__main__":
    unittest.main()
