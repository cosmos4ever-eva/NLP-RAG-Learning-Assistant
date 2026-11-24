import unittest

from evaluation.retrieval_sensitivity import paired_delta, summarize


class RetrievalSensitivityTest(unittest.TestCase):
    def test_summary_excludes_boundary_from_evidence_rates(self):
        rows = [
            {"id": "a", "category": "course_grounded", "gold_pages": 2, "anchor_hit": True, "context_hit": True, "gold_evidence_recall": 0.5, "context_chunks": 3},
            {"id": "b", "category": "boundary", "gold_pages": 0, "anchor_hit": None, "context_hit": None, "gold_evidence_recall": 0.0, "context_chunks": 0},
        ]
        result = summarize(rows)
        self.assertEqual(1, result["evidence_cases"])
        self.assertEqual(1, result["context_hits"])
        self.assertEqual(0.5, result["mean_gold_evidence_recall"])

    def test_paired_delta_reports_recall_changes(self):
        base = [{"id": "a", "anchor_hit": False, "context_hit": True, "gold_evidence_recall": 0.5}]
        candidate = [{"id": "a", "anchor_hit": True, "context_hit": True, "gold_evidence_recall": 1.0}]
        result = paired_delta(base, candidate)
        self.assertEqual(1, result["improved_recall"])
        self.assertEqual(0, result["regressed_recall"])


if __name__ == "__main__":
    unittest.main()
