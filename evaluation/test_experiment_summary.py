import unittest

from evaluation.run_experiment import load_cases, summarize_retrieval


class ExperimentSummaryTest(unittest.TestCase):
    def test_frozen_dataset_has_balanced_splits(self):
        self.assertEqual(28, len(load_cases("dev")))
        self.assertEqual(28, len(load_cases("test")))

    def test_boundary_only_subset_has_null_rates_instead_of_dividing_by_zero(self):
        rows = [{
            "category": "boundary",
            "anchor_hit_at_3": None,
            "context_hit": None,
        }]

        summary = summarize_retrieval(rows)

        self.assertEqual(0, summary["evidence_cases"])
        self.assertIsNone(summary["anchor_hit_at_3"])
        self.assertIsNone(summary["context_hit_rate"])


if __name__ == "__main__":
    unittest.main()
