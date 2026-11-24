import unittest

from evaluation.analyze_repeats import aggregate


class AnalyzeRepeatsTest(unittest.TestCase):
    def test_missing_run_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate([], "graph_rag")


if __name__ == "__main__":
    unittest.main()
