import unittest

from evaluation.score_experiment import apply_user_overrides, finalize_model_scores, validate_scores


class ScoreExperimentTest(unittest.TestCase):
    def test_validate_scores_accepts_three_complete_systems(self):
        data = {"scores": [
            {"system": system, "correctness": 4, "groundedness": 4, "relation": 2, "boundary": 2, "note": "ok"}
            for system in ("llm_only", "tfidf_rag", "graph_rag")
        ]}
        self.assertEqual(3, len(validate_scores("EVAL-001", data)))

    def test_validate_scores_accepts_requested_system_subset(self):
        data = {"scores": [
            {"system": "graph_rag", "correctness": 4, "groundedness": 4, "relation": 2, "boundary": 2, "note": "ok"}
        ]}

        self.assertEqual(
            1,
            len(validate_scores("EVAL-001", data, expected_systems={"graph_rag"})),
        )

    def test_user_override_recomputes_total(self):
        rows = [{"id": "EVAL-020", "system": "graph_rag", "correctness": 4,
                 "groundedness": 2, "relation": 2, "boundary": 2, "note": "judge"}]
        result = apply_user_overrides(rows)[0]
        self.assertEqual(2, result["correctness"])
        self.assertEqual(4, result["groundedness"])
        self.assertEqual(10, result["total"])
        self.assertEqual("user_approved_override", result["adjudication"])

    def test_fresh_answers_keep_model_scores_without_old_overrides(self):
        rows = [{"id": "EVAL-020", "system": "graph_rag", "correctness": 4,
                 "groundedness": 2, "relation": 2, "boundary": 2, "note": "judge"}]
        result = finalize_model_scores(rows)[0]
        self.assertEqual(4, result["correctness"])
        self.assertEqual(10, result["total"])
        self.assertEqual("model_assisted_first_pass", result["adjudication"])


if __name__ == "__main__":
    unittest.main()
