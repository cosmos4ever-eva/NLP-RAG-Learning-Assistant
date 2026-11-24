import unittest

from agent_benchmark.shared.completion_policy import decide_completion


class CompletionPolicyTests(unittest.TestCase):
    def test_truncated_response_must_continue(self):
        decision = decide_completion(
            finish_reason="length", write_calls=1, test_calls=1, turn=3, max_turns=12
        )
        self.assertEqual(decision, {"action": "continue", "reason": "output_truncated"})

    def test_no_write_must_continue(self):
        decision = decide_completion(
            finish_reason="stop", write_calls=0, test_calls=0, turn=3, max_turns=12
        )
        self.assertEqual(decision, {"action": "continue", "reason": "no_write"})

    def test_write_without_agent_test_must_continue(self):
        decision = decide_completion(
            finish_reason="stop", write_calls=1, test_calls=0, turn=3, max_turns=12
        )
        self.assertEqual(decision, {"action": "continue", "reason": "no_test"})

    def test_write_and_test_can_complete(self):
        decision = decide_completion(
            finish_reason="stop", write_calls=1, test_calls=1, turn=3, max_turns=12
        )
        self.assertEqual(decision, {"action": "stop", "reason": "completed"})

    def test_turn_budget_takes_precedence(self):
        decision = decide_completion(
            finish_reason="length", write_calls=0, test_calls=0, turn=12, max_turns=12
        )
        self.assertEqual(decision, {"action": "stop", "reason": "turn_budget_exhausted"})


if __name__ == "__main__":
    unittest.main()
