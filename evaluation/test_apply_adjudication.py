import unittest

from evaluation.apply_adjudication import answer_sha256


class ApplyAdjudicationTest(unittest.TestCase):
    def test_answer_hash_is_stable_and_answer_specific(self):
        self.assertEqual(answer_sha256("same answer"), answer_sha256("same answer"))
        self.assertNotEqual(answer_sha256("same answer"), answer_sha256("changed answer"))


if __name__ == "__main__":
    unittest.main()
