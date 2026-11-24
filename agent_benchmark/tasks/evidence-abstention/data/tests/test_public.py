import unittest
from pipeline import answer_question
class PublicTest(unittest.TestCase):
    def test_answer_with_evidence(self):
        result = answer_question("q", lambda q: ["page"], lambda q: [], lambda *args: "answer")
        self.assertEqual("answered", result["status"])
        self.assertEqual("answer", result["answer"])
if __name__ == "__main__": unittest.main()
