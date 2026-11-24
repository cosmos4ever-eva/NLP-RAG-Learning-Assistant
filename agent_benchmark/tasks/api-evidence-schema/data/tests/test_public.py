import unittest
from api import build_response
class PublicTest(unittest.TestCase):
    def test_answer_is_preserved(self): self.assertEqual("ok",build_response("ok",[],[],"low")["answer"])
if __name__ == "__main__": unittest.main()
