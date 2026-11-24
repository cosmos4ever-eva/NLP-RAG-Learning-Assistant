import unittest
from pipeline import answer_parts
class PublicTest(unittest.TestCase):
    def test_all_supported(self):
        result=answer_parts(["a"],lambda q:["p1"],lambda q,s:"ok")
        self.assertEqual("answered",result["status"])
if __name__=="__main__": unittest.main()
