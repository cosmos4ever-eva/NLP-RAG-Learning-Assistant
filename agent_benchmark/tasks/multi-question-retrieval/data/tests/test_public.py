import unittest
from retriever import search_context
class PublicTest(unittest.TestCase):
    def test_single_question(self):
        chunks=[{"id":"a","source":"L1","page":1,"text":"编辑距离"},{"id":"b","source":"L1","page":2,"text":"语言模型"}]
        self.assertEqual("a",search_context(chunks,"编辑距离",top_k=1)[0]["id"])
if __name__ == "__main__": unittest.main()
