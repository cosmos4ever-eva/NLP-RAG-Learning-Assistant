import unittest
from retriever import search_context
class PublicTest(unittest.TestCase):
    def test_exact_term(self):
        chunks=[{"id":"x","source":"L","page":1,"text":"rare concept"},{"id":"y","source":"L","page":4,"text":"other"}]
        self.assertEqual("x",search_context(chunks,"rare concept",top_k=1)[0]["id"])
if __name__=="__main__": unittest.main()
