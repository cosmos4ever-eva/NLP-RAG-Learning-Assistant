import unittest

from graph import KnowledgeGraph


class GraphPublicTest(unittest.TestCase):
    def test_unrelated_edge_is_available(self):
        graph = KnowledgeGraph()
        self.assertTrue(graph.get_neighbors("词嵌入"))


if __name__ == "__main__":
    unittest.main()
