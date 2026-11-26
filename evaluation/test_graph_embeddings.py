import sys
import unittest
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from knowledge_graph.embeddings import FastRPStyleEmbedding


class GraphEmbeddingTest(unittest.TestCase):
    def test_structural_twins_are_retrieved_as_similar(self):
        graph = nx.MultiDiGraph()
        graph.add_edge("概念A", "共同先修", relation="先修知识")
        graph.add_edge("概念C", "共同先修", relation="先修知识")
        graph.add_edge("无关概念", "其他节点", relation="相关概念")

        embedding = FastRPStyleEmbedding(graph, dimensions=16, random_seed=7)

        result = embedding.similar_nodes(["概念A"], top_k=1)

        self.assertEqual("概念C", result[0][0])

    def test_embedding_is_reproducible_for_a_fixed_seed(self):
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", relation="相关概念")
        graph.add_edge("B", "C", relation="相关概念")

        first = FastRPStyleEmbedding(graph, dimensions=8, random_seed=11)
        second = FastRPStyleEmbedding(graph, dimensions=8, random_seed=11)

        self.assertEqual(first.similar_nodes(["A"], 2), second.similar_nodes(["A"], 2))

    def test_unknown_seeds_return_no_candidates(self):
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", relation="相关概念")
        embedding = FastRPStyleEmbedding(graph, dimensions=8, random_seed=3)

        self.assertEqual([], embedding.similar_nodes(["不存在"], top_k=3))


if __name__ == "__main__":
    unittest.main()
