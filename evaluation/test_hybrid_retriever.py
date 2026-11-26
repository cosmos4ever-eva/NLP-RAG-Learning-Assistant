import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from shared.schemas import ChunkItem
from vector_search.hybrid_retriever import QdrantHybridRetriever


def chunk(chunk_id: str, text: str, source: str, page: int) -> ChunkItem:
    return ChunkItem(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page=page,
        chunk_index=0,
    )


class FakeEncoder:
    """Small deterministic semantic space independent from the retriever."""

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if "稠密表示" in lowered or "词嵌入" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "编辑距离" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class ReverseReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(index) for index, _ in enumerate(passages)]


class HybridRetrieverTest(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            chunk("embedding", "词嵌入把词表示为稠密向量", "lecture_6", 10),
            chunk("distance", "编辑距离允许插入删除和替换", "lecture_2", 40),
            chunk("other", "语言模型估计序列概率", "lecture_3", 12),
        ]

    def test_dense_search_bridges_different_surface_words(self):
        retriever = QdrantHybridRetriever(self.chunks, FakeEncoder())

        results = retriever.search("如何获得稠密表示", top_k=1, mode="dense")

        self.assertEqual("embedding", results[0].chunk.chunk_id)

    def test_hybrid_search_returns_unique_ranked_chunks(self):
        retriever = QdrantHybridRetriever(self.chunks, FakeEncoder())

        results = retriever.search("编辑距离与稠密表示", top_k=3, mode="hybrid")

        self.assertEqual(3, len({result.chunk.chunk_id for result in results}))
        self.assertEqual("semantic", results[0].retrieval_reason)

    def test_reranker_controls_final_candidate_order(self):
        retriever = QdrantHybridRetriever(
            self.chunks,
            FakeEncoder(),
            reranker=ReverseReranker(),
        )

        results = retriever.search(
            "编辑距离",
            top_k=2,
            candidate_k=3,
            mode="hybrid",
            rerank=True,
        )

        self.assertEqual("embedding", results[0].chunk.chunk_id)

    def test_context_expansion_marks_neighbor_pages_separately(self):
        chunks = [
            chunk("p9", "无关目录", "lecture_2", 9),
            chunk("p10", "编辑距离", "lecture_2", 10),
            chunk("p11", "插入删除替换", "lecture_2", 11),
        ]
        retriever = QdrantHybridRetriever(chunks, FakeEncoder())

        results = retriever.search_context(
            "编辑距离",
            top_k=1,
            neighbor_window=1,
            mode="hybrid",
        )

        self.assertEqual({9, 10, 11}, {result.chunk.page for result in results})
        self.assertEqual("semantic", results[0].retrieval_reason)
        self.assertTrue(
            all(result.retrieval_reason == "neighbor" for result in results[1:])
        )


if __name__ == "__main__":
    unittest.main()
