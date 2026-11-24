import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from shared.schemas import ChunkItem
from parsing.parser import parse_all_course_files
from vector_search.retriever import VectorRetriever


def chunk(chunk_id: str, text: str, source: str, page: int) -> ChunkItem:
    return ChunkItem(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page=page,
        chunk_index=0,
    )


class RetrievalContextTest(unittest.TestCase):
    def build(self, chunks):
        retriever = VectorRetriever()
        retriever._save = lambda _: None
        retriever.build_index(chunks)
        return retriever

    def test_neighbor_pages_are_added_around_anchor(self):
        retriever = self.build([
            chunk("a_9", "无关内容", "lecture_a", 9),
            chunk("a_10", "编辑距离", "lecture_a", 10),
            chunk("a_11", "插入 删除 替换", "lecture_a", 11),
            chunk("a_12", "动态规划", "lecture_a", 12),
        ])

        results = retriever.search_context("编辑距离", top_k=1, neighbor_window=1)

        self.assertEqual({9, 10, 11}, {r.chunk.page for r in results})
        self.assertEqual("semantic", results[0].retrieval_reason)
        self.assertEqual("neighbor", results[1].retrieval_reason)

    def test_comparison_query_includes_a_second_source(self):
        retriever = self.build([
            chunk("a_1", "编辑距离 编辑距离", "lecture_a", 1),
            chunk("a_2", "编辑距离 字符串", "lecture_a", 2),
            chunk("a_3", "编辑距离 操作", "lecture_a", 3),
            chunk("b_1", "余弦相似度 向量", "lecture_b", 20),
        ])

        results = retriever.search_context(
            "编辑距离与余弦相似度有什么根本不同", top_k=3, neighbor_window=0
        )

        self.assertEqual({"lecture_a", "lecture_b"}, {r.chunk.source for r in results})

    def test_one_neighbor_page_cannot_consume_the_context_budget(self):
        retriever = self.build([
            chunk("a_10", "编辑距离", "lecture_a", 10),
            chunk("a_11_1", "表格一", "lecture_a", 11),
            chunk("a_11_2", "表格二", "lecture_a", 11),
            chunk("a_11_3", "表格三", "lecture_a", 11),
            chunk("a_12", "递推公式", "lecture_a", 12),
        ])

        results = retriever.search_context(
            "编辑距离", top_k=1, neighbor_window=2, max_context_chunks=4
        )

        self.assertIn(12, {r.chunk.page for r in results})


class EvaluationRetrievalRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        materials = parse_all_course_files()
        chunks = [chunk for document in materials.values() for chunk in document["chunks"]]
        cls.retriever = VectorRetriever()
        cls.retriever._save = lambda _: None
        cls.retriever.build_index(chunks)

    def test_distributional_hypothesis_evidence_reaches_context(self):
        results = self.retriever.search_context(
            "什么是分布式假设？其中‘上下文’指什么？",
            top_k=3,
            neighbor_window=2,
        )

        pages = {(result.chunk.source, result.chunk.page) for result in results}
        self.assertTrue(
            pages & {("CS-NLP-L6-Word_embedding", 9), ("CS-NLP-L6-Word_embedding", 10)}
        )

    def test_pmi_low_frequency_evidence_reaches_context(self):
        results = self.retriever.search_context(
            "为什么实际构造词项-上下文矩阵时常使用 PPMI？PMI 对低频事件有什么问题？",
            top_k=3,
            neighbor_window=2,
        )

        pages = {(result.chunk.source, result.chunk.page) for result in results}
        self.assertIn(("CS-NLP-L6-Word_embedding", 40), pages)

    def test_cbow_skipgram_diagram_evidence_reaches_context(self):
        results = self.retriever.search_context(
            "CBOW 和 Skip-gram 的输入输出分别是什么？两者预测方向有何不同？",
            top_k=3,
            neighbor_window=2,
        )

        pages = {(result.chunk.source, result.chunk.page) for result in results}
        self.assertTrue(
            {("CS-NLP-L7-Word2Vec", 13), ("CS-NLP-L7-Word2Vec", 14)} <= pages
        )

    def test_normalization_problem_example_reaches_context(self):
        results = self.retriever.search_context(
            "为什么文本预处理需要做归一化？请给出课件中的一种典型问题。",
            top_k=3,
            neighbor_window=2,
        )
        pages = {(result.chunk.source, result.chunk.page) for result in results}
        self.assertTrue(
            pages & {
                ("CS-NLP-L2_Text_Preprocession", 14),
                ("CS-NLP-L2_Text_Preprocession", 20),
                ("CS-NLP-L2_Text_Preprocession", 21),
            }
        )

    def test_edit_distance_backtrace_evidence_reaches_context(self):
        results = self.retriever.search_context(
            "编辑距离数值为什么不足以给出两个字符串的字符对齐？系统还需要保存什么信息？",
            top_k=3,
            neighbor_window=2,
        )
        pages = {(result.chunk.source, result.chunk.page) for result in results}
        self.assertTrue(
            pages & {
                ("CS-NLP-L2_Text_Preprocession", page) for page in range(51, 59)
            }
        )


if __name__ == "__main__":
    unittest.main()
