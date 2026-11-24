import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from graph_rag.pipeline import extract_entities, extract_grounded_entities, get_kg_triples
from knowledge_graph.graph import KnowledgeGraph


class GraphCoverageTest(unittest.TestCase):
    def setUp(self):
        self.kg = KnowledgeGraph()

    def test_every_course_lecture_has_entities(self):
        by_lecture = Counter(entity["lecture"] for entity in self.kg.entities.values())
        self.assertEqual(
            {"L2", "L3", "L4", "L5", "L6", "L7", "L8"},
            set(by_lecture),
        )
        self.assertTrue(all(by_lecture[lecture] > 0 for lecture in by_lecture))

    def test_runtime_graph_preserves_every_declared_relation(self):
        self.assertEqual(len(self.kg.triples), self.kg.graph.number_of_edges())
        for triple in self.kg.triples:
            with self.subTest(triple=triple):
                self.assertIn(triple, self.kg.get_neighbors(triple["head"], depth=1))

    def test_question_lookup_reaches_each_lecture(self):
        representative_entities = {
            "L2": "文本预处理",
            "L3": "语言模型",
            "L4": "文本分类",
            "L5": "逻辑回归",
            "L6": "词嵌入",
            "L7": "Word2Vec",
            "L8": "隐马尔可夫模型(HMM)",
        }
        for lecture, entity in representative_entities.items():
            with self.subTest(lecture=lecture, entity=entity):
                self.assertTrue(get_kg_triples(entity, depth=1))

    def test_user_aliases_resolve_to_graph_entities(self):
        cases = {
            "中文分词为什么比英文更难？": "分词/Tokenization",
            "预处理通常包含哪些步骤？": "文本预处理",
            "PMI 对低频事件有什么问题？": "点互信息(PMI)",
            "One-hot 与稠密向量有什么差别？": "One-hot编码",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertIn(expected, extract_entities(question))

    def test_retrieved_course_context_can_seed_graph_lookup(self):
        triples = get_kg_triples(
            "根据课件例子解释这个现象",
            context_texts=["词嵌入建立在分布式假设之上。"],
        )

        self.assertIn(
            ("词嵌入", "先修知识", "分布式假设"),
            {(item["head"], item["relation"], item["tail"]) for item in triples},
        )

    def test_grounded_lookup_limits_prompt_noise(self):
        triples = get_kg_triples(
            "比较课程里的多个 NLP 概念",
            context_texts=["文本预处理、词嵌入、Word2Vec、语言模型和文本分类。"],
        )

        self.assertLessEqual(len(triples), 15)

    def test_unrelated_neighbor_context_does_not_pollute_question_entities(self):
        entities = extract_grounded_entities(
            "朴素贝叶斯和逻辑回归有什么区别？",
            context_texts=["语言模型可以使用平滑(Smoothing)处理零概率。"],
        )

        self.assertNotIn("平滑(Smoothing)", entities)


if __name__ == "__main__":
    unittest.main()
