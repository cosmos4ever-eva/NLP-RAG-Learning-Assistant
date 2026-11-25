import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from graph_rag.pipeline import (
    build_evidence_items,
    build_prompt,
    evaluate_evidence,
    format_kg_context,
    format_ppt_context,
    normalize_answer_markdown,
    select_cited_evidence,
)
from shared.schemas import ChunkItem, SearchResult


class PromptGroundingTest(unittest.TestCase):
    def test_product_prompt_forbids_filling_evidence_gaps(self):
        prompt = build_prompt("写出 LSTM 公式", [], [])

        self.assertIn("只能依据", prompt)
        self.assertIn("不要继续给出课外公式", prompt)
        self.assertNotIn("可以使用模型已有知识回答", prompt)

    def test_llm_only_baseline_can_use_labeled_model_knowledge(self):
        prompt = build_prompt(
            "写出 LSTM 公式", [], [], allow_model_knowledge=True
        )

        self.assertIn("可以使用模型已有知识回答", prompt)

    def test_partial_evidence_is_answered_per_subquestion(self):
        prompt = build_prompt("先解释 A，再比较 B", [], [])

        self.assertIn("逐个子问题", prompt)
        self.assertIn("有证据的部分", prompt)
        self.assertIn("只对缺少证据的部分", prompt)

    def test_product_prompt_is_concise_and_uses_checkable_ids(self):
        prompt = build_prompt("比较两个概念", [], [])

        self.assertIn("不超过 120 个汉字", prompt)
        self.assertIn("最多 3 个要点", prompt)
        self.assertIn("[S1]", prompt)
        self.assertIn("[G2]", prompt)
        self.assertNotIn("概念逻辑链条", prompt)

    def test_prompt_requires_task_model_algorithm_chain_and_limits_citations(self):
        prompt = build_prompt("HMM 如何用于词性标注？", [], [])

        self.assertIn("任务 → 模型 → 算法", prompt)
        self.assertIn("具体小例子", prompt)
        self.assertIn("最多使用 4 个不同的证据编号", prompt)

    def test_context_labels_match_evidence_cards(self):
        result = SearchResult(
            chunk=ChunkItem(
                chunk_id="lecture_8_3",
                text="  前向算法对所有路径概率求和。  ",
                source="CS-NLP-L8-Sequence_Models",
                page=3,
                chunk_index=0,
            ),
            similarity=0.88,
            retrieval_reason="semantic",
        )

        context = format_ppt_context([result])
        items = build_evidence_items([result], max_excerpt_chars=20)

        self.assertIn("[S1]", context)
        self.assertEqual("S1", items[0]["evidence_id"])
        self.assertEqual(3, items[0]["page"])
        self.assertLessEqual(len(items[0]["excerpt"]), 20)

    def test_graph_context_has_matching_relation_ids(self):
        context = format_kg_context([
            {"head": "POS tagging", "relation": "可使用", "tail": "HMM"},
            {"head": "HMM", "relation": "可使用", "tail": "Viterbi"},
        ])

        self.assertIn("[G1]", context)
        self.assertIn("[G2]", context)

    def test_evidence_overview_contains_only_ids_cited_by_answer(self):
        answer = "结论由第二段课件和第一条图关系支持 [S2][G1]。"
        evidence_items = [
            {"evidence_id": "S1", "excerpt": "未被正文引用"},
            {"evidence_id": "S2", "excerpt": "正文实际引用"},
        ]
        triples = [
            {"head": "POS tagging", "relation": "使用模型", "tail": "HMM"},
            {"head": "HMM", "relation": "相关概念", "tail": "Viterbi"},
        ]

        cited = select_cited_evidence(answer, evidence_items, triples)

        self.assertEqual(["S2"], [item["evidence_id"] for item in cited["course"]])
        self.assertEqual(["G1"], [item["evidence_id"] for item in cited["graph"]])

    def test_neighbor_volume_cannot_create_high_support(self):
        weak_anchor = SearchResult(
            chunk=ChunkItem(
                chunk_id="weak_1", text="弱相关内容", source="lecture", page=1, chunk_index=0
            ),
            similarity=0.12,
            retrieval_reason="semantic",
        )
        neighbors = [
            SearchResult(
                chunk=ChunkItem(
                    chunk_id=f"neighbor_{index}",
                    text="相邻页",
                    source="lecture",
                    page=index + 2,
                    chunk_index=0,
                ),
                similarity=0.0,
                retrieval_reason="neighbor",
            )
            for index in range(10)
        ]
        triples = [
            {"head": "A", "relation": "相关概念", "tail": str(index)}
            for index in range(8)
        ]

        evidence = evaluate_evidence([weak_anchor, *neighbors], triples)

        self.assertNotEqual("high", evidence["evidence_level"])
        self.assertIn("最高相似度 0.12", evidence["evidence_summary"])

    def test_strong_semantic_anchor_can_create_high_support(self):
        strong_anchor = SearchResult(
            chunk=ChunkItem(
                chunk_id="strong_1", text="直接定义", source="lecture", page=3, chunk_index=0
            ),
            similarity=0.6,
            retrieval_reason="semantic",
        )

        evidence = evaluate_evidence([strong_anchor], [])

        self.assertEqual("high", evidence["evidence_level"])

    def test_inline_section_content_is_moved_below_heading(self):
        answer = normalize_answer_markdown(
            "### 结论：这是正文 [S1]\n\n### 关键辨析:\n- 要点"
        )

        self.assertIn("### 结论\n这是正文 [S1]", answer)
        self.assertIn("### 关键辨析\n- 要点", answer)


if __name__ == "__main__":
    unittest.main()
