import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from graph_rag.pipeline import build_prompt


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


if __name__ == "__main__":
    unittest.main()
