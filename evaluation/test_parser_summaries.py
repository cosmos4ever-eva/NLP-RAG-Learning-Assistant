import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from parsing.parser import parse_single_file


class ParserSummaryTest(unittest.TestCase):
    def test_image_heavy_word2vec_slides_have_grounded_manual_summaries(self):
        result = parse_single_file(ROOT / "course_materials" / "CS-NLP-L7-Word2Vec.pdf")
        text_by_page = {}
        for chunk in result["chunks"]:
            text_by_page.setdefault(chunk.page, "")
            text_by_page[chunk.page] += "\n" + chunk.text

        self.assertIn("上下文词", text_by_page[13])
        self.assertIn("预测中心词", text_by_page[13])
        self.assertIn("中心词", text_by_page[14])
        self.assertIn("预测上下文词", text_by_page[14])


if __name__ == "__main__":
    unittest.main()
