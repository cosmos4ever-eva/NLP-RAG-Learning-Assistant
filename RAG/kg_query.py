import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KG_FILE = PROJECT_ROOT / "data" / "processed" / "conceptnet_zh.csv"


class KnowledgeGraph:
    def __init__(self, kg_file=KG_FILE):
        self.df = pd.read_csv(kg_file)

    def query(self, word, topk=10):
        result = self.df[
            (self.df["head"] == word) | (self.df["tail"] == word)
        ]

        if "weight" in result.columns:
            result = result.sort_values("weight", ascending=False)

        return result.head(topk)

    def query_as_text(self, word, topk=10):
        facts = self.query(word, topk)

        lines = []
        for _, row in facts.iterrows():
            lines.append(f"{row['head']} -{row['relation']}-> {row['tail']}")

        return "\n".join(lines)


if __name__ == "__main__":
    kg = KnowledgeGraph()

    word = "苹果"
    print(f"查询词：{word}")
    print(kg.query_as_text(word, topk=10))