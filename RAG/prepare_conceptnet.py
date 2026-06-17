import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "assertions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "conceptnet_zh.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as fout:

    reader = csv.reader(fin, delimiter="\t")
    writer = csv.writer(fout)

    writer.writerow(["head", "relation", "tail", "weight"])

    for row in reader:
        if len(row) < 5:
            continue

        edge_id, relation, head, tail, meta = row

        if head.startswith("/c/zh/") and tail.startswith("/c/zh/"):
            data = json.loads(meta)
            weight = data.get("weight", 1.0)

            head = head.replace("/c/zh/", "").replace("_", " ")
            tail = tail.replace("/c/zh/", "").replace("_", " ")
            relation = relation.replace("/r/", "")

            writer.writerow([head, relation, tail, weight])

print(f"处理完成，保存到：{OUTPUT_FILE}")