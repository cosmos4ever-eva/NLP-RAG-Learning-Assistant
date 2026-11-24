"""Model-assisted rubric scoring; old human overrides are opt-in."""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def post_with_retries(session, url, key, payload):
    for attempt in range(3):
        try:
            response = session.post(
                url,
                headers={"Authorization": "Bearer " + key},
                json=payload,
                timeout=180,
            )
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
            raise RuntimeError(f"Judge request failed: HTTP {response.status_code}: {response.text[:500]}")
        time.sleep(2 ** attempt)
    raise RuntimeError("Judge request exhausted retries")


def build_judge_prompt(case, retrieval, answers):
    evidence = [
        {
            "source": item["source"],
            "locators": item["locators"],
            "supports": item["supports"],
        }
        for item in case["source_evidence"]
    ]
    retrieved_chunks = [
        {
            "source": item["chunk"]["source"],
            "page": item["chunk"]["page"],
            "text": item["chunk"]["text"],
        }
        for item in retrieval["sources"]
    ]
    compact_answers = {item["system"]: item["answer"] for item in answers}
    score_schema = {
        "scores": [
            {
                "system": system,
                "correctness": 0,
                "groundedness": 0,
                "relation": 0,
                "boundary": 0,
                "note": "简短中文理由",
            }
            for system in compact_answers
        ]
    }
    return f"""
你是严格的 NLP 课程问答评测员。请只根据下列金标准、证据摘要、实际检索上下文和评分规则评分。

评分规则：
- correctness 0-4：4覆盖全部要点无错误；3主要正确但漏次要点；2部分正确但漏关键机制或有一处明显错误；1零散相关；0错误或未回答。
- groundedness 0-4：4关键结论均受课程证据支持且无课外扩写；3主要有依据但有轻微扩展；2部分有依据且有明显扩展；1主要结论缺乏支持；0严重幻觉或伪造来源。
- relation 0-2：2正确表达所需概念关系；1关系不完整；0错误或缺失。若题目没有 required_relations 且没有强行建立错误关系，给2。
- boundary 0-2：expected_behavior=answer 时，2表示范围与证据匹配，1轻微过度扩展，0大量课外内容；expected_behavior=abstain 时，2明确拒答并说明需补什么资料，1拒答但仍给少量课外内容，0给出完整课外实质答案或伪造来源。

注意：llm_only 被允许明确标注后使用模型知识，但它仍要按上述 groundedness 和 boundary 规则评分；不要因为文风长或结构完整而加分。只判断学生问题是否被正确、有依据地完成。

题目：{case['question']}
expected_behavior：{case['expected_behavior']}
expected_answer_points：{json.dumps(case['expected_answer_points'], ensure_ascii=False)}
required_relations：{json.dumps(case['required_relations'], ensure_ascii=False)}
forbidden_claims：{json.dumps(case['forbidden_claims'], ensure_ascii=False)}
金标准证据摘要：{json.dumps(evidence, ensure_ascii=False)}
实际 RAG 检索上下文：{json.dumps(retrieved_chunks, ensure_ascii=False)}
实际图谱三元组：{json.dumps(retrieval['triples'], ensure_ascii=False)}
待评分系统回答：{json.dumps(compact_answers, ensure_ascii=False)}

返回严格 JSON，不要 Markdown：
{json.dumps(score_schema, ensure_ascii=False)}
""".strip()


def validate_scores(case_id, data, expected_systems=None):
    rows = data.get("scores", [])
    systems = {row.get("system") for row in rows}
    expected = expected_systems or {"llm_only", "tfidf_rag", "graph_rag"}
    if systems != set(expected):
        raise ValueError(f"{case_id}: invalid systems {systems}")
    limits = {"correctness": 4, "groundedness": 4, "relation": 2, "boundary": 2}
    for row in rows:
        for field, maximum in limits.items():
            value = row.get(field)
            if not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError(f"{case_id}/{row['system']}: invalid {field}={value}")
    return rows


def apply_user_overrides(rows):
    overrides = {
        ("EVAL-001", "tfidf_rag"): {"correctness": 3},
        ("EVAL-010", "tfidf_rag"): {"correctness": 2},
        ("EVAL-010", "graph_rag"): {"correctness": 2},
        ("EVAL-012", "tfidf_rag"): {"correctness": 3, "groundedness": 4, "relation": 2},
        ("EVAL-012", "graph_rag"): {"correctness": 3, "groundedness": 4, "relation": 2},
        ("EVAL-020", "tfidf_rag"): {"correctness": 2, "groundedness": 4},
        ("EVAL-020", "graph_rag"): {"correctness": 2, "groundedness": 4},
        ("EVAL-053", "tfidf_rag"): {"correctness": 2, "groundedness": 4, "relation": 2},
        ("EVAL-053", "graph_rag"): {"correctness": 2, "groundedness": 4, "relation": 2},
        ("EVAL-055", "tfidf_rag"): {"correctness": 1, "groundedness": 4, "relation": 1},
        ("EVAL-055", "graph_rag"): {"correctness": 1, "groundedness": 4, "relation": 1},
        ("EVAL-045", "llm_only"): {"correctness": 2, "groundedness": 0, "relation": 2, "boundary": 0},
        ("EVAL-046", "llm_only"): {"correctness": 2, "groundedness": 0, "relation": 2, "boundary": 0},
        ("EVAL-047", "llm_only"): {"correctness": 2, "groundedness": 0, "relation": 2, "boundary": 0},
        ("EVAL-048", "llm_only"): {"correctness": 4, "groundedness": 1, "relation": 2, "boundary": 1},
        ("EVAL-045", "tfidf_rag"): {"correctness": 4, "groundedness": 4, "relation": 2, "boundary": 2},
        ("EVAL-045", "graph_rag"): {"correctness": 4, "groundedness": 4, "relation": 2, "boundary": 2},
        ("EVAL-046", "tfidf_rag"): {"correctness": 4, "groundedness": 4, "relation": 2, "boundary": 2},
        ("EVAL-046", "graph_rag"): {"correctness": 4, "groundedness": 4, "relation": 2, "boundary": 2},
        ("EVAL-047", "tfidf_rag"): {"correctness": 4, "groundedness": 4, "relation": 2, "boundary": 2},
        ("EVAL-047", "graph_rag"): {"correctness": 4, "groundedness": 4, "relation": 2, "boundary": 2},
        ("EVAL-048", "tfidf_rag"): {"correctness": 4, "groundedness": 4, "relation": 2, "boundary": 2},
        ("EVAL-048", "graph_rag"): {"correctness": 4, "groundedness": 4, "relation": 2, "boundary": 2},
    }
    for row in rows:
        override = overrides.get((row["id"], row["system"]))
        if override:
            row.update(override)
            row["adjudication"] = "user_approved_override"
            row["note"] = row["note"] + "；已应用用户确认裁决"
        else:
            row["adjudication"] = "model_assisted_first_pass"
        row["total"] = sum(row[field] for field in ("correctness", "groundedness", "relation", "boundary"))
    return rows


def finalize_model_scores(rows):
    """Finalize a fresh run without leaking adjudications from older answers."""
    for row in rows:
        row["adjudication"] = "model_assisted_first_pass"
        row["total"] = sum(
            row[field] for field in ("correctness", "groundedness", "relation", "boundary")
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument(
        "--apply-user-overrides",
        action="store_true",
        help="Apply the legacy, answer-specific adjudications from the previous run.",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output_csv = run_dir / (
        "annotations_adjudicated.csv" if args.apply_user_overrides else "annotations_model_scored.csv"
    )
    raw_path = run_dir / "judge_raw.jsonl"
    if output_csv.exists() or raw_path.exists():
        raise SystemExit("Scoring outputs already exist; use a fresh run or move them explicitly.")
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is missing")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    split = manifest.get("split")
    if split not in {"dev", "test"}:
        raise SystemExit("Run manifest has an invalid split")
    cases = {
        row["id"]: row
        for row in load_jsonl(ROOT / "evaluation/questions.jsonl")
        if row["split"] == split
    }
    retrieval = {row["id"]: row for row in load_jsonl(run_dir / "retrieval.jsonl")}
    answers = load_jsonl(run_dir / "answers.jsonl")
    answers_by_id = {case_id: [row for row in answers if row["id"] == case_id] for case_id in cases}
    expected_systems = {row["system"] for row in answers}

    session = requests.Session()
    session.trust_env = not args.direct
    rows = []
    for index, case_id in enumerate(cases, 1):
        prompt = build_judge_prompt(cases[case_id], retrieval[case_id], answers_by_id[case_id])
        payload = {
            "model": args.model,
            "thinking": {"type": "disabled"},
            "temperature": 0.0,
            "max_tokens": 1800,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "你是严格、保守、一致的课程问答评分员，只输出JSON。"},
                {"role": "user", "content": prompt},
            ],
        }
        data = post_with_retries(session, "https://api.deepseek.com/chat/completions", key, payload)
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        scored = validate_scores(case_id, parsed, expected_systems=expected_systems)
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": case_id, "model": data.get("model"), "usage": data.get("usage"), "scores": scored}, ensure_ascii=False) + "\n")
        for row in scored:
            rows.append({"id": case_id, **row})
        print(f"{index}/{len(cases)} {case_id} judged", flush=True)

    rows = apply_user_overrides(rows) if args.apply_user_overrides else finalize_model_scores(rows)
    fields = ["id", "system", "correctness", "groundedness", "relation", "boundary", "total", "adjudication", "note"]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
