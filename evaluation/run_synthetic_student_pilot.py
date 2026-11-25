"""Run a small synthetic-student formative pilot over frozen assistant answers.

This is not a human learning experiment.  The student profile schema and the
perceive -> understand -> answer -> reflect loop are adapted from AgentSchool
(MIT License, https://github.com/epitome-AISS/AgentSchool).  Raw generations
remain in evaluation/private_runs because they contain course-derived text.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "evaluation/private_runs/frozen_test_8c55b6d_20261124"
API_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
LESSON_IDS = ("EVAL-033", "EVAL-038", "EVAL-039")


PROFILES = [
    {
        "id": "U01_global_high",
        "prior_knowledge": "较高：理解 HMM、动态规划与序列标注的基本定义",
        "metacognition": "较准确，会主动检查概念层次是否混淆",
        "affective_state": "动机高、压力低",
        "attention": 0.82,
        "learning_style": "直觉型、主动型、全局型；先抓任务—模型—算法框架",
        "common_mistake": "容易跳过具体递推与回溯细节",
    },
    {
        "id": "U02_sequential_mid",
        "prior_knowledge": "中等：学过 HMM，但常混淆前向算法和 Viterbi",
        "metacognition": "基本准确，遇到不确定会请求例子",
        "affective_state": "动机中高、压力中等",
        "attention": 0.68,
        "learning_style": "感官型、反思型、顺序型；依赖步骤和对比例子",
        "common_mistake": "能复述定义，但迁移到新问题时层次混乱",
    },
    {
        "id": "U03_anxious_low",
        "prior_knowledge": "较低：只知道 POS tagging 是给词贴标签",
        "metacognition": "容易低估能力，害怕答错而省略推理",
        "affective_state": "动机中等、压力较高",
        "attention": 0.48,
        "learning_style": "感官型、反思型、顺序型；需要短步骤和明确反馈",
        "common_mistake": "机械记忆术语，无法说明任务、模型、算法的关系",
    },
    {
        "id": "U04_overconfident_skim",
        "prior_knowledge": "中等偏低：见过相关术语但掌握不稳定",
        "metacognition": "倾向高估理解程度，很少核对引用与条件",
        "affective_state": "动机中等、压力低",
        "attention": 0.42,
        "learning_style": "直觉型、主动型、全局型；偏好快速结论",
        "common_mistake": "把 HMM 当成任务、把 POS tagging 当成算法",
    },
]


TRANSFER_ITEMS = [
    {
        "id": "T1",
        "question": "某观测序列只有三条可能隐藏路径，联合概率分别为 0.24、0.18、0.08。前向算法与 Viterbi 各应返回什么量？请给出数值，并解释为什么不能把两个结果互换。",
        "rubric": ["前向序列概率为三条路径求和 0.50", "Viterbi 最优路径概率为最大值 0.24，并对应最优路径", "说明求和与取最大值目标不同，不能互换"],
    },
    {
        "id": "T2",
        "question": "诊断这句话中的概念错误并改正：‘HMM 是一种词性标注算法，Viterbi 是给 HMM 提供转移概率的概率模型，而 POS tagging 是最后的解码步骤。’",
        "rubric": ["指出 POS tagging 才是任务", "指出 HMM 是可用于该任务的概率模型而非标注算法", "指出 Viterbi 是求最可能状态/标签序列的解码算法，而非概率模型或转移概率来源"],
    },
    {
        "id": "T3",
        "question": "一个学生在每个位置都选择当前发射概率最高的词性，并声称这等价于 Viterbi。这个推理错在哪里？什么情况下局部选择尤其可能失败？",
        "rubric": ["Viterbi 优化整条序列的联合/路径概率而非逐位置独立最大", "还要考虑状态转移或相邻标签依赖", "局部最优在转移约束强、歧义词或后续证据改变选择时可能不是全局最优"],
    },
    {
        "id": "T4",
        "question": "诊断这句话：‘编辑距离和 Viterbi 都用动态规划，所以二者的 DP 状态含义相同，回溯结果也都是字符编辑操作。’请分别说明两者的状态与回溯产物。",
        "rubric": ["共同使用动态规划不代表状态语义相同", "编辑距离状态对应两个字符串前缀/子问题，回溯为编辑操作或对齐", "Viterbi 状态对应时间位置与隐藏状态，回溯为最可能隐藏状态/标签序列"],
    },
    {
        "id": "T5",
        "question": "诊断这句话：‘既然课件用 HMM 和 Viterbi 讲 POS tagging，所以所有词性标注系统都必须用 HMM，而且只要用了 HMM 就一定要逐位置选概率最大的标签。’至少指出两个独立错误。",
        "rubric": ["HMM 只是 POS tagging 的一种可用模型，不是所有系统必用", "HMM 序列解码不能等同于逐位置贪心最大，应考虑整条路径与转移", "保持任务—模型—算法层次并给出正确改写"],
    },
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_panel_items(answer: str, sources: list[dict], triples: list[dict]) -> dict:
    """Mirror the product UI: show only evidence IDs used in the answer."""
    cited_ids = []
    for evidence_id in re.findall(r"\b([SG]\d+)\b", answer or "", flags=re.IGNORECASE):
        normalized = evidence_id.upper()
        if normalized not in cited_ids:
            cited_ids.append(normalized)
    source_by_id = {item["evidence_id"]: item for item in sources}
    triple_by_id = {item["evidence_id"]: item for item in triples}
    selected = [
        source_by_id.get(evidence_id) or triple_by_id.get(evidence_id)
        for evidence_id in cited_ids[:4]
    ]
    return {
        "course_sources": [item for item in selected if item and item["evidence_id"].startswith("S")],
        "graph_relations": [item for item in selected if item and item["evidence_id"].startswith("G")],
    }


def load_lessons(
    source: Path,
    *,
    include_evidence_panel: bool = False,
) -> dict[str, dict[str, dict]]:
    cases = {row["id"]: row for row in read_jsonl(ROOT / "evaluation/questions.jsonl")}
    answers: dict[str, dict[str, dict]] = {item_id: {} for item_id in LESSON_IDS}
    for row in read_jsonl(source / "answers.jsonl"):
        if row["id"] in answers and row["system"] in {"tfidf_rag", "graph_rag"}:
            answers[row["id"]][row["system"]] = row
    for item_id in LESSON_IDS:
        if set(answers[item_id]) != {"tfidf_rag", "graph_rag"}:
            raise RuntimeError(f"Missing frozen answers for {item_id}")
        answers[item_id]["question"] = cases[item_id]["question"]
    if include_evidence_panel:
        retrieval_path = source / "retrieval.jsonl"
        if not retrieval_path.exists():
            raise RuntimeError("Evidence-panel evaluation requires retrieval.jsonl")
        retrieval = {row["id"]: row for row in read_jsonl(retrieval_path)}
        for item_id in LESSON_IDS:
            row = retrieval.get(item_id)
            if row is None:
                raise RuntimeError(f"Missing retrieval evidence for {item_id}")
            sources = []
            for index, result in enumerate(row.get("sources", []), start=1):
                chunk = result.get("chunk", {})
                excerpt = " ".join(str(chunk.get("text", "")).split())
                sources.append({
                    "evidence_id": f"S{index}",
                    "source": chunk.get("source", ""),
                    "page": chunk.get("page", 0),
                    "excerpt": excerpt[:180],
                })
            triples = [
                {
                    "evidence_id": f"G{index}",
                    "head": triple.get("head", ""),
                    "relation": triple.get("relation", ""),
                    "tail": triple.get("tail", ""),
                }
                for index, triple in enumerate(row.get("triples", []), start=1)
            ]
            answers[item_id]["tfidf_rag"]["evidence_panel"] = select_panel_items(
                answers[item_id]["tfidf_rag"]["answer"], sources, []
            )
            answers[item_id]["graph_rag"]["evidence_panel"] = select_panel_items(
                answers[item_id]["graph_rag"]["answer"], sources, triples
            )
    return answers


def extract_json(text: str):
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        return json.loads(cleaned[start : end + 1])


def post_model(key: str, messages: list[dict], *, max_tokens: int = 2400, temperature: float = 0.2) -> tuple[str, dict]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if MODEL == "deepseek-flash":
        payload["thinking"] = {"type": "disabled"}
    else:
        payload["temperature"] = temperature
    last_error = None
    session = requests.Session()
    # Match the frozen DeepSeek evaluation runner's direct-connect option.
    # Certificate verification remains enabled.
    session.trust_env = False
    for attempt in range(3):
        try:
            response = session.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=180,
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"], data.get("usage", {})
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2**attempt)
    raise RuntimeError(f"Model call failed: {last_error}")


def learning_prompt(profile: dict, system: str, lessons: dict) -> str:
    material = []
    for item_id in LESSON_IDS:
        material.append(
            {
                "original_question": lessons[item_id]["question"],
                "assistant_answer": lessons[item_id][system]["answer"],
                **(
                    {"evidence_panel": lessons[item_id][system]["evidence_panel"]}
                    if "evidence_panel" in lessons[item_id][system]
                    else {}
                ),
            }
        )
    return f"""
你在执行一个合成学习者预实验。请化身为一名大学 NLP 课程学生，并严格保持下面画像；不要表现成全知专家。

学生画像：{json.dumps(profile, ensure_ascii=False)}

现在只执行 AgentSchool 风格闭环的前半段：
1. perceive：按注意力水平接收课程助手回答；
2. understand：形成自己的理解，必须允许遗忘、压缩和画像对应的误解；
3. reflect：报告仍然困惑之处和是否需要追问。

你刚才使用了一个匿名课程助手，以下是三段真实系统输出。不要猜测它是什么检索系统：
{json.dumps(material, ensure_ascii=False)}

仅输出一个 JSON 对象，结构必须为：
{{
  "perceived": [{{"lesson_id":"EVAL-033","understanding":"...","confusion":"...","needs_follow_up":true}}],
  "memory_trace": ["离开助手后仍能记住的一条知识；不要复制原文"],
  "experience": {{
    "clarity": 1,
    "cognitive_load": 1,
    "evidence_checkability": 1,
    "trust": 1,
    "would_reuse": 1,
    "most_helpful":"...",
    "main_problem":"..."
  }}
}}
所有 1—5 评分均为整数。perceived 必须有 3 项；memory_trace 最多 8 条，并且只能包含该画像实际保留下来的信息。
""".strip()


def posttest_prompt(profile: dict, memory_trace: list[str]) -> str:
    return f"""
你是同一名大学 NLP 学生，但这是一个全新的闭卷后测会话。你看不到课程助手原文，也不得调用预训练知识补全记忆之外的细节。

学生画像：{json.dumps(profile, ensure_ascii=False)}
学习后实际留下的记忆痕迹：{json.dumps(memory_trace, ensure_ascii=False)}

请仅依靠这些记忆作答。若记忆不足，可以答错、不完整或明确说不确定；不要为了显得聪明而恢复未保留的知识。
题目：{json.dumps([{"id": x["id"], "question": x["question"]} for x in TRANSFER_ITEMS], ensure_ascii=False)}

仅输出 JSON 对象：
{{"posttest":[{{"id":"T1","answer":"...","confidence":1}}]}}
posttest 必须有 5 项；confidence 为 1—5 的整数。
""".strip()


def validate_learning_result(result: dict):
    if len(result.get("perceived", [])) != 3:
        raise ValueError("Learning result has wrong perceived-item count")
    if not isinstance(result.get("memory_trace"), list) or not 1 <= len(result["memory_trace"]) <= 8:
        raise ValueError("Learning result has invalid memory trace")
    for key in ("clarity", "cognitive_load", "evidence_checkability", "trust", "would_reuse"):
        value = result.get("experience", {}).get(key)
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"Invalid experience rating: {key}={value}")


def validate_posttest_result(result: dict):
    expected_ids = {item["id"] for item in TRANSFER_ITEMS}
    if len(result.get("posttest", [])) != len(TRANSFER_ITEMS):
        raise ValueError("Posttest result has wrong item count")
    if {x.get("id") for x in result["posttest"]} != expected_ids:
        raise ValueError("Posttest result has wrong ids")
    for item in result["posttest"]:
        if not isinstance(item.get("confidence"), int) or not 1 <= item["confidence"] <= 5:
            raise ValueError(f"Invalid confidence for {item.get('id')}")


def judge_prompt(blinded: list[dict]) -> str:
    return f"""
你是 NLP 课程后测评分员。下面是匿名化的学生回答，评分时不知道学生使用了哪种课程助手。
逐项严格按 rubric 给 0、1、2、3 分：每覆盖一个 rubric 要点得 1 分；同义表达可以得分，但错误结论不能因语言流畅而得分。
不要因为语言流畅、信心或篇幅加分。只评答案内容。

题目与 rubric：
{json.dumps(TRANSFER_ITEMS, ensure_ascii=False)}

匿名回答：
{json.dumps(blinded, ensure_ascii=False)}

仅输出 JSON 数组，每个元素为：
{{"blind_id":"S01","scores":{{"T1":0,"T2":0,"T3":0,"T4":0,"T5":0}},"total":0,"brief_reason":"..."}}
total 必须等于五项之和，范围 0—15。
""".strip()


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for system in ("tfidf_rag", "graph_rag"):
        subset = [row for row in rows if row["system"] == system]
        summary[system] = {
            "n": len(subset),
            "mean_posttest_0_15": round(statistics.mean(row["judge"]["total"] for row in subset), 3),
            "perfect_score_rate": round(sum(row["judge"]["total"] == 15 for row in subset) / len(subset), 3),
            "mean_by_item_0_3": {
                item["id"]: round(statistics.mean(row["judge"]["scores"][item["id"]] for row in subset), 3)
                for item in TRANSFER_ITEMS
            },
            "mean_clarity_1_5": round(statistics.mean(row["student"]["experience"]["clarity"] for row in subset), 3),
            "mean_cognitive_load_1_5": round(statistics.mean(row["student"]["experience"]["cognitive_load"] for row in subset), 3),
            "mean_evidence_checkability_1_5": round(statistics.mean(row["student"]["experience"]["evidence_checkability"] for row in subset), 3),
            "mean_trust_1_5": round(statistics.mean(row["student"]["experience"]["trust"] for row in subset), 3),
            "mean_would_reuse_1_5": round(statistics.mean(row["student"]["experience"]["would_reuse"] for row in subset), 3),
            "follow_up_rate": round(
                sum(bool(item.get("needs_follow_up")) for row in subset for item in row["student"]["perceived"])
                / (len(subset) * 3),
                3,
            ),
        }
    paired = []
    for profile in PROFILES:
        a = next(row for row in rows if row["profile_id"] == profile["id"] and row["system"] == "tfidf_rag")
        b = next(row for row in rows if row["profile_id"] == profile["id"] and row["system"] == "graph_rag")
        paired.append({"profile_id": profile["id"], "tfidf": a["judge"]["total"], "graph": b["judge"]["total"], "difference": b["judge"]["total"] - a["judge"]["total"]})
    summary["paired"] = paired
    summary["mean_paired_posttest_difference_graph_minus_tfidf"] = round(statistics.mean(x["difference"] for x in paired), 3)
    return summary


def write_report(
    path: Path,
    run_dir: Path,
    rows: list[dict],
    summary: dict,
    title: str,
    evaluation_scope: str,
):
    table = [
        "| 条件 | n | 后测 /15 | 清晰度 /5 | 认知负荷 /5 | 证据可核查 /5 | 信任 /5 | 愿意复用 /5 | 追问率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system, label in (("tfidf_rag", "TF-IDF RAG"), ("graph_rag", "Graph-RAG")):
        s = summary[system]
        table.append(
            f"| {label} | {s['n']} | {s['mean_posttest_0_15']:.2f} | {s['mean_clarity_1_5']:.2f} | "
            f"{s['mean_cognitive_load_1_5']:.2f} | {s['mean_evidence_checkability_1_5']:.2f} | "
            f"{s['mean_trust_1_5']:.2f} | {s['mean_would_reuse_1_5']:.2f} | {s['follow_up_rate']:.1%} |"
        )
    item_table = [
        "| 题目 | 能力 | TF-IDF /3 | Graph /3 |",
        "|---|---|---:|---:|",
    ]
    item_labels = {
        "T1": "数值辨析：求和 vs 最大值",
        "T2": "错误诊断：任务—模型—算法",
        "T3": "错误诊断：局部贪心 vs 序列解码",
        "T4": "错误诊断：两类动态规划",
        "T5": "迁移纠错：模型非唯一、解码非贪心",
    }
    for item in TRANSFER_ITEMS:
        item_id = item["id"]
        item_table.append(
            f"| {item_id} | {item_labels[item_id]} | "
            f"{summary['tfidf_rag']['mean_by_item_0_3'][item_id]:.2f} | "
            f"{summary['graph_rag']['mean_by_item_0_3'][item_id]:.2f} |"
        )
    profile_lines = [
        f"- `{x['profile_id']}`：TF-IDF {x['tfidf']}/15，Graph {x['graph']}/15，差值 {x['difference']:+d}。"
        for x in summary["paired"]
    ]
    problems = {}
    for row in rows:
        problem = str(row["student"]["experience"].get("main_problem", "")).strip()
        if problem:
            problems[problem] = problems.get(problem, 0) + 1
    problem_lines = [f"- {text}（{count} 次）" for text, count in sorted(problems.items(), key=lambda x: (-x[1], x[0]))[:6]]
    content = f"""# {title}

## 结论先行

这是 **synthetic-user pilot，不是真人教育实验**。4 个 AgentSchool 风格大学生画像分别独立使用 TF-IDF RAG 与 Graph-RAG，共 8 个条件。每个条件先生成受画像约束的记忆痕迹，再开启看不到助手原文的独立后测会话，完成 5 道计算、迁移与错误诊断题。评分员在不知道系统条件的情况下按固定 rubric 评分。

评估范围：{evaluation_scope}

本次 Graph-RAG 相对 TF-IDF RAG 的平均配对后测差值为 **{summary['mean_paired_posttest_difference_graph_minus_tfidf']:+.2f}/15**。样本很小、学生与评分员都由同一基础模型生成，因此只能用来发现交互问题和生成真人实验假设，不能证明学习效果。

## 汇总

{chr(10).join(table)}

## 同画像配对结果

{chr(10).join(profile_lines)}

满分率：TF-IDF RAG 为 {summary['tfidf_rag']['perfect_score_rate']:.1%}，Graph-RAG 为 {summary['graph_rag']['perfect_score_rate']:.1%}。本轮结果仅用于定位不同画像下的交互问题，不据此推断真实学习效果。

## 分题表现

{chr(10).join(item_table)}

## 模拟学生报告的主要问题

{chr(10).join(problem_lines) if problem_lines else '- 未报告。'}

## 方法边界

- 学生画像采用 AgentSchool 的人格/认知状态/情感状态/注意力/学习风格/常见错误维度，并将原本的中学生样例改写为大学 NLP 场景。
- 学习与后测被拆成两个模型调用；后测只接收记忆痕迹，不接收课程助手原文，系统名称始终隐藏。
- 后测评分时条件标签被随机替换为 `S01`—`S08`，但学生 Agent 和评分 Agent 使用同一模型，存在共同偏差。
- 冻结回答来自既有 test run；原始回答、学生输出、盲评结果和 manifest 保存在私有目录 `{run_dir}`，不应随公开仓库发布。
- n=4 对画像、每格只有一次生成，没有统计功效，也没有真实行为、延迟保持或真实满意度证据。

## 下一步真人验证

招募 6—8 名真实同学做 20—30 分钟可用性测试：随机或交叉分配两个系统，保留这组迁移与错误诊断题，记录完成时间、追问、引用核对、think-aloud 困惑，并做 3 天延迟后测。真人结果与本报告必须分开呈现。
"""
    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--report-title",
        default="合成大学生 Agent 形成性预实验 V2（2026-11-24）",
    )
    parser.add_argument(
        "--evaluation-scope",
        default="冻结回答上的形成性预实验；不是独立学习效果估计。",
    )
    parser.add_argument("--protocol-version", type=int, default=2)
    parser.add_argument(
        "--include-evidence-panel",
        action="store_true",
        help="Expose the product evidence cards and graph relations to synthetic users.",
    )
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is missing")
    lessons = load_lessons(
        args.source,
        include_evidence_panel=args.include_evidence_panel,
    )
    run_dir = args.output or ROOT / "evaluation/private_runs" / f"synthetic_student_pilot_v2_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "framework": "AgentSchool profile schema and learning-loop adaptation",
        "framework_source": "https://github.com/epitome-AISS/AgentSchool",
        "framework_license": "MIT",
        "model": MODEL,
        "provider": "deepseek",
        "base_url": API_URL.rsplit("/chat/completions", 1)[0],
        "source_run": str(args.source),
        "lesson_ids": list(LESSON_IDS),
        "profiles": PROFILES,
        "transfer_items": TRANSFER_ITEMS,
        "seed": args.seed,
        "protocol_version": args.protocol_version,
        "include_evidence_panel": args.include_evidence_panel,
        "closed_book_isolation": "separate model call receiving memory trace only",
        "limitations": ["synthetic users", "same base model for students and judge", "one generation per cell", "not a learning-effect estimate"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for profile in PROFILES:
        for system in ("tfidf_rag", "graph_rag"):
            learning_raw, learning_usage = post_model(
                key,
                [
                    {"role": "system", "content": "你是教育研究中的合成学生 Agent。必须遵守学生画像并只输出合法 JSON。"},
                    {"role": "user", "content": learning_prompt(profile, system, lessons)},
                ],
            )
            try:
                learning = extract_json(learning_raw)
                validate_learning_result(learning)
            except Exception as exc:
                (run_dir / "learning_parse_failure.txt").write_text(learning_raw, encoding="utf-8")
                raise RuntimeError(f"Invalid learning result for {profile['id']} {system}: {exc}") from exc
            posttest_raw, posttest_usage = post_model(
                key,
                [
                    {"role": "system", "content": "你是参加闭卷后测的合成学生。只能使用提供的记忆痕迹，并只输出合法 JSON。"},
                    {"role": "user", "content": posttest_prompt(profile, learning["memory_trace"])},
                ],
            )
            try:
                posttest = extract_json(posttest_raw)
                validate_posttest_result(posttest)
            except Exception as exc:
                (run_dir / "posttest_parse_failure.txt").write_text(posttest_raw, encoding="utf-8")
                raise RuntimeError(f"Invalid posttest result for {profile['id']} {system}: {exc}") from exc
            student = {**learning, "posttest": posttest["posttest"]}
            row = {
                "profile_id": profile["id"],
                "system": system,
                "student": student,
                "usage": {"learning": learning_usage, "posttest": posttest_usage},
            }
            rows.append(row)
            with (run_dir / "student_sessions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"completed {profile['id']} {system}", flush=True)

    rng = random.Random(args.seed)
    blinded_order = list(range(len(rows)))
    rng.shuffle(blinded_order)
    blind_map = {row_index: f"S{position + 1:02d}" for position, row_index in enumerate(blinded_order)}
    blinded = [
        {"blind_id": blind_map[index], "answers": rows[index]["student"]["posttest"]}
        for index in blinded_order
    ]
    judge_raw, judge_usage = post_model(
        key,
        [
            {"role": "system", "content": "你是严格、条件盲化的 NLP 课程评分员。只输出合法 JSON。"},
            {"role": "user", "content": judge_prompt(blinded)},
        ],
        max_tokens=3000,
        temperature=0,
    )
    judge_results = extract_json(judge_raw)
    by_blind = {result["blind_id"]: result for result in judge_results}
    for index, row in enumerate(rows):
        result = by_blind[blind_map[index]]
        item_ids = tuple(item["id"] for item in TRANSFER_ITEMS)
        if int(result["total"]) != sum(int(result["scores"][item]) for item in item_ids):
            raise RuntimeError(f"Judge total mismatch for {blind_map[index]}")
        row["blind_id"] = blind_map[index]
        row["judge"] = result
    (run_dir / "blind_judgments.json").write_text(
        json.dumps({"blind_map": blind_map, "results": judge_results, "usage": judge_usage}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = summarize(rows)
    (run_dir / "scored_sessions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = args.report or ROOT / "evaluation/SYNTHETIC_STUDENT_PILOT_V2_20261124.md"
    write_report(
        report,
        run_dir,
        rows,
        summary,
        title=args.report_title,
        evaluation_scope=args.evaluation_scope,
    )
    print(json.dumps({"run_dir": str(run_dir), "report": str(report), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
