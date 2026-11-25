# Learning Agent Acceptance Evaluation — 2026-11-25

## Protocol

- Frozen dataset: `evaluation/agent_acceptance_v1.jsonl`
- Frozen dataset commit: `4fee24a`
- Git object hash: `fb9295b29e57b363eebe1dbc218b4f247f58e090`
- Tasks: 10 total — 4 single-tool, 4 multi-tool, 2 evidence-boundary tasks
- Model: DeepSeek Flash for planning, tool generation, and the separate reflection role
- Limits: maximum 5 executor steps and 1 reflection retry per task
- Run policy: one run after the task file and metric code were committed; no post-result tuning
- Raw answers, tool arguments, reviews, and traces: ignored under
  `evaluation/private_runs/agent_acceptance_v1_20261125/results.json`

A task passes only when the final status is `completed`, every required tool is
successfully called, no tool call fails, and a boundary task contains an explicit refusal.

## Results

| Metric | Result |
|---|---:|
| Deterministic task pass | 10/10 (100%) |
| End-to-end completed | 10/10 (100%) |
| Required-tool recall | 15/15 (100%) |
| Tool calls completed successfully | 17/17 (100%) |
| Mean unique-tool precision | 100% |
| Evidence-boundary refusal | 2/2 (100%) |
| First-pass review acceptance | 10/10 (100%) |
| Reflection retries triggered | 0/10 |
| Mean end-to-end latency | 10.31 s |
| Median end-to-end latency | 9.40 s |

The reflection role assigned mean goal completion 1.00, mean grounding 0.985,
and mean tool reliability 1.00. These model-assisted scores are supporting
diagnostics rather than independent ground truth.

## Per-task trace summary

| Task | Category | Required/called tools | Attempts | Latency | Pass |
|---|---|---|---:|---:|---:|
| AGENT-001 | single | `course_qa` | 1 | 8.94 s | yes |
| AGENT-002 | single | `concept_compare` | 1 | 10.95 s | yes |
| AGENT-003 | single | `learning_path` | 1 | 4.70 s | yes |
| AGENT-004 | single | `generate_quiz` | 1 | 7.80 s | yes |
| AGENT-005 | multi | `course_qa → concept_compare → generate_quiz` | 1 | 19.43 s | yes |
| AGENT-006 | multi | `learning_path → generate_quiz` | 1 | 9.86 s | yes |
| AGENT-007 | multi | `concept_compare → create_todo ×2` | 1 | 13.98 s | yes |
| AGENT-008 | multi | `learning_report → create_todo` | 1 | 6.35 s | yes |
| AGENT-009 | boundary | `course_qa ×2` | 1 | 13.22 s | yes |
| AGENT-010 | boundary | `course_qa` | 1 | 7.90 s | yes |

AGENT-007 created one separate Todo for each compared model. AGENT-009 used the
second course query to confirm, item by item, that no BERT/Transformer evidence
was present. The arguments differed, so neither was a blocked duplicate loop.

Both boundary answers explicitly stated that the course evidence was insufficient,
did not provide the requested BERT or BLEU formulas, identified the missing
materials, and distinguished unrelated retrieved course fragments from valid evidence.

## What this evaluation supports

1. The current executor can route covered single- and multi-step learning goals to
   the required allowlisted tools in this 10-task acceptance set.
2. All observed tool calls completed successfully and all tasks were accepted by
   the reflection role on the first attempt.
3. The Agent preserved evidence-boundary refusal behavior on both dedicated cases.
4. Agent latency is materially different from the earlier 5.98-second Graph-RAG
   generation latency: 10.31 seconds here includes planning, tool execution,
   final synthesis, and reflection review.

## Limitations

- Ten tasks are sufficient for an acceptance smoke test, not a broad benchmark.
- The task content uses the existing NLP course domain and is not an external blind set.
- DeepSeek Flash served both executor and reviewer roles; the reviewer is role-separated,
  not model-independent or human-independent.
- No frozen task triggered a reflection retry, so this run does not estimate reflection
  recovery rate. Retry behavior is covered by deterministic integration tests, and a
  separate live development case previously demonstrated `revise → pass`, but that case
  is not part of the frozen acceptance score.
- Refusal was checked by deterministic markers plus post-run answer inspection; there
  was no second independent human annotator.

## Resume-safe claim

> 构建基于DeepSeek的执行Agent与评测/反思Agent协作流程，将课程问答、概念对比、
> 学习路径、自动出题、学习诊断和Todo封装为6类工具；在冻结的10题Agent验收集中，
> 端到端任务通过率、必需工具召回率和工具调用成功率均为100%，2/2道证据不足任务
> 均明确拒答，平均端到端延迟10.31秒。
