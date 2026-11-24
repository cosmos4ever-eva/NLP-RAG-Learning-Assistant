# NLP Learning Assistant: Two-Part Evaluation Report

Date: 2026-11-24

## Part 1 — NLP-RAG product evaluation

Evaluated object: the learning assistant itself.

Dev-only acceptance results (the test split remains sealed):

- Automated tests: 23/23 passed.
- Evidence context hit rate: 24/24 (100%).
- Top-3 anchor hit rate: 22/24 (91.7%).
- Required graph relations: 15/18 cases complete; mean recall 86.1%.
- DeepSeek answers: 84/84 present and non-empty.
- Adjudicated mean score: LLM-only 5.43/12; TF-IDF RAG 11.11/12; Graph-RAG 11.14/12.
- Boundary abstention compliance: TF-IDF RAG 100%; Graph-RAG 100%; LLM-only 0%.

Conclusion: the repaired RAG modes satisfy the dev acceptance gates. Graph-RAG is only marginally above TF-IDF RAG, so the current evidence supports parity/slight improvement, not a large graph advantage.

## Part 2 — Coding-agent maintenance benchmark

Evaluated object: a tool-using Coding Agent repairing real failure patterns extracted from Part 1.

Tasks:

1. Preserve parallel knowledge-graph relations.
2. Abstain without invoking generation when evidence is absent.
3. Retrieve evidence for every subquestion and add neighboring context.
4. Align API evidence fields with frontend consumption.
5. Produce a reproducible three-system evaluation matrix with a zero-generation retrieval-only mode.

### Dataset and evaluator validation

- All five broken baselines: FAIL.
- All five hidden reference implementations: PASS.
- All five tasks pass `ageval lock` with ageval-cli 0.10.0.
- Deterministic evaluators decide PASS/FAIL; no LLM judge is used.

### Compatibility pilot results

The same task instructions and evaluators were run with a local compatibility harness because ageval 0.10.0's local environment uses `socket.AF_UNIX`, which is unavailable in the installed native-Windows Python runtime. These are therefore pilot results, not official ageval Attempts.

| Configuration | Passed | Pass rate | Mean time/task | Mean tool calls/task |
|---|---:|---:|---:|---:|
| DeepSeek Flash | 5/5 | 100% | 7.88 s | 6.2 |
| DeepSeek V4 Pro | 5/5 | 100% | 27.96 s | 6.2 |

Both configurations passed every task. Flash was approximately 3.55x faster in this single-run pilot. Because there was only one attempt per task and both achieved 100%, the benchmark currently does not distinguish their functional capability. A harder second version and repeated trials are required before model-ranking claims.

### Evaluator iteration found during the pilot

The initial abstention evaluator accepted only Chinese keywords and only `answer/message` fields. Both agents implemented semantically correct alternatives in English or structured fields (`explanation`, `material_needed`). The evaluator was broadened to accept equivalent, explicit evidence-shortage and required-material statements. Saved submissions were regraded without new model calls.

## Overall interpretation

- Part 1 demonstrates RAG evaluation, groundedness, retrieval diagnostics, boundary behavior, and system iteration.
- Part 2 demonstrates benchmark construction, task isolation, hidden deterministic evaluation, tool trajectories, and model/harness comparison.
- For an Agent Evaluation role, Part 2 is the primary artifact; Part 1 supplies the authentic system failures and domain evidence that make the benchmark credible.

## Remaining acceptance gap

Run the dataset on Linux/WSL/Docker with official `ageval run`, preferably with at least two distinct Agent harnesses and three attempts per task. Preserve official `lock.json`, `result.json`, and `trajectory.jsonl` before presenting the experiment as a full ageval comparison.
