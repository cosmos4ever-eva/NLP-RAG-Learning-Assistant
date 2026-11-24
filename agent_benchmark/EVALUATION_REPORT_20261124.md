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

## Version 0.2 harder-task extension

Three adversarial tasks were added after both configurations passed the five foundation tasks:

1. `partial-evidence-policy`: a two-file contract that retrieves each subquestion exactly once, generates only supported parts, preserves order, and reports `answered` / `partial` / `insufficient_evidence` correctly.
2. `hybrid-retrieval-regression`: simultaneous stopword removal, configured synonym expansion, broad-parent downweighting, rarity-aware ranking, clause coverage, neighboring pages, and context budgets.
3. `leakage-safe-evaluation`: dev/test sealing, gold-field sanitization before generation, complete three-system matrices, immutable inputs, and digest-bound adjudication overrides.

All eight broken baselines fail and all eight reference implementations pass. All eight task packages pass `ageval lock`.

### Hard-task compatibility pilot

| Configuration | Task PASS | Hidden checks passed | Mean time/task | Mean tool calls/task |
|---|---:|---:|---:|---:|
| DeepSeek Flash | 1/3 (33.3%) | 19/23 (82.6%) | 10.22 s | 8.33 |
| DeepSeek V4 Pro | 1/3 (33.3%) | 14/23 (60.9%) | 46.59 s | 6.00 |

Both configurations passed `partial-evidence-policy` and failed the other two tasks. Flash came closer on the failed tasks: it missed only the directory-vs-specific ranking check in hybrid retrieval, while Pro also missed clause coverage and neighbor budgeting. In leakage-safe evaluation, Flash correctly sealed gold fields, produced the ordered matrix, preserved inputs, and rejected invalid matrices, but omitted the required manifest count and digest-bound adjudication semantics. Pro spent the available interaction budget reasoning about the task and did not update the two implementation files.

This harder group now distinguishes partial progress even though binary task pass rates remain tied. It also exposes a meaningful agent failure mode: extensive reasoning without completing an artifact before the turn budget expires.

### Completion-guard repair and controlled rerun

The pilot harness previously treated every text-only model response as final. In the original V4 Pro run, the two failed tasks ended after long responses reached the 4,096-token completion limit, before the model wrote or tested an implementation. This was a harness interaction failure, not sufficient evidence that the model was intrinsically worse.

The repaired harness now:

- continues after `finish_reason=length`;
- refuses completion before at least one successful implementation write and one agent-initiated public-test run;
- records model finish reasons, token usage, writes, tests, correction prompts, termination reason, and failure class;
- keeps the original 12-turn limit and runs the independent evaluator only after the attempt ends.

The same model identifiers and three hard tasks were then rerun once:

| Configuration | Task PASS | Hidden checks passed | Mean time/task | Mean tool calls/task | Guard corrections |
|---|---:|---:|---:|---:|---:|
| DeepSeek Flash | 1/3 (33.3%) | 18/23 (78.3%) | 9.73 s | 8.00 | 0 |
| DeepSeek V4 Pro | 1/3 (33.3%) | 19/23 (82.6%) | 53.93 s | 7.67 | 2 |

V4 Pro's two 4,096-token responses were caught and redirected to tool use. It then wrote implementation files and ran public tests on both formerly no-action tasks, raising its deterministic hidden-check result from 14/23 to 19/23. Binary task PASS remains 1/3 because the resulting implementations still missed hidden requirements. Flash changed from 19/23 to 18/23 on a single run, showing why repeated attempts are still required before ranking models.

The corrected conclusion is therefore narrower: the earlier 60.9% V4 Pro hidden-check result was confounded by premature harness termination. With that confound removed, V4 Pro no longer trails Flash on hidden checks in this one-run comparison, although it remains much slower and neither model solves the two hardest tasks completely.
