# Dev Evaluation and Graph-RAG Iteration — 2026-11-24

## Scope and leakage boundary

- Dataset validation: 56 cases, with 28 dev and 28 sealed test cases.
- This iteration executed dev only. No test generation or scoring was run.
- The existing 28-case DeepSeek run still matches the current dataset, parser, retriever, Graph-RAG pipeline, graph, and experiment-runner hashes.
- Raw prompts, answers, and course excerpts remain under `evaluation/private_runs/` and are not publishable artifacts.

## Confirmed 28-case dev baseline

- 28 retrieval rows and 84 answers: 28 each for LLM-only, TF-IDF RAG, and Graph-RAG.
- Evidence-bearing cases: 24; boundary/abstention cases: 4.
- Context hit rate: 24/24 (100%).
- Top-3 anchor hit rate: 22/24 (91.7%).
- Adjudicated mean score: LLM-only 5.43/12, TF-IDF RAG 11.11/12, Graph-RAG 11.14/12.
- Adjudicated boundary compliance: TF-IDF RAG 100%, Graph-RAG 100%, LLM-only 0%.

The paired TF-IDF/Graph-RAG scores were 1 Graph-RAG win, 0 losses, and 27 ties. The only gain was one relation point on EVAL-018. The baseline therefore supports parity or a slight graph benefit, not a large graph advantage.

## Retrieval sensitivity

The deterministic dev-only sensitivity artifact is `retrieval_sensitivity_dev_20261124.json`.

| Retrieval configuration | Context hits | Mean gold-evidence recall | Mean context chunks |
|---|---:|---:|---:|
| top-3, no neighbors | 24/24 | 42.95% | 3.00 |
| top-3, neighbor window 2 | 24/24 | 86.45% | 10.82 |
| top-5, no neighbors | 24/24 | 53.74% | 5.00 |
| top-5, neighbor window 2 | 24/24 | 80.61% | 11.89 |

Adding neighbors at top-3 improved evidence recall on 22 cases and regressed none. Increasing top-k from 3 to 5 under the fixed 12-chunk context budget improved no cases and regressed seven, because extra semantic anchors displaced neighboring evidence. The retained configuration is therefore top-3 with neighbor window 2; no paid top-5 generation was justified.

## Grounded graph lookup iteration

Root cause: graph lookup used only entity mentions in the question even when the retrieved course chunks contained the missing graph entity. Three relation-bearing dev cases were incomplete: EVAL-007, EVAL-015, and EVAL-017.

Implemented policy:

1. Prefer explicit entities in the question.
2. Fill remaining entity slots from retrieved course text.
3. If the question already contains graph entities, accept only context entities directly adjacent to them in the graph.
4. Limit lookup to four entities and 15 unique triples.

| Version | Complete relation cases | Mean required-relation recall | Mean triples | Maximum triples |
|---|---:|---:|---:|---:|
| Question-only baseline | 15/18 | 86.11% | 2.46 | 11 |
| Unfiltered context seed | 18/18 | 100% | 9.43 | 22 |
| Filtered v3 candidate | 18/18 | 100% | 5.68 | 15 |

The filtered policy preserves complete required-relation coverage while removing unrelated neighbor-derived graph topics such as smoothing relations from the Naive Bayes versus logistic-regression question.

## Graph-RAG-only end-to-end check

The v3 candidate generated 28/28 non-empty Graph-RAG answers with DeepSeek Flash. A fresh DeepSeek V4 Pro rubric pass produced:

- Answer cases only: 11.33/12, versus 11.00/12 for the baseline model-scored Graph-RAG answers.
- `course_grounded`: 11.30/12, versus 11.00/12.
- `cross_concept`: 11.50/12, versus 11.00/12.
- Paired result: 5 wins, 4 losses, 19 ties; net change +4 points across 28 cases.
- Mean generation latency: 4.75 seconds; total generation usage: 54,925 tokens.

This is supporting evidence, not a final quality claim. Generation was repeated and the judge was a single model pass, so answer and judge variance are confounded. The boundary scores were visibly inconsistent with the rubric: all four answers explicitly refused the unsupported request and identified missing material, yet the judge assigned zero total to the group. Boundary acceptance must therefore use deterministic checks plus human adjudication rather than the raw judge mean.

## Decision and remaining gate

The v3 graph lookup is accepted as a dev candidate because its deterministic relation recall improves from 86.11% to 100% without retrieval regression, and its graph size is bounded. It is not yet frozen for the sealed test split.

Before test evaluation:

1. Human-review the changed answer cases EVAL-003, 004, 006, 010, 017, 018, and 053.
2. Re-adjudicate the four fresh boundary answers; do not reuse answer-specific overrides from the older run.
3. Repeat Graph-RAG generation/scoring at least twice or use a second human annotator to separate generation variance from graph-policy impact.
4. Freeze code, prompt, dataset, graph, and model configuration hashes, then run the sealed 28-case test once.

## Run-specific adjudication and repeatability gate

Eleven changed or boundary cases from the first v3 run were reviewed against the rubric. Each override is bound to the exact answer SHA-256; stale overrides now stop the adjudication command instead of silently applying. The review is labeled `assistant_rubric_review`, not human annotation.

- Run-specific adjudicated Graph-RAG mean: 11.46/12.
- `course_grounded`: 11.55/12.
- `cross_concept`: 11.25/12.
- `boundary`: 11.25/12.
- Strict boundary compliance: 3/4. EVAL-045 correctly refused the unsupported BLEU premise but added a small unsupported BLEU explanation, so it received boundary=1 rather than 2.

Two additional independent Graph-RAG generations and judge passes were then run with the same v3 configuration. The safe aggregate is stored in `repeat_stability_dev_20261124.json`.

| Run | Mean on 24 answer cases | Generation tokens | Mean latency |
|---|---:|---:|---:|
| v3 run 1 | 11.33 | 54,925 | 4.75 s |
| v3 run 2 | 10.96 | 55,491 | 4.91 s |
| v3 run 3 | 11.33 | 55,628 | 4.80 s |

Across the three runs, the answer-case mean was 11.21/12 with sample standard deviation 0.22 and range 10.96–11.33. No case produced byte-identical answers in all three runs. Nine cases had different judge totals; EVAL-010 ranged from 6 to 10 and EVAL-003 ranged from 9 to 12.

The earlier baseline model-scored answer mean was 11.00/12. The repeated v3 mean is only +0.21, approximately one observed run-level standard deviation. Therefore the end-to-end quality gain is not yet strong enough to claim a stable improvement. The accepted result is narrower: v3 deterministically improves required-relation recall from 86.11% to 100% with bounded graph context and no retrieval regression. A second independent annotator remains necessary before freezing for test.
