# Frozen Test Evaluation — 2026-11-24

## Freeze and protocol

- Frozen commit: `8c55b6d` (`feat: add sealed test evaluation entrypoint`).
- Test was generated and scored once after the dev configuration was frozen.
- Split: 28 test cases; no test case was used for tuning.
- Systems: LLM-only, TF-IDF RAG, Graph-RAG.
- Generator: DeepSeek Flash, non-thinking mode, max 1,200 tokens.
- Judge: DeepSeek V4 Pro, one model-assisted rubric pass.
- Retrieval: top-3 semantic anchors, neighbor window 2, maximum 12 context chunks.
- Raw prompts, answers, judge traces, and course excerpts remain in ignored `evaluation/private_runs/`.

## Retrieval and graph results

- Evidence-bearing cases: 24; boundary cases: 4.
- Top-3 anchor hit: 20/24 (83.3%).
- Expanded-context hit: 22/24 (91.7%).
- Required graph relations: 20/22 cases complete; mean recall 93.2%.
- Retrieval misses: EVAL-031 and EVAL-038 had no gold page in expanded context.
- Partial anchor/context cases: EVAL-029 and EVAL-056 were recovered only after context expansion.

This is a real generalization drop from dev, where expanded-context hit was 24/24 and required-relation recall was 100%. The frozen system was not changed after observing it.

## Final scores

The raw judge correctly recognized RAG refusal behavior but assigned compliant abstentions only 2/12 because it set correctness, groundedness, and relation to zero. Eight RAG boundary answers were therefore reviewed under the same rubric used for dev and bound to their exact answer SHA-256. No answer-case test score was overridden.

| System | Final mean | Answer-case mean | Course-grounded | Cross-concept | Boundary |
|---|---:|---:|---:|---:|---:|
| LLM-only | 5.57 | 6.50 | 6.65 | 6.14 | 0.00 |
| TF-IDF RAG | 9.89 | 9.96 | 10.76 | 8.00 | 9.50 |
| Graph-RAG | 10.00 | 10.08 | 10.76 | 8.43 | 9.50 |

Additional raw answer-case diagnostics:

| System | Grounded accuracy | Relation accuracy | Mean latency | Generation tokens |
|---|---:|---:|---:|---:|
| LLM-only | 0.0% | 81.8% | 6.87 s | 30,156 |
| TF-IDF RAG | 66.7% | 77.3% | 5.54 s | 51,636 |
| Graph-RAG | 70.8% | 81.8% | 5.98 s | 55,000 |

LLM-only grounded accuracy is zero under the course-evidence rubric because it is allowed to use model knowledge but has no supplied course evidence. It should not be interpreted as general factual accuracy.

## Graph-RAG versus TF-IDF RAG

- Paired outcomes: Graph-RAG 1 win, 0 losses, 27 ties.
- Net score difference: +3 points across 28 cases.
- Mean difference: +0.11/12.
- The only win was EVAL-038, where graph relations helped connect POS tagging, HMM, and Viterbi despite a retrieval miss.

The test result matches the dev conclusion: Graph-RAG is at parity with or slightly above TF-IDF RAG, but the experiment does not support a large graph advantage.

## Boundary behavior

- All four TF-IDF and all four Graph-RAG answers explicitly refused the unsupported request and requested missing material.
- All eight also included some course-external explanatory detail because the fixed answer template requires a concept chain and learning suggestion.
- Strict `boundary=2` compliance was therefore 0/4 for each RAG system; the answers received `boundary=1` rather than being counted as complete failures.
- LLM-only answered all four unsupported questions substantively and had 0/4 refusal success.

This identifies a product-level iteration for the next version: abstention responses should bypass the ordinary three-section educational template and return a short refusal plus required-material list. That change must be evaluated in a new version, not retrofitted into this frozen test result.

## Representative bad cases

1. **EVAL-024 — incomplete alternative path**: retrieval hit evidence, but both RAG systems explained One-hot limitations without completing the expected context-vector, weighting, dimensionality-reduction, and cosine-similarity path. Both scored 6/12.
2. **EVAL-031 — retrieval failure**: no HMM basic-problem page reached the expanded context. Both systems safely refused and scored 5/12 instead of hallucinating.
3. **EVAL-038 — graph-only recovery**: no gold page was retrieved, but graph relations supported the POS-tagging → HMM → Viterbi hierarchy. Graph-RAG scored 9 versus TF-IDF RAG 6. This is the only clear graph win.
4. **EVAL-040 — low evidence recall**: context hit at least one expected page but gold-page recall was only 6.25%; both systems produced an incomplete learning path and scored 5/12.
5. **EVAL-056 — partial context recovery**: neighboring pages recovered some negative-sampling evidence, but neither system explained the positive/negative example update mechanism. Both scored 9/12.

## Accepted conclusion and limitations

The frozen test supports three claims:

1. Course-grounded RAG substantially outperforms LLM-only under a course-evidence rubric.
2. Neighbor expansion is useful but does not eliminate retrieval failures on unseen test questions.
3. Graph-RAG offers a small, localized benefit over TF-IDF RAG; its added latency and token use are not justified by a broad quality gain in this dataset.

Limitations:

- There was no second independent human annotator; eight boundary scores use answer-bound assistant rubric review.
- Generation and LLM judging are stochastic.
- The test set has 28 cases and only four boundary cases.
- This is a local course-material evaluation, not evidence of general educational effectiveness.
