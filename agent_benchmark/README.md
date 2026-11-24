# NLP-RAG Maintenance Benchmark

An ageval-compatible coding-agent benchmark derived from real maintenance failures in the NLP course learning assistant. It evaluates the coding agent, not the assistant's answer quality. Version 0.2 contains five foundation tasks and three adversarial tasks.

## Tasks

1. `graph-multiedge`: preserve parallel knowledge-graph relationships.
2. `evidence-abstention`: do not call the generator when no evidence exists.
3. `multi-question-retrieval`: cover every subquestion and nearby context.
4. `api-evidence-schema`: align API evidence fields with frontend consumption.
5. `reproducible-eval-runner`: emit all three baselines and support retrieval-only runs.
6. `partial-evidence-policy`: make per-subquestion generation decisions while preserving a cross-file response contract.
7. `hybrid-retrieval-regression`: satisfy stopword, rarity, synonym, clause, neighbor, and budget constraints together.
8. `leakage-safe-evaluation`: seal test cases and gold fields, validate answer matrices, and prevent stale adjudication reuse.

Every task contains a broken implementation in `data/`, public smoke tests, an independent `evaluator.py`, and hidden reference material under `evaluation/`. Deterministic evaluator checks alone decide PASS/FAIL.

## Official ageval commands

```powershell
ageval lock agent_benchmark --task graph-multiedge
ageval run agent_benchmark --task graph-multiedge
ageval run agent_benchmark --profiles agent_benchmark/profiles.deepseek-pro.yaml
```

On native Windows, ageval 0.10.0's local runtime currently requires `socket.AF_UNIX`; use Linux/WSL/Docker for official Attempts. `pilot_runner.py` is a clearly labeled local compatibility runner for validating the same task instructions and evaluators; its outputs are not presented as official ageval evidence.
