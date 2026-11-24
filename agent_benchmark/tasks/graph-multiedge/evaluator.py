import json
import tempfile
from pathlib import Path


def evaluate(inputs):
    submission = json.loads(Path(inputs["artifacts"]["submission"]).read_text(encoding="utf-8"))
    source = submission.get("files", {}).get("graph.py")
    checks = {"source_present": isinstance(source, str), "public_tests_passed": bool(submission.get("public_tests", {}).get("ok"))}
    if isinstance(source, str):
        namespace = {}
        try:
            exec(compile(source, "graph.py", "exec"), namespace)
            graph = namespace["KnowledgeGraph"]()
            got = graph.get_neighbors("分布式表示")
            keys = {(x["head"], x["relation"], x["tail"]) for x in got}
            checks["all_parallel_relations"] = {
                ("分布式表示", "相关概念", "One-hot编码"),
                ("分布式表示", "容易混淆", "One-hot编码"),
                ("词嵌入", "相关概念", "分布式表示"),
            } <= keys
            checks["duplicates_removed"] = len(got) == len(keys) == 3
        except Exception:
            checks["all_parallel_relations"] = False
            checks["duplicates_removed"] = False
    passed = all(checks.values())
    return {"status": "PASS" if passed else "FAIL", "score": 1.0 if passed else 0.0, "metrics": checks, "checks": checks}
