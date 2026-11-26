"""Reproducible retrieval ablation for the modern Graph-RAG stack.

The default run is dev-only. Access to the sealed test split requires an
explicit flag so model and fusion choices cannot silently tune on test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from graph_rag.pipeline import extract_grounded_entities
from knowledge_graph.embeddings import FastRPStyleEmbedding
from knowledge_graph.graph import get_kg
from shared.schemas import ChunkItem, SearchResult
from vector_search.hybrid_retriever import (
    CrossEncoderReranker,
    QdrantHybridRetriever,
    SentenceTransformerEncoder,
    expand_neighbor_context,
)
from vector_search.retriever import VectorRetriever


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(
    path: Path,
    split: str = "dev",
    allow_sealed_test: bool = False,
) -> list[dict]:
    if split == "test" and not allow_sealed_test:
        raise ValueError("sealed test需要显式传入--allow-sealed-test，禁止默认读取。")
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [case for case in cases if case.get("split") == split]


def _gold_pages(case: dict) -> set[tuple[str, int]]:
    return {
        (Path(evidence["source"]).stem.replace(" ", "_")[:50], int(page))
        for evidence in case.get("source_evidence", [])
        for page in evidence.get("locators", [])
    }


def _required_relations(case: dict) -> set[tuple[str, str, str]]:
    return {
        (relation["head"], relation["relation"], relation["tail"])
        for relation in case.get("required_relations", [])
    }


def _relation_triples(
    question: str,
    context: list[SearchResult],
    graph_embedding: FastRPStyleEmbedding | None = None,
    graph_embedding_candidates: int = 2,
) -> tuple[set[tuple[str, str, str]], list[dict]]:
    kg = get_kg()
    texts = [result.chunk.text for result in context]
    entities = extract_grounded_entities(question, texts, max_entities=4)
    candidate_details = []
    if graph_embedding is not None:
        candidate_details = [
            {"entity": entity, "similarity": score}
            for entity, score in graph_embedding.similar_nodes(
                entities,
                top_k=graph_embedding_candidates,
            )
        ]
        entities = entities + [item["entity"] for item in candidate_details]

    triples: set[tuple[str, str, str]] = set()
    for entity in entities:
        for triple in kg.get_neighbors(entity, depth=1):
            triples.add((triple["head"], triple["relation"], triple["tail"]))
    return triples, candidate_details


def _reciprocal_rank(
    anchors: list[SearchResult],
    gold_pages: set[tuple[str, int]],
) -> float:
    for rank, result in enumerate(anchors, start=1):
        if (result.chunk.source, result.chunk.page) in gold_pages:
            return 1.0 / rank
    return 0.0


def should_apply_reranker(
    candidates: list[SearchResult],
    tie_margin: float = 0.0,
) -> bool:
    """Route ambiguous sparse retrievals without using labels or test metadata."""
    if len(candidates) < 2:
        return False
    margin = float(candidates[0].similarity) - float(candidates[1].similarity)
    return margin <= tie_margin + 1e-12


def evaluate_configuration(
    cases: list[dict],
    search: Callable[[str], list[SearchResult]],
    search_context: Callable[[str], list[SearchResult]],
    include_graph: bool = False,
    graph_embedding: FastRPStyleEmbedding | None = None,
    graph_embedding_candidates: int = 2,
    rerank_decisions: dict[str, bool] | None = None,
) -> list[dict]:
    rows = []
    for case in cases:
        start = time.perf_counter()
        anchors = search(case["question"])
        context = search_context(case["question"])
        triples: set[tuple[str, str, str]] = set()
        graph_candidates: list[dict] = []
        if include_graph:
            triples, graph_candidates = _relation_triples(
                case["question"],
                context,
                graph_embedding=graph_embedding,
                graph_embedding_candidates=graph_embedding_candidates,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        gold_pages = _gold_pages(case)
        required_relations = _required_relations(case)
        anchor_pages = {(result.chunk.source, result.chunk.page) for result in anchors}
        context_pages = {(result.chunk.source, result.chunk.page) for result in context}
        rows.append(
            {
                "id": case["id"],
                "category": case.get("category"),
                "question": case["question"],
                "gold_pages": len(gold_pages),
                "anchor_hit": bool(gold_pages & anchor_pages) if gold_pages else None,
                "context_hit": bool(gold_pages & context_pages) if gold_pages else None,
                "gold_evidence_recall": (
                    len(gold_pages & context_pages) / len(gold_pages)
                    if gold_pages
                    else None
                ),
                "reciprocal_rank": (
                    _reciprocal_rank(anchors, gold_pages) if gold_pages else None
                ),
                "required_relations": len(required_relations),
                "relation_recall": (
                    len(required_relations & triples) / len(required_relations)
                    if include_graph and required_relations
                    else None
                ),
                "retrieved_triples": len(triples),
                "graph_embedding_candidates": graph_candidates,
                "rerank_applied": (
                    rerank_decisions.get(case["question"])
                    if rerank_decisions is not None
                    else None
                ),
                "anchor_chunks": [result.chunk.chunk_id for result in anchors],
                "context_chunks": [result.chunk.chunk_id for result in context],
                "latency_ms": round(latency_ms, 3),
            }
        )
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    evidence_rows = [row for row in rows if row["gold_pages"]]
    relation_rows = [
        row
        for row in rows
        if row["required_relations"] and row["relation_recall"] is not None
    ]
    routed_rows = [row for row in rows if row.get("rerank_applied") is not None]
    return {
        "cases": len(rows),
        "evidence_cases": len(evidence_rows),
        "evidence_hit_at_3": (
            sum(bool(row["anchor_hit"]) for row in evidence_rows) / len(evidence_rows)
            if evidence_rows
            else None
        ),
        "context_hit_rate": (
            sum(bool(row["context_hit"]) for row in evidence_rows) / len(evidence_rows)
            if evidence_rows
            else None
        ),
        "mean_gold_evidence_recall": (
            sum(float(row["gold_evidence_recall"]) for row in evidence_rows)
            / len(evidence_rows)
            if evidence_rows
            else None
        ),
        "mrr_at_3": (
            sum(float(row["reciprocal_rank"]) for row in evidence_rows)
            / len(evidence_rows)
            if evidence_rows
            else None
        ),
        "relation_cases": len(relation_rows),
        "mean_relation_recall": (
            sum(float(row["relation_recall"]) for row in relation_rows)
            / len(relation_rows)
            if relation_rows
            else None
        ),
        "mean_retrieved_triples": (
            sum(int(row["retrieved_triples"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "mean_latency_ms": (
            sum(float(row["latency_ms"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "rerank_rate": (
            sum(bool(row["rerank_applied"]) for row in routed_rows) / len(routed_rows)
            if routed_rows
            else None
        ),
    }


def _bad_cases(rows: list[dict]) -> list[dict]:
    return [
        {
            "id": row["id"],
            "question": row["question"],
            "anchor_hit": row["anchor_hit"],
            "context_hit": row["context_hit"],
            "relation_recall": row["relation_recall"],
            "anchor_chunks": row["anchor_chunks"],
            "graph_embedding_candidates": row["graph_embedding_candidates"],
        }
        for row in rows
        if row["anchor_hit"] is False
        or (row["relation_recall"] is not None and row["relation_recall"] < 1.0)
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-sealed-test", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--neighbor-window", type=int, default=2)
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--graph-embedding-candidates", type=int, default=2)
    parser.add_argument(
        "--configs",
        default="",
        help="逗号分隔的配置名；留空运行全部配置。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("输出文件已存在，请选择新路径，避免覆盖历史实验。")

    known_configurations = {
        "A_tfidf",
        "B_bge_dense",
        "C_bge_hybrid",
        "D_hybrid_rerank",
        "E_explicit_graph",
        "F_graph_embedding",
        "G_tfidf_rerank_graph",
        "H_adaptive_tfidf_rerank_graph",
    }
    requested = {
        name.strip() for name in args.configs.split(",") if name.strip()
    }
    unknown = requested - known_configurations
    if unknown:
        raise SystemExit(f"未知配置: {', '.join(sorted(unknown))}")

    needs_modern = not requested or bool(
        requested
        & {
            "B_bge_dense",
            "C_bge_hybrid",
            "D_hybrid_rerank",
            "E_explicit_graph",
            "F_graph_embedding",
        }
    )
    needs_reranker = not args.skip_reranker and (
        not requested
        or bool(
            requested
            & {
                "D_hybrid_rerank",
                "E_explicit_graph",
                "F_graph_embedding",
                "G_tfidf_rerank_graph",
                "H_adaptive_tfidf_rerank_graph",
            }
        )
    )

    questions_path = ROOT / "evaluation" / "questions.jsonl"
    chunks_path = ROOT / "backend" / "vector_search" / "index" / "chunks.json"
    cases = load_cases(
        questions_path,
        split=args.split,
        allow_sealed_test=args.allow_sealed_test,
    )
    chunks = [
        ChunkItem(**item)
        for item in json.loads(chunks_path.read_text(encoding="utf-8"))
    ]

    tfidf = VectorRetriever()
    tfidf._save = lambda _: None
    tfidf.build_index(chunks)

    reranker = None
    if needs_reranker:
        reranker = CrossEncoderReranker(args.reranker_model, device=args.device)
    modern = None
    if needs_modern:
        encoder = SentenceTransformerEncoder(args.embedding_model, device=args.device)
        modern = QdrantHybridRetriever(chunks, encoder, reranker=reranker)

    def modern_search(mode: str, rerank: bool, context: bool):
        if modern is None:
            raise RuntimeError("当前配置未加载BGE/Qdrant检索器。")
        if context:
            return lambda question: modern.search_context(
                question,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                neighbor_window=args.neighbor_window,
                mode=mode,
                rerank=rerank,
            )
        return lambda question: modern.search(
            question,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            mode=mode,
            rerank=rerank,
        )

    configurations = {
        "A_tfidf": {
            "search": lambda question: tfidf.search(question, top_k=args.top_k),
            "context": lambda question: tfidf.search_context(
                question,
                top_k=args.top_k,
                neighbor_window=args.neighbor_window,
            ),
            "include_graph": False,
            "graph_embedding": None,
        },
    }
    if modern is not None:
        configurations["B_bge_dense"] = {
            "search": modern_search("dense", False, False),
            "context": modern_search("dense", False, True),
            "include_graph": False,
            "graph_embedding": None,
        }
        configurations["C_bge_hybrid"] = {
            "search": modern_search("hybrid", False, False),
            "context": modern_search("hybrid", False, True),
            "include_graph": False,
            "graph_embedding": None,
        }

    best_mode_name = "C_bge_hybrid"
    if reranker is not None and modern is not None:
        configurations["D_hybrid_rerank"] = {
            "search": modern_search("hybrid", True, False),
            "context": modern_search("hybrid", True, True),
            "include_graph": False,
            "graph_embedding": None,
        }
        best_mode_name = "D_hybrid_rerank"

    if reranker is not None:
        tfidf_rerank_cache: dict[str, list[SearchResult]] = {}

        def tfidf_rerank(question: str) -> list[SearchResult]:
            if question in tfidf_rerank_cache:
                return list(tfidf_rerank_cache[question])
            candidates = tfidf.search(question, top_k=args.candidate_k)
            scores = reranker.score(
                question,
                [result.chunk.text for result in candidates],
            )
            ranked = [
                SearchResult(
                    chunk=result.chunk,
                    similarity=round(float(score), 6),
                    retrieval_reason="semantic",
                )
                for result, score in sorted(
                    zip(candidates, scores),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )[: args.top_k]
            ]
            tfidf_rerank_cache[question] = ranked
            return list(ranked)

        configurations["G_tfidf_rerank_graph"] = {
            "search": tfidf_rerank,
            "context": lambda question: expand_neighbor_context(
                chunks,
                tfidf_rerank(question),
                neighbor_window=args.neighbor_window,
            ),
            "include_graph": True,
            "graph_embedding": None,
        }

        adaptive_cache: dict[str, list[SearchResult]] = {}
        adaptive_decisions: dict[str, bool] = {}

        def adaptive_tfidf_rerank(question: str) -> list[SearchResult]:
            if question in adaptive_cache:
                return list(adaptive_cache[question])
            candidates = tfidf.search(question, top_k=args.candidate_k)
            use_reranker = should_apply_reranker(candidates)
            adaptive_decisions[question] = use_reranker
            if use_reranker:
                scores = reranker.score(
                    question,
                    [result.chunk.text for result in candidates],
                )
                ranked = [
                    SearchResult(
                        chunk=result.chunk,
                        similarity=round(float(score), 6),
                        retrieval_reason="semantic",
                    )
                    for result, score in sorted(
                        zip(candidates, scores),
                        key=lambda item: float(item[1]),
                        reverse=True,
                    )[: args.top_k]
                ]
            else:
                ranked = candidates[: args.top_k]
            adaptive_cache[question] = ranked
            return list(ranked)

        configurations["H_adaptive_tfidf_rerank_graph"] = {
            "search": adaptive_tfidf_rerank,
            "context": lambda question: expand_neighbor_context(
                chunks,
                adaptive_tfidf_rerank(question),
                neighbor_window=args.neighbor_window,
            ),
            "include_graph": True,
            "graph_embedding": None,
            "rerank_decisions": adaptive_decisions,
        }

    if modern is not None:
        configurations["E_explicit_graph"] = {
            **configurations[best_mode_name],
            "include_graph": True,
            "graph_embedding": None,
        }
        configurations["F_graph_embedding"] = {
            **configurations[best_mode_name],
            "include_graph": True,
            "graph_embedding": FastRPStyleEmbedding(
                get_kg().graph,
                dimensions=32,
                random_seed=42,
            ),
        }

    if requested:
        missing = requested - set(configurations)
        if missing:
            raise SystemExit(
                "所选配置与--skip-reranker冲突或缺少依赖: "
                + ", ".join(sorted(missing))
            )
        configurations = {
            name: configurations[name]
            for name in args.configs.split(",")
            if name in configurations
        }

    rows_by_configuration = {}
    summaries = {}
    bad_cases = {}
    for name, config in configurations.items():
        print(f"[experiment] running {name}", flush=True)
        rows = evaluate_configuration(
            cases,
            config["search"],
            config["context"],
            include_graph=config["include_graph"],
            graph_embedding=config["graph_embedding"],
            graph_embedding_candidates=args.graph_embedding_candidates,
            rerank_decisions=config.get("rerank_decisions"),
        )
        rows_by_configuration[name] = rows
        summaries[name] = summarize_rows(rows)
        bad_cases[name] = _bad_cases(rows)
        print(json.dumps({name: summaries[name]}, ensure_ascii=False), flush=True)

    result = {
        "experiment": "retrieval_upgrade_ablation_v1",
        "split": args.split,
        "sealed_test_explicitly_enabled": bool(args.allow_sealed_test),
        "cases": len(cases),
        "chunk_count": len(chunks),
        "models": {
            "embedding": args.embedding_model if needs_modern else None,
            "reranker": args.reranker_model if needs_reranker else None,
            "device": args.device,
        },
        "parameters": {
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "neighbor_window": args.neighbor_window,
            "graph_embedding_candidates": args.graph_embedding_candidates,
        },
        "hashes": {
            "questions_sha256": _sha256(questions_path),
            "chunks_sha256": _sha256(chunks_path),
            "script_sha256": _sha256(Path(__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "summaries": summaries,
        "bad_cases": bad_cases,
        "rows": rows_by_configuration,
        "interpretation_boundary": (
            "F使用离线FastRP风格结构向量，仅扩展候选实体；相似性不是事实关系。"
            "只有显式图边计入required relation命中。"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[experiment] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
