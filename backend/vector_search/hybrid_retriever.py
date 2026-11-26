"""Modern retrieval adapter used by the dense/hybrid retrieval experiment.

The public interface mirrors the existing retriever while hiding model loading,
Qdrant indexing, dense/sparse fusion, reranking, and neighbor-page expansion.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import TfidfVectorizer

from shared.schemas import ChunkItem, SearchResult
from vector_search.retriever import _jieba_tokenizer, _prepare_query


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one dense vector per text."""


class PairReranker(Protocol):
    def score(self, query: str, passages: list[str]) -> Sequence[float]:
        """Return one relevance score per query/passage pair."""


class SentenceTransformerEncoder:
    """Lazy sentence-transformers adapter with normalized output vectors."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 32,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )


class CrossEncoderReranker:
    """Lazy sentence-transformers cross-encoder adapter."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        pairs = [(query, passage) for passage in passages]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return np.asarray(scores).reshape(-1).astype(float).tolist()


def _as_sparse_vector(row) -> models.SparseVector:
    row = row.tocsr()
    return models.SparseVector(
        indices=row.indices.astype(int).tolist(),
        values=row.data.astype(float).tolist(),
    )


def expand_neighbor_context(
    chunks: list[ChunkItem],
    anchors: list[SearchResult],
    neighbor_window: int = 2,
    max_context_chunks: int = 12,
    max_neighbor_chunks_per_page: int = 1,
) -> list[SearchResult]:
    """Add nearby pages without changing the semantic-anchor ranking."""
    results = list(anchors)
    seen = {result.chunk.chunk_id for result in anchors}
    neighbors = []
    for anchor_rank, anchor in enumerate(anchors):
        for chunk in chunks:
            distance = abs(chunk.page - anchor.chunk.page)
            if (
                chunk.source == anchor.chunk.source
                and 0 < distance <= neighbor_window
                and chunk.chunk_id not in seen
            ):
                neighbors.append((distance, anchor_rank, chunk.page, chunk))

    page_counts: dict[tuple[str, int], int] = {}
    for _, anchor_rank, _, chunk in sorted(neighbors, key=lambda item: item[:3]):
        page_key = (chunk.source, chunk.page)
        if chunk.chunk_id in seen or len(results) >= max_context_chunks:
            continue
        if page_counts.get(page_key, 0) >= max_neighbor_chunks_per_page:
            continue
        seen.add(chunk.chunk_id)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        results.append(
            SearchResult(
                chunk=chunk,
                similarity=0.0,
                retrieval_reason="neighbor",
            )
        )
    return results


class QdrantHybridRetriever:
    """Dense/sparse retrieval with optional reranking and page expansion."""

    def __init__(
        self,
        chunks: list[ChunkItem],
        encoder: TextEncoder,
        reranker: PairReranker | None = None,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
    ):
        if not chunks:
            raise ValueError("至少需要一个课程片段来构建检索索引。")

        self.chunks = chunks
        self.encoder = encoder
        self.reranker = reranker
        self.client = client or QdrantClient(location=":memory:")
        self.collection_name = collection_name or f"course_chunks_{uuid4().hex}"

        texts = [chunk.text for chunk in chunks]
        self.dense_matrix = np.asarray(encoder.encode(texts), dtype=np.float32)
        if self.dense_matrix.ndim != 2 or len(self.dense_matrix) != len(chunks):
            raise ValueError("文本编码器必须为每个课程片段返回一个向量。")

        self.sparse_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            tokenizer=_jieba_tokenizer,
            token_pattern=None,
        )
        sparse_matrix = self.sparse_vectorizer.fit_transform(texts)
        self._build_collection(sparse_matrix)

    def _build_collection(self, sparse_matrix) -> None:
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=int(self.dense_matrix.shape[1]),
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        points = []
        for index, (chunk, dense) in enumerate(zip(self.chunks, self.dense_matrix)):
            points.append(
                models.PointStruct(
                    id=index,
                    vector={
                        "dense": dense.astype(float).tolist(),
                        "sparse": _as_sparse_vector(sparse_matrix[index]),
                    },
                    payload={"chunk_index": index, "chunk_id": chunk.chunk_id},
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def _query_candidates(
        self,
        query: str,
        candidate_k: int,
        mode: str,
    ) -> list[tuple[int, float]]:
        if mode not in {"dense", "hybrid"}:
            raise ValueError("mode必须是dense或hybrid。")

        query_dense = np.asarray(self.encoder.encode([query])[0], dtype=np.float32)
        if mode == "dense":
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_dense.astype(float).tolist(),
                using="dense",
                limit=candidate_k,
                with_payload=True,
            )
        else:
            prepared_query = _prepare_query(query)
            sparse_row = self.sparse_vectorizer.transform([prepared_query])
            prefetch = [
                models.Prefetch(
                    query=query_dense.astype(float).tolist(),
                    using="dense",
                    limit=candidate_k,
                )
            ]
            if sparse_row.nnz:
                prefetch.append(
                    models.Prefetch(
                        query=_as_sparse_vector(sparse_row),
                        using="sparse",
                        limit=candidate_k,
                    )
                )
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=candidate_k,
                with_payload=True,
            )

        return [
            (int(point.payload["chunk_index"]), float(point.score))
            for point in response.points
        ]

    def search(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 20,
        mode: str = "hybrid",
        rerank: bool = False,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        candidate_k = min(max(candidate_k, top_k), len(self.chunks))
        candidates = self._query_candidates(query, candidate_k, mode)

        if rerank:
            if self.reranker is None:
                raise RuntimeError("请求了重排，但没有配置reranker。")
            scores = self.reranker.score(
                query,
                [self.chunks[index].text for index, _ in candidates],
            )
            candidates = sorted(
                (
                    (index, float(score))
                    for (index, _), score in zip(candidates, scores)
                ),
                key=lambda item: item[1],
                reverse=True,
            )

        return [
            SearchResult(
                chunk=self.chunks[index],
                similarity=round(score, 6),
                retrieval_reason="semantic",
            )
            for index, score in candidates[:top_k]
        ]

    def search_context(
        self,
        query: str,
        top_k: int = 3,
        neighbor_window: int = 2,
        max_context_chunks: int = 12,
        max_neighbor_chunks_per_page: int = 1,
        candidate_k: int = 20,
        mode: str = "hybrid",
        rerank: bool = False,
    ) -> list[SearchResult]:
        anchors = self.search(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            mode=mode,
            rerank=rerank,
        )
        return expand_neighbor_context(
            self.chunks,
            anchors,
            neighbor_window=neighbor_window,
            max_context_chunks=max_context_chunks,
            max_neighbor_chunks_per_page=max_neighbor_chunks_per_page,
        )
