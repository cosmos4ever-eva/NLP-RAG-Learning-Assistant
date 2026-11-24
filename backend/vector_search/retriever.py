"""
向量检索引擎
负责: TF-IDF 向量化 → 余弦相似度检索 → top-k 课程片段
纯离线方案，不依赖外部模型下载。后续可切换到 sentence-BERT + FAISS。
"""
import json
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shared.schemas import ChunkItem, SearchResult

# 索引存储目录
INDEX_DIR = Path(__file__).resolve().parent / "index"
INDEX_DIR.mkdir(exist_ok=True)

CHUNKS_PATH = INDEX_DIR / "chunks.json"
VECTORIZER_PATH = INDEX_DIR / "vectorizer.pkl"
TFIDF_MATRIX_PATH = INDEX_DIR / "tfidf_matrix.npy"
LEXICAL_COVERAGE_WEIGHT = 0.35

# 这些词对“用户在问什么”有帮助，但对“答案在哪一页”没有区分度。
# 只在查询侧过滤，课程原文和索引保持不变。
QUERY_STOPWORDS = {
    "什么", "其中", "请问", "请", "为什么", "如何", "怎么", "怎样",
    "哪些", "分别", "各自", "是否", "有什么", "课程", "课件", "资料",
    "需要", "给出", "一种", "典型", "系统", "还要", "信息",
}

QUERY_EXPANSIONS = {
    "不足": ("不够",),
    "保存": ("记录", "记住", "来源", "回溯"),
}


def _jieba_tokenizer(text: str) -> list[str]:
    """jieba 中文分词器（模块级函数，保证可序列化）"""
    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]


def _prepare_query(text: str) -> str:
    """删除低信息问句词，保留真正决定证据位置的概念词。"""
    tokens = _jieba_tokenizer(text)
    informative = [token for token in tokens if token.casefold() not in QUERY_STOPWORDS]
    # 当问题已有更具体的处理步骤时，去掉过宽的父概念，避免总命中目录页。
    if any(term in text for term in ("归一化", "词干", "词形", "分词", "停用词")):
        informative = [
            token for token in informative
            if token not in {"文本预处理", "文本", "预处理"}
        ]
    for phrase, expansions in QUERY_EXPANSIONS.items():
        if phrase in text:
            informative.extend(expansions)
    return " ".join(informative) or text


def _query_clauses(text: str) -> list[str]:
    """把多问句拆开，避免第一小问的高频词淹没后续证据。"""
    clauses = [part.strip() for part in re.split(r"[？?；;。]+", text)]
    return [part for part in clauses if len(_prepare_query(part)) >= 2]


class VectorRetriever:
    """向量检索器：TF-IDF + 余弦相似度"""

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None  # shape: (n_chunks, vocab_size)
        self.chunks: list[ChunkItem] = []
        self._built = False

    def build_index(self, chunks: list[ChunkItem]) -> dict:
        """
        构建 TF-IDF 索引
        步骤: 文本 → TF-IDF 向量化 → 存 numpy 矩阵
        """
        if not chunks:
            return {"error": "没有可用的 chunk，请先解析课件"}

        texts = [c.text for c in chunks]
        self.chunks = chunks

        # TF-IDF 向量化（用 jieba 做中文分词）
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            tokenizer=_jieba_tokenizer,
            token_pattern=None,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        # 存磁盘
        self._save(chunks)

        self._built = True
        return {
            "chunk_count": len(chunks),
            "vocab_size": len(self.vectorizer.vocabulary_),
            "method": "TF-IDF + Cosine Similarity",
        }

    def load_index(self) -> bool:
        """从磁盘加载已有索引（避免重复构建）"""
        if not VECTORIZER_PATH.exists() or not CHUNKS_PATH.exists():
            return False

        try:
            with open(VECTORIZER_PATH, "rb") as f:
                self.vectorizer = pickle.load(f)
            # scipy sparse matrix
            from scipy.sparse import load_npz
            matrix_path = INDEX_DIR / "tfidf_matrix.npz"
            if matrix_path.exists():
                self.tfidf_matrix = load_npz(str(matrix_path))
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.chunks = [ChunkItem(**c) for c in raw]
            self._built = True
            return True
        except Exception:
            return False

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        检索与 query 最相关的 top_k 个 chunk
        步骤: query → TF-IDF 向量 → 余弦相似度 → top-k
        """
        if not self._built or self.vectorizer is None:
            raise RuntimeError("索引未构建，请先调用 POST /api/search/build_index")

        prepared_query = _prepare_query(query)
        # query → TF-IDF 向量
        query_vec = self.vectorizer.transform([prepared_query])

        # 余弦相似度
        cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        query_terms = set(_jieba_tokenizer(prepared_query))
        if query_terms:
            coverage_scores = np.array([
                sum(term.casefold() in chunk.text.casefold() for term in query_terms)
                / len(query_terms)
                for chunk in self.chunks
            ])
            sims = (
                (1 - LEXICAL_COVERAGE_WEIGHT) * cosine_scores
                + LEXICAL_COVERAGE_WEIGHT * coverage_scores
            )
        else:
            sims = cosine_scores

        # top-k 索引
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if sims[idx] > 0:
                chunk = self.chunks[idx]
                results.append(SearchResult(
                    chunk=chunk,
                    similarity=round(float(sims[idx]), 4),
                ))

        return results

    def search_context(
        self,
        query: str,
        top_k: int = 3,
        neighbor_window: int = 2,
        max_context_chunks: int = 12,
        max_neighbor_chunks_per_page: int = 1,
    ) -> list[SearchResult]:
        """Retrieve semantic anchors, then add slide context around them.

        Comparison questions reserve one anchor for another source when the
        ordinary top-k is monopolized by a single document. Nearby slides are
        labeled separately so callers do not confuse context expansion with a
        semantic match.
        """
        if not self._built or self.vectorizer is None:
            raise RuntimeError("索引未构建，请先调用 POST /api/search/build_index")
        if top_k <= 0:
            return []

        candidate_count = max(top_k * 4, 12)
        candidates = self.search(query, top_k=candidate_count)
        clauses = _query_clauses(query)
        anchors = []
        if len(clauses) > 1:
            # 每个小问至少保留一个语义锚点，再用整句结果补足。
            for clause in clauses:
                clause_results = self.search(clause, top_k=3)
                alternative = next(
                    (
                        result for result in clause_results
                        if result.chunk.chunk_id not in {item.chunk.chunk_id for item in anchors}
                    ),
                    None,
                )
                if alternative is not None:
                    anchors.append(alternative)
                if len(anchors) >= top_k:
                    break
        for candidate in candidates:
            if len(anchors) >= top_k:
                break
            if candidate.chunk.chunk_id not in {item.chunk.chunk_id for item in anchors}:
                anchors.append(candidate)
        comparison_query = bool(re.search(r"(区别|不同|比较|对比|关系)", query))
        if comparison_query and top_k >= 2 and anchors:
            anchor_sources = {result.chunk.source for result in anchors}
            if len(anchor_sources) == 1:
                alternative = next(
                    (result for result in candidates if result.chunk.source not in anchor_sources),
                    None,
                )
                if alternative is not None:
                    anchors[-1] = alternative

        results = []
        seen = set()
        for anchor in anchors:
            seen.add(anchor.chunk.chunk_id)
            results.append(anchor.model_copy(update={"retrieval_reason": "semantic"}))

        neighbors = []
        for anchor_rank, anchor in enumerate(anchors):
            for chunk in self.chunks:
                distance = abs(chunk.page - anchor.chunk.page)
                if (
                    chunk.source == anchor.chunk.source
                    and 0 < distance <= neighbor_window
                    and chunk.chunk_id not in seen
                ):
                    neighbors.append((distance, anchor_rank, chunk.page, chunk))

        page_counts = {}
        for _, anchor_rank, _, chunk in sorted(neighbors, key=lambda item: item[:3]):
            page_key = (chunk.source, chunk.page)
            if chunk.chunk_id in seen or len(results) >= max_context_chunks:
                continue
            if page_counts.get(page_key, 0) >= max_neighbor_chunks_per_page:
                continue
            seen.add(chunk.chunk_id)
            page_counts[page_key] = page_counts.get(page_key, 0) + 1
            results.append(SearchResult(
                chunk=chunk,
                similarity=0.0,
                retrieval_reason="neighbor",
            ))

        return results

    def _save(self, chunks: list[ChunkItem]):
        """持久化索引"""
        # chunk 元数据
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in chunks], f, ensure_ascii=False, indent=2)

        # TF-IDF vectorizer
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)

        # TF-IDF 矩阵 (sparse → 压缩存储)
        from scipy.sparse import save_npz
        save_npz(str(INDEX_DIR / "tfidf_matrix.npz"), self.tfidf_matrix)

    def get_status(self) -> dict:
        """索引状态"""
        return {
            "built": self._built,
            "method": "TF-IDF + Cosine Similarity",
            "chunk_count": len(self.chunks),
            "vocab_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
        }


# 全局单例
_retriever_instance: Optional[VectorRetriever] = None


def get_retriever() -> VectorRetriever:
    """获取检索器全局实例"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = VectorRetriever()
        loaded = _retriever_instance.load_index()
        if loaded:
            print(f"[Retriever] 已加载本地索引（{len(_retriever_instance.chunks)} chunks）")
    return _retriever_instance
