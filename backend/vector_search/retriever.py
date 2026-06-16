"""
向量检索引擎
负责: TF-IDF 向量化 → 余弦相似度检索 → top-k 课程片段
纯离线方案，不依赖外部模型下载。后续可切换到 sentence-BERT + FAISS。
"""
import json
import pickle
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


def _jieba_tokenizer(text: str) -> list[str]:
    """jieba 中文分词器（模块级函数，保证可序列化）"""
    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]


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

        # query → TF-IDF 向量
        query_vec = self.vectorizer.transform([query])

        # 余弦相似度
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

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
