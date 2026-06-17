"""
向量检索模块
负责: sentence-transformers + FAISS → top-k 课程片段检索
"""
from fastapi import APIRouter, HTTPException

from shared.schemas import SearchRequest, SearchResponse
from .retriever import get_retriever

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    向量检索 top-k 课程片段
    输入: query + top_k
    返回: 按相似度排序的 chunk 列表 + 相似度分数
    """
    retriever = get_retriever()
    try:
        results = retriever.search(req.query, req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return SearchResponse(
        query=req.query,
        results=results,
    )


@router.post("/build_index")
async def build_index():
    """
    构建/重建向量索引
    从 parsing 模块获取所有已解析的 chunk → BERT 编码 → FAISS 存索引
    """
    # 从 parsing 模块获取 chunk
    try:
        from parsing import _all_chunks
    except ImportError:
        raise HTTPException(status_code=500, detail="无法访问解析模块")

    if not _all_chunks:
        raise HTTPException(status_code=400, detail="请先解析课件（POST /api/parsing/parse-all），再构建索引")

    chunks = list(_all_chunks.values())
    retriever = get_retriever()
    info = retriever.build_index(chunks)
    return {"status": "ok", **info}


@router.get("/index")
async def index_status():
    """查看向量索引状态"""
    retriever = get_retriever()
    return retriever.get_status()
