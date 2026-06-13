"""
向量检索模块 - 你的模块
负责: TF-IDF 或 sentence-transformers + FAISS → top-k 课程片段检索
"""
from fastapi import APIRouter

router = APIRouter()


@router.post("/search")
async def search():
    """向量检索 top-k 课程片段 → 返回 chunk 内容、相似度、来源"""
    raise NotImplementedError("待实现")


@router.post("/build_index")
async def build_index():
    """构建/重建向量索引"""
    raise NotImplementedError("待实现")
