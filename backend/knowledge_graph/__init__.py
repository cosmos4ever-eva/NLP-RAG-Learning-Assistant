"""
知识图谱构建模块 - 你的模块
负责: 整理 NLP 知识点 → 构建三元组 → networkx 读入 → 邻居/先修/后续查询
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/entities")
async def list_entities():
    """列出所有知识点实体"""
    raise NotImplementedError("待实现")


@router.get("/entity/{name}")
async def query_entity(name: str):
    """查询某个知识点的邻居、先修、后续关系"""
    raise NotImplementedError("待实现")


@router.get("/triples")
async def list_triples():
    """列出所有三元组"""
    raise NotImplementedError("待实现")
