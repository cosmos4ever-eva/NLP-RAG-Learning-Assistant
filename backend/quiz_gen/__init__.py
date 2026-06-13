"""
自动出题模块 - TA 的模块
负责: 基于检索片段+图谱关系 → 生成练习题 → 输出结构化 JSON
"""
from fastapi import APIRouter

router = APIRouter()


@router.post("/generate")
async def generate_quiz():
    """选择知识点、题型、数量 → 生成题目 → 返回 JSON"""
    raise NotImplementedError("待实现")
