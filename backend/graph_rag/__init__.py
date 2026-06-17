"""
Graph-RAG 问答模块 - TA 的模块
负责: 识别用户问题中的知识点 → 同时查图谱+向量库 → 拼 prompt → 调 LLM → 返回答案+来源+关系
"""
from fastapi import APIRouter

router = APIRouter()


@router.post("/ask")
async def ask():
    """RAG 问答：输入问题，返回答案、来源片段、相关知识子图"""
    raise NotImplementedError("待实现")
