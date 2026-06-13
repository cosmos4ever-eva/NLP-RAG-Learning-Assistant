"""
课程资料解析模块 - 你的模块
负责: PDF/PPT 文本提取 → 清洗 → chunk 切分 → 来源标注
"""
from fastapi import APIRouter

router = APIRouter()


@router.post("/parse")
async def parse_document():
    """上传并解析课程资料"""
    # TODO: 接收上传文件 → 提取文本 → 清洗 → 切 chunk
    raise NotImplementedError("待实现")


@router.get("/chunks")
async def list_chunks():
    """列出已解析的所有 chunk"""
    raise NotImplementedError("待实现")
