"""
学习路径推荐模块 - TA 的模块
负责: 根据图谱找先修和后续知识 → 最短路径 → 输出学习路径图
"""
from fastapi import APIRouter

router = APIRouter()


@router.post("/path")
async def get_learning_path():
    """输入目标知识点 → 返回先修、当前、后续 + 路径图"""
    raise NotImplementedError("待实现")
