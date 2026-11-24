"""
FastAPI 主入口
启动: uvicorn main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.schemas import HealthResponse

app = FastAPI(
    title="NLP RAG Teaching Assistant",
    description="NLP 课程学习智能助手 - Graph-RAG 后端",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.1.0")


# 各模块路由注册
from parsing import router as parsing_router
from knowledge_graph import router as kg_router
from vector_search import router as vs_router
from graph_rag import router as rag_router
from quiz_gen import router as quiz_router
from learning_path import router as learning_router
from concept_compare import router as compare_router
from learning_analytics import router as analytics_router
from learning_agent.routes import router as agent_router
app.include_router(parsing_router, prefix="/api/parsing", tags=["课程资料解析"])
app.include_router(kg_router, prefix="/api/kg", tags=["知识图谱"])
app.include_router(vs_router, prefix="/api/search", tags=["向量检索"])
app.include_router(rag_router, prefix="/api/rag", tags=["Graph-RAG 问答"])
app.include_router(quiz_router, prefix="/api/quiz", tags=["自动出题"])
app.include_router(learning_router, prefix="/api/learning", tags=["学习路径"])
app.include_router(compare_router, prefix="/api/compare", tags=["概念对比"])
app.include_router(analytics_router, prefix="/api/analytics", tags=["学习分析"])
app.include_router(agent_router, prefix="/api/agent", tags=["学习Agent"])

# 后续模块（逐步打开）
# from vector_search import router as vs_router
# from graph_rag import router as rag_router
# from learning_path import router as lp_router
# from quiz_gen import router as quiz_router
# app.include_router(vs_router, prefix="/api/search", tags=["向量检索"])
# app.include_router(rag_router, prefix="/api/rag", tags=["Graph-RAG 问答"])
# app.include_router(lp_router, prefix="/api/learning", tags=["学习路径"])
# app.include_router(quiz_router, prefix="/api/quiz", tags=["自动出题"])
