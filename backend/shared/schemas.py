"""
公共数据模型 (Pydantic Schemas)
你和 TA 都引用这里的结构，保证接口一致
"""
from pydantic import BaseModel, Field
from typing import Optional


# ============ 通用 ============
class HealthResponse(BaseModel):
    status: str
    version: str


# ============ 课程资料解析（你） ============
class ChunkItem(BaseModel):
    """文本分块"""
    chunk_id: str          # 如 "lec1_p3_c2"
    text: str              # 分块文本
    source: str            # 来源文件名
    page: int = 0          # 页码
    chunk_index: int = 0   # 块序号


class ParseRequest(BaseModel):
    """上传文件后的解析请求"""
    filename: str


class ParseResponse(BaseModel):
    """解析结果"""
    filename: str
    full_text: str
    chunks: list[ChunkItem]
    chunk_count: int


# ============ 知识图谱（你） ============
class TripleItem(BaseModel):
    """三元组"""
    head: str       # 实体1
    relation: str   # 关系
    tail: str       # 实体2
    evidence_id: Optional[str] = None


class KGQueryRequest(BaseModel):
    """图谱查询请求"""
    entity: str             # 要查的知识点
    depth: int = 1          # 邻居深度


class KGQueryResponse(BaseModel):
    """图谱查询结果"""
    entity: str
    neighbors: list[TripleItem]
    prerequisites: list[str]   # 先修知识
    follow_ups: list[str]      # 后续知识
    subgraph_json: dict        # pyvis 子图数据


# ============ 向量检索（你） ============
class SearchRequest(BaseModel):
    """检索请求"""
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    """单条检索结果"""
    chunk: ChunkItem
    similarity: float
    retrieval_reason: str = "semantic"  # semantic / neighbor


class SearchResponse(BaseModel):
    """检索结果"""
    query: str
    results: list[SearchResult]


# ============ Graph-RAG 问答（TA） ============
class RAGRequest(BaseModel):
    """RAG 问答请求"""
    question: str


class EvidenceItem(BaseModel):
    """A short, user-checkable course evidence card."""
    evidence_id: str
    chunk_id: str
    source: str
    page: int = 0
    excerpt: str
    retrieval_reason: str = "semantic"
    similarity: float = 0.0


class RAGResponse(BaseModel):
    """RAG 问答结果"""
    question: str
    answer: str
    sources: list[str]              # 来源片段编号
    related_triples: list[TripleItem]  # 相关图谱关系
    intent: str                     # qa / compare / learning_path / quiz
    evidence_score: float = 0.0
    evidence_level: str = "low"
    evidence_summary: str = ""
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    cited_evidence_items: list[EvidenceItem] = Field(default_factory=list)
    cited_triples: list[TripleItem] = Field(default_factory=list)


# ============ 学习路径（TA） ============
class LearningPathRequest(BaseModel):
    """学习路径请求"""
    target_entity: str


class LearningPathResponse(BaseModel):
    """学习路径结果"""
    target: str
    path: list[str]            # 路径节点列表
    edges: list[TripleItem]    # 路径边
    path_json: dict            # pyvis 路径图数据


# ============ 自动出题（TA） ============
class QuizRequest(BaseModel):
    """出题请求"""
    concept: str          # 考察的知识点
    num_questions: int = 3
    question_types: list[str] = [
        "choice",
        "multiple_choice",
        "true_false",
        "fill",
        "short_answer",
        "compare",
        "application",
    ]
    difficulty: str = "medium"  # easy / medium / hard


class QuizItem(BaseModel):
    """单道题"""
    id: int
    type: str               # choice / fill / short_answer
    question: str
    options: Optional[list[str]] = None  # 选择题选项
    answer: str
    explanation: str
    source_chunk_ids: list[str] = []     # 来源片段


class QuizResponse(BaseModel):
    """出题结果"""
    concept: str
    questions: list[QuizItem]
