"""
Graph-RAG API routes.
"""
from fastapi import APIRouter

from shared.schemas import RAGRequest, RAGResponse, TripleItem

from .pipeline import answer_question

router = APIRouter()


@router.post("/ask", response_model=RAGResponse)
async def ask(req: RAGRequest):
    """Answer a question with retrieved course chunks and KG triples."""
    result = answer_question(req.question, top_k=3, depth=1)

    return RAGResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        related_triples=[
            TripleItem(
                head=triple["head"],
                relation=triple["relation"],
                tail=triple["tail"],
            )
            for triple in result["related_triples"]
        ],
        intent="qa",
    )
