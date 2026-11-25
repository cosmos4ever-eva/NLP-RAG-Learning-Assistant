"""
Graph-RAG API routes.
"""
from fastapi import APIRouter

from learning_analytics import record_question_entities
from shared.schemas import EvidenceItem, RAGRequest, RAGResponse, TripleItem

from .pipeline import answer_question

router = APIRouter()


@router.post("/ask", response_model=RAGResponse)
async def ask(req: RAGRequest):
    """Answer a question with retrieved course chunks and KG triples."""
    record_question_entities(req.question)
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
        evidence_score=result["evidence_score"],
        evidence_level=result["evidence_level"],
        evidence_summary=result["evidence_summary"],
        evidence_items=[EvidenceItem(**item) for item in result["evidence_items"]],
        cited_evidence_items=[
            EvidenceItem(**item) for item in result["cited_evidence_items"]
        ],
        cited_triples=[
            TripleItem(
                head=triple["head"],
                relation=triple["relation"],
                tail=triple["tail"],
                evidence_id=triple.get("evidence_id"),
            )
            for triple in result["cited_triples"]
        ],
    )
