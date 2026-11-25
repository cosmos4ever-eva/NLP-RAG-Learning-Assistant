"""
Learning path recommendation module.
"""
from fastapi import APIRouter
from pydantic import BaseModel

from knowledge_graph.graph import get_kg
from learning_analytics import record_concepts
from shared.schemas import TripleItem

router = APIRouter()


class LearningPathRequest(BaseModel):
    target_entity: str


class LearningPathResponse(BaseModel):
    target: str
    path: list[str]
    edges: list[TripleItem]
    follow_ups: list[str]


def _build_prerequisite_path(entity: str, visited: set[str] | None = None) -> list[str]:
    if visited is None:
        visited = set()

    if entity in visited:
        return []

    visited.add(entity)
    kg = get_kg()
    path = []

    for prereq in kg.get_prerequisites(entity):
        path.extend(_build_prerequisite_path(prereq, visited))
        if prereq not in path:
            path.append(prereq)

    if entity not in path:
        path.append(entity)

    return path


def _path_edges(path: list[str]) -> list[TripleItem]:
    kg = get_kg()
    edges = []
    path_nodes = set(path)

    for concept in path:
        for prerequisite in kg.get_prerequisites(concept):
            if prerequisite in path_nodes:
                edges.append(TripleItem(
                    head=concept,
                    relation="先修知识",
                    tail=prerequisite,
                ))

    return edges


def _path_prerequisites(path: list[str], target: str) -> list[str]:
    """Return only prerequisite concepts already shown in the learning path."""
    return [
        concept for concept in path
        if concept and concept != target
    ]




@router.post("/path", response_model=LearningPathResponse)
async def get_learning_path(req: LearningPathRequest):
    """Return prerequisite-based learning path for a target concept."""
    kg = get_kg()
    record_concepts([req.target_entity])
    path = _build_prerequisite_path(req.target_entity)

    return LearningPathResponse(
        target=req.target_entity,
        path=path,
        edges=_path_edges(path),
        follow_ups=_path_prerequisites(path, req.target_entity),
    )
