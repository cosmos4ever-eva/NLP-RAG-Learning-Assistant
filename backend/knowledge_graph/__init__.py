"""
知识图谱构建模块
负责: 整理 NLP 知识点 → 构建三元组 → networkx 读入 → 邻居/先修/后续查询
"""
from fastapi import APIRouter, HTTPException, Query

from shared.schemas import TripleItem, KGQueryResponse
from .graph import get_kg

router = APIRouter()


@router.get("/entities")
async def list_entities(type: str = Query(None, description="按类型筛选: 概念/方法/模型/算法/应用任务/参数指标")):
    """列出所有知识点实体，可按类型筛选"""
    kg = get_kg()
    entities = kg.get_all_entities()
    if type:
        entities = [e for e in entities if e["type"] == type]
    return {"count": len(entities), "entities": entities}


@router.get("/entity/{name}")
async def query_entity(name: str, depth: int = Query(1, ge=1, le=3)):
    """查询某个知识点的邻居、先修、后续关系 + pyvis 子图"""
    kg = get_kg()
    entity = kg.get_entity(name)
    if not entity:
        raise HTTPException(status_code=404, detail=f"知识点 '{name}' 不在图谱中")

    neighbors_raw = kg.get_neighbors(name, depth=depth)
    neighbors = [
        TripleItem(head=n["head"], relation=n["relation"], tail=n["tail"])
        for n in neighbors_raw
    ]

    subgraph = kg.get_subgraph(name, depth=depth)

    return KGQueryResponse(
        entity=name,
        neighbors=neighbors,
        prerequisites=kg.get_prerequisites(name),
        follow_ups=kg.get_follow_ups(name),
        subgraph_json=subgraph,
    )


@router.get("/triples")
async def list_triples(relation: str = Query(None, description="按关系筛选: 先修知识/包含/相关概念/应用场景")):
    """列出所有三元组，可按关系类型筛选"""
    kg = get_kg()
    triples = kg.get_all_triples()
    if relation:
        triples = [t for t in triples if t["relation"] == relation]
    return {"count": len(triples), "triples": triples}


@router.get("/path")
async def find_path(source: str = Query(...), target: str = Query(...)):
    """查询两个知识点之间的最短学习路径"""
    kg = get_kg()
    path = kg.get_shortest_path(source, target)
    if not path:
        raise HTTPException(status_code=404, detail=f"'{source}' 到 '{target}' 之间没有路径")

    # 收集路径上的边
    edges = []
    for i in range(len(path) - 1):
        # 查是否有直接边
        if kg.graph.has_edge(path[i], path[i + 1]):
            rel = kg.graph.edges[path[i], path[i + 1]]["relation"]
            edges.append(TripleItem(head=path[i], relation=rel, tail=path[i + 1]))
        elif kg.graph.has_edge(path[i + 1], path[i]):
            rel = kg.graph.edges[path[i + 1], path[i]]["relation"]
            edges.append(TripleItem(head=path[i + 1], relation=rel, tail=path[i]))

    return {
        "source": source,
        "target": target,
        "path": path,
        "length": len(path) - 1,
        "edges": [e.model_dump() for e in edges],
    }
