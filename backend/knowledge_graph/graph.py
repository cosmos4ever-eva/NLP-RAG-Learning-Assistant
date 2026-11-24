"""
NLP 课程知识图谱
负责: 加载三元组 → 构建 networkx 有向图 → 提供查询接口 → 导出 pyvis 可视化
"""
import json
from pathlib import Path
from typing import Optional

import networkx as nx
from pyvis.network import Network


TRIPLES_PATH = Path(__file__).resolve().parent / "triples.json"

# 关系类型说明
RELATION_TYPES = {
    "先修知识": "prerequisite",
    "包含": "contains",
    "相关概念": "related_to",
    "应用场景": "applied_to",
    "属于": "belongs_to",
    "容易混淆": "easy_to_confuse",
}


class KnowledgeGraph:
    """NLP 课程知识图谱"""

    def __init__(self, triples_path: Path = TRIPLES_PATH):
        # Preserve multiple semantic relations between the same ordered pair.
        self.graph = nx.MultiDiGraph()
        self.entities: dict[str, dict] = {}  # id -> entity info
        self.triples: list[dict] = []
        self._load(triples_path)

    def _load(self, path: Path):
        """从 JSON 加载实体和三元组"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 加载实体
        for ent in data.get("entities", []):
            self.entities[ent["id"]] = ent
            self.graph.add_node(
                ent["name"],
                id=ent["id"],
                type=ent["type"],
                description=ent.get("description", ""),
            )

        # 加载三元组
        self.triples = data.get("triples", [])
        for t in self.triples:
            self.graph.add_edge(
                t["head"],
                t["tail"],
                relation=t["relation"],
            )

    def get_all_entities(self) -> list[dict]:
        """返回所有实体"""
        return [
            {
                "id": eid,
                "name": info["name"],
                "type": info["type"],
                "description": info.get("description", ""),
            }
            for eid, info in self.entities.items()
        ]

    def get_entity(self, name: str) -> Optional[dict]:
        """查询单个实体"""
        if name not in self.graph:
            return None
        node = self.graph.nodes[name]
        return {
            "id": node["id"],
            "name": name,
            "type": node["type"],
            "description": node.get("description", ""),
        }

    def get_all_triples(self) -> list[dict]:
        """返回所有三元组"""
        return self.triples

    def get_neighbors(self, entity: str, depth: int = 1) -> list[dict]:
        """获取实体的邻居（支持多跳）"""
        if entity not in self.graph:
            return []

        neighbors = []
        seen_triples = set()
        # 使用 BFS 获取指定深度的邻居
        seen = {entity}
        frontier = {entity}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                # 出边
                for _, tail, data in self.graph.out_edges(node, data=True):
                    key = (node, data["relation"], tail)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        neighbors.append({
                            "head": node,
                            "relation": data["relation"],
                            "tail": tail,
                        })
                    if tail not in seen:
                        seen.add(tail)
                        next_frontier.add(tail)
                # 入边
                for head, _, data in self.graph.in_edges(node, data=True):
                    key = (head, data["relation"], node)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        neighbors.append({
                            "head": head,
                            "relation": data["relation"],
                            "tail": node,
                        })
                    if head not in seen:
                        seen.add(head)
                        next_frontier.add(head)
            frontier = next_frontier

        return neighbors

    def get_prerequisites(self, entity: str) -> list[str]:
        """获取先修知识点（该实体依赖的知识）"""
        if entity not in self.graph:
            return []

        prereqs = []
        for _, tail, data in self.graph.out_edges(entity, data=True):
            if data["relation"] == "先修知识":
                prereqs.append(tail)
        return prereqs

    def get_follow_ups(self, entity: str) -> list[str]:
        """获取后续知识（哪些知识依赖该实体）"""
        if entity not in self.graph:
            return []

        follow_ups = []
        for head, _, data in self.graph.in_edges(entity, data=True):
            if data["relation"] == "先修知识":
                follow_ups.append(head)
        return follow_ups

    def get_shortest_path(self, source: str, target: str) -> Optional[list[str]]:
        """获取两个实体之间的最短学习路径"""
        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_subgraph(self, entity: str, depth: int = 1) -> dict:
        """
        获取某个实体的子图，导出为 pyvis 可用的 JSON 格式
        用于前端可视化展示
        """
        neighbors = self.get_neighbors(entity, depth=depth)
        if not neighbors:
            return {"nodes": [], "edges": []}

        # 收集子图中所有节点
        node_names = {entity}
        for n in neighbors:
            node_names.add(n["head"])
            node_names.add(n["tail"])

        # 构建 pyvis 网络
        net = Network(height="400px", width="100%", directed=True, notebook=False)

        # 实体类型对应颜色
        type_colors = {
            "概念": "#FFD700",
            "方法": "#87CEEB",
            "模型": "#FF6B6B",
            "算法": "#90EE90",
            "应用任务": "#DDA0DD",
            "参数/指标": "#F0E68C",
        }

        for name in node_names:
            info = self.entities.get(
                self.graph.nodes[name].get("id", ""),
                self.graph.nodes[name],
            )
            node_type = info.get("type", "概念") if isinstance(info, dict) else "概念"
            color = type_colors.get(node_type, "#CCCCCC")
            title = f"{name}\n类型: {node_type}"
            if isinstance(info, dict) and info.get("description"):
                title += f"\n{info['description']}"
            net.add_node(name, label=name, color=color, title=title)

        # 边
        seen_edges = set()
        for n in neighbors:
            edge_key = (n["head"], n["tail"], n["relation"])
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                net.add_edge(n["head"], n["tail"], title=n["relation"], label=n["relation"])

        # pyvis 导出为 JSON 节点/边结构
        nodes_json = [{"id": n["id"], "label": n["label"], "color": n.get("color", ""),
                        "title": n.get("title", "")} for n in net.nodes]
        edges_json = [{"from": e["from"], "to": e["to"], "label": e.get("label", ""),
                        "title": e.get("title", "")} for e in net.edges]

        return {"nodes": nodes_json, "edges": edges_json}


# 全局单例
_kg_instance: Optional[KnowledgeGraph] = None


def get_kg() -> KnowledgeGraph:
    """获取知识图谱全局实例（延迟加载）"""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraph()
    return _kg_instance
