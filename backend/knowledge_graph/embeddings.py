"""Deterministic FastRP-style structural embeddings for small course graphs.

This offline adapter is intentionally separate from graph facts: similarity can
add retrieval candidates, but it never creates or asserts a new relationship.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


DEFAULT_RELATION_WEIGHTS = {
    "先修知识": 1.0,
    "包含": 0.9,
    "容易混淆": 0.8,
    "相关概念": 0.6,
    "应用场景": 0.5,
    "属于": 0.9,
}


class FastRPStyleEmbedding:
    """Encode weighted one- and two-hop graph structure into dense vectors."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        dimensions: int = 32,
        random_seed: int = 42,
        relation_weights: dict[str, float] | None = None,
        iteration_weights: tuple[float, float] = (1.0, 0.5),
    ):
        if dimensions <= 0:
            raise ValueError("dimensions必须为正数。")
        self.nodes = sorted(str(node) for node in graph.nodes)
        self.node_to_index = {node: index for index, node in enumerate(self.nodes)}
        self.relation_weights = relation_weights or DEFAULT_RELATION_WEIGHTS
        self.vectors = self._fit(graph, dimensions, random_seed, iteration_weights)

    def _fit(
        self,
        graph: nx.MultiDiGraph,
        dimensions: int,
        random_seed: int,
        iteration_weights: tuple[float, float],
    ) -> np.ndarray:
        node_count = len(self.nodes)
        if node_count == 0:
            return np.empty((0, dimensions), dtype=np.float32)

        adjacency = np.zeros((node_count, node_count), dtype=np.float32)
        for head, tail, data in graph.edges(data=True):
            head_index = self.node_to_index[str(head)]
            tail_index = self.node_to_index[str(tail)]
            weight = float(self.relation_weights.get(data.get("relation", ""), 0.5))
            # Structural similarity is treated as undirected; factual direction
            # remains available only through explicit graph traversal.
            adjacency[head_index, tail_index] += weight
            adjacency[tail_index, head_index] += weight

        row_sums = adjacency.sum(axis=1, keepdims=True)
        transition = np.divide(
            adjacency,
            row_sums,
            out=np.zeros_like(adjacency),
            where=row_sums != 0,
        )
        rng = np.random.default_rng(random_seed)
        initial = rng.normal(0.0, 1.0, size=(node_count, dimensions)).astype(np.float32)
        one_hop = transition @ initial
        two_hop = transition @ one_hop
        vectors = iteration_weights[0] * one_hop + iteration_weights[1] * two_hop
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(
            vectors,
            norms,
            out=np.zeros_like(vectors),
            where=norms != 0,
        )

    def similar_nodes(self, seeds: list[str], top_k: int = 3) -> list[tuple[str, float]]:
        seed_indices = [self.node_to_index[seed] for seed in seeds if seed in self.node_to_index]
        if not seed_indices or top_k <= 0:
            return []

        query = self.vectors[seed_indices].mean(axis=0)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query /= norm
        scores = self.vectors @ query
        excluded = set(seed_indices)
        ranking = [
            (self.nodes[index], float(scores[index]))
            for index in np.argsort(scores)[::-1]
            if index not in excluded and scores[index] > 0
        ]
        return [(node, round(score, 6)) for node, score in ranking[:top_k]]
