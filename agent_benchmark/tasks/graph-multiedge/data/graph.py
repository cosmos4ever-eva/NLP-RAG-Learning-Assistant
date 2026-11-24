TRIPLES = [
    {"head": "分布式表示", "relation": "相关概念", "tail": "One-hot编码"},
    {"head": "分布式表示", "relation": "容易混淆", "tail": "One-hot编码"},
    {"head": "词嵌入", "relation": "相关概念", "tail": "分布式表示"},
    {"head": "词嵌入", "relation": "相关概念", "tail": "分布式表示"},
]


class KnowledgeGraph:
    def __init__(self, triples=None):
        self.triples = list(triples or TRIPLES)
        self.edges = {}
        for triple in self.triples:
            self.edges[(triple["head"], triple["tail"])] = triple["relation"]

    def get_neighbors(self, entity):
        found = []
        for (head, tail), relation in self.edges.items():
            if entity in (head, tail):
                found.append({"head": head, "relation": relation, "tail": tail})
        return found
