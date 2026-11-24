TRIPLES=[{"head":"分布式表示","relation":"相关概念","tail":"One-hot编码"},{"head":"分布式表示","relation":"容易混淆","tail":"One-hot编码"},{"head":"词嵌入","relation":"相关概念","tail":"分布式表示"},{"head":"词嵌入","relation":"相关概念","tail":"分布式表示"}]
class KnowledgeGraph:
    def __init__(self,triples=None):
        self.triples=[]; seen=set()
        for item in triples or TRIPLES:
            key=(item["head"],item["relation"],item["tail"])
            if key not in seen: seen.add(key); self.triples.append(dict(item))
    def get_neighbors(self,entity):
        return [dict(t) for t in self.triples if entity in (t["head"],t["tail"])]
