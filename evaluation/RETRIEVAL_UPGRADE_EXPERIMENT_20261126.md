# 检索与图 Embedding 升级实验（2026-11-26）

## 目标与约束

本实验检验“更主流的向量检索、重排和图 Embedding 是否优于现有 TF-IDF + NetworkX Graph-RAG”。

- 数据：516 个本地课程片段，56 题评测集（dev/test 各 28 题）。
- 调参：只使用 dev；冻结候选后仅运行一次 sealed test。
- 主指标：Evidence Hit@3、MRR@3、扩展证据页召回、必需关系召回、CPU 查询延迟。
- 证据边界：图 Embedding 只扩展候选实体，不把结构相似性当作事实关系。
- 原始课件文本、逐题上下文和私有运行结果保存在被 Git 忽略的 `evaluation/private_runs/`。

## 实验配置

| 编号 | 配置 |
|---|---|
| A | 当前 TF-IDF Top-3 + 相邻页扩展 |
| B | BGE-M3 稠密向量 Top-3 |
| C | Qdrant 融合 BGE-M3 稠密向量与 TF-IDF 稀疏向量 |
| D | C 的 Top-12 候选经 `bge-reranker-v2-m3` 重排至 Top-3 |
| E | D + 一跳显式知识图谱关系 |
| F | E + 离线 FastRP 风格结构向量候选扩展 |
| G | TF-IDF Top-12 + BGE Reranker Top-3 + 一跳显式图（dev 裁决候选） |
| H | 仅当TF-IDF前两名分数打平时执行G的重排（后续按需重排实验） |

统一参数：`top_k=3`、`candidate_k=12`、`neighbor_window=2`、CPU 推理。

## Dev 消融结果

dev 中有课程证据的题目为 24 道、有必需图关系的题目为 18 道。

| 配置 | Hit@3 | 扩展命中 | 证据页平均召回 | MRR@3 | 关系召回 | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|
| A TF-IDF | 91.7% | 100.0% | 86.4% | 0.813 | — | 约 23 ms |
| B BGE dense | 87.5% | 91.7% | 84.4% | 0.785 | — | 约 386 ms |
| C BGE hybrid | 83.3% | 91.7% | 80.4% | 0.750 | — | 约 489 ms |
| D hybrid + rerank | 95.8% | 95.8% | 85.4% | 0.910 | — | 约 7.06 s |
| E D + explicit graph | 95.8% | 95.8% | 85.4% | 0.910 | 97.2% | 约 7.67 s |
| F E + graph embedding | 95.8% | 95.8% | 85.4% | 0.910 | 97.2% | 约 7.35 s |
| G TF-IDF + rerank + graph | **100.0%** | **100.0%** | **87.6%** | **0.931** | 97.2% | 约 3.65 s |
| H 按需rerank + graph | **100.0%** | **100.0%** | 86.7% | 0.854 | **100.0%** | 约 0.25 s |

不同运行的绝对延迟会受模型热身和系统负载影响；表中延迟用于判断数量级，不作为跨机器基准。

## Dev 裁决

1. **拒绝“纯 BGE 替换 TF-IDF”**：B、C 的证据命中均下降，课程术语和页码证据仍依赖精确词匹配。
2. **保留 Reranker，但不采用混合候选池**：Reranker 能明显改善排序；以 TF-IDF Top-12 为候选比 Qdrant 混合候选更准、更快。
3. **保留一跳显式图**：一跳关系可控且可作为事实证据。
4. **不将图 Embedding 合入主流程**：F 未提高关系召回，却把平均检索关系从 5.07 条增至 9.29 条，增加噪声。
5. **不为单个 dev Bad Case 统一改成两跳**：两跳可把 dev 关系召回补到 100%，但18道关系题的平均关系量由 5.06 条升至 14.78 条，过拟合与上下文污染风险更高。

## Sealed test（冻结后唯一一次运行）

冻结配置：G（TF-IDF Top-12 → BGE Reranker Top-3 → 相邻页 → 一跳显式图）。

| 指标 | 结果 |
|---|---:|
| Evidence Hit@3 | **23/24（95.8%）** |
| 扩展上下文命中 | **23/24（95.8%）** |
| 金标准证据页平均召回 | 63.6% |
| MRR@3 | 0.882 |
| 必需关系平均召回 | 93.2%（22 道关系题） |
| CPU平均查询延迟 | 3.44 s |

相较原冻结检索结果的22/24证据命中，新候选多命中1题；关系召回仍约为93.2%，没有证明图Embedding的增量价值。

## Bad Case

- `EVAL-031`：HMM“三个基本问题”的金标准页未进入Top-3或相邻页，属于文本排序失败。
- `EVAL-023`：处理链问题只召回一半必需关系，一跳图不足以覆盖完整链条。
- `EVAL-025`：课件证据命中，但实体映射未触发必需图关系，属于实体链接与图查询脱节。

sealed Bad Case仅记录，不用于继续修改本冻结版本。后续若创建新版本，应新增dev题或新split后再开发。

## 后续按需重排实验

为降低G的CPU延迟，后续版本仅在TF-IDF Top-1与Top-2分数完全打平时触发Reranker。该规则只依据dev观察到的检索分数设计，不使用题型、难度或sealed标签。

| 指标 | Dev | 后续sealed整体比较 |
|---|---:|---:|
| Evidence Hit@3 | 24/24（100.0%） | 22/24（91.7%） |
| 扩展上下文命中 | 24/24（100.0%） | 23/24（95.8%） |
| MRR@3 | 0.854 | 0.715 |
| 必需关系平均召回 | 100.0% | 93.2% |
| Reranker触发率 | 2/28（7.1%） | 5/28（17.9%） |
| CPU平均查询延迟 | 0.25 s | 0.52 s |

**裁决：拒绝H。** 它在dev上看似同时改善准确率与延迟，但没有在sealed数据上保持Top-3命中，属于路由规则对dev检索分布的过拟合。最终保留两个明确档位：A作为低延迟基线，G作为证据命中优先的实验候选；在获得新的独立路由评测集前，不宣称按需重排已经解决准确率—延迟权衡。

## 复现

```powershell
python -m pip install -r requirements-retrieval-experiment.txt

# Dev消融，不读取sealed test
python evaluation/run_retrieval_upgrade_experiment.py `
  --split dev `
  --candidate-k 12 `
  --output evaluation/private_runs/retrieval_upgrade_dev.json

# 仅在版本冻结后运行一次
python evaluation/run_retrieval_upgrade_experiment.py `
  --split test `
  --allow-sealed-test `
  --configs G_tfidf_rerank_graph `
  --candidate-k 12 `
  --output evaluation/private_runs/retrieval_upgrade_sealed_test.json
```

脚本会记录题集、片段索引和自身代码的 SHA-256、模型名称、参数、逐题检索结果及延迟。
