# Graph-RAG 教学助手评测集

这套评测用于回答一个具体问题：

> 显式课程知识图谱是否比普通 RAG 更能帮助系统生成有课程依据的概念解释、概念关系和学习路径？

评测集不是产品功能清单。它固定问题、证据与评分标准，用于比较不同系统版本并记录 Bad Case。

## 文件

- `questions.jsonl`：56 个带金标准的问题，dev/test 各 28 个。
- `rubric.md`：人工评分与汇总规则。
- `annotations_template.csv`：人工标注记录模板。
- `validate_dataset.py`：结构、编号、证据定位和图谱关系校验。

## 数据构成

题目分为三组：

1. `course_grounded`：答案应由当前仓库中的 L2-L8 课程资料支持。
2. `cross_concept`：需要组合多个课件页或知识图谱关系，不能只复述一个片段。
3. `boundary`：当前课程资料和知识图谱不足以回答，系统应明确说明证据不足，而不是依赖模型常识扩写。

`dev` 用于调试检索、提示词和图谱逻辑；`test` 只用于冻结版本后的最终比较。不要根据 `test` 的金标准逐题调参。

`DEV_EVALUATION_20261123.md` 是扩充前 24 条 dev 的历史快照。加入 L3、L4、L5、L7 后，新的 dev 有 28 条；旧报告中的比例不能直接当作新数据集结果，必须重新运行实验。

`DEV_ITERATION_20261124.md` 汇总了扩充后 28 条 dev 的有效基线、top-k/邻页敏感性实验、上下文补图迭代和最新 Graph-RAG-only 复测。确定性敏感性明细保存在 `retrieval_sensitivity_dev_20261124.json`；其中不包含课程原文。原始回答和检索片段仍只保存在被忽略的 `private_runs/`。

`repeat_stability_dev_20261124.json` 汇总三轮 Graph-RAG dev 重复实验的均值、波动、token、延迟和逐题分数稳定性，不包含回答正文。可用 `analyze_repeats.py` 对新的重复运行目录重新生成。

`TEST_EVALUATION_20261124.md` 是提交 `8c55b6d` 冻结后唯一一次 sealed-test 运行的最终报告。该结果只能用于报告与未来版本的整体比较，不能再用于修改当前冻结版本。

## 建议比较的系统

保持同一大模型、temperature、最大输出长度和问题顺序，只改变可用上下文：

1. `llm_only`：不提供课程片段和知识图谱。
2. `tfidf_rag`：仅提供 TF-IDF top-k 课程片段。
3. `graph_rag`：提供相同 top-k 课程片段及知识图谱关系。

推荐固定 `top_k=3` 作为主结果，再用 `top_k=5` 做敏感性检查。所有调用保存模型名称、时间、prompt版本、响应和耗时。

生成器支持 `--provider ecnu` 和 `--provider deepseek`。DeepSeek 批量评测默认使用 `deepseek-flash`、关闭思考模式，并从 `DEEPSEEK_API_KEY` 读取密钥；三个对照系统必须使用同一供应商和模型。`deepseek-flash` 的非思考模式温度由服务端固定，因此实验清单将 `temperature` 记录为 `null`，不把未生效的温度参数写成受控变量。

## 运行顺序

```powershell
python evaluation/validate_dataset.py
```

通过校验后：

1. 对三个系统逐题生成回答，保存原始输出。
2. 计算检索 `Evidence Hit@3`：top-3 chunk 中是否包含任一金标准页/幻灯片。
3. 两名标注者依据 `rubric.md` 独立评分。
4. 对分歧大于 2 分的样本进行复核。
5. 分别报告 dev/test、题目类型和边界题结果。
6. 至少整理 5 个 Bad Case：问题、检索证据、系统回答、错误类型、修改、复测结果。

冻结 dev 版本后，最终 test 只运行一次：

```powershell
python evaluation/run_experiment.py --split test --generate --systems llm_only,tfidf_rag,graph_rag --provider deepseek --output evaluation/private_runs/<frozen-test-run>
python evaluation/score_experiment.py --run-dir evaluation/private_runs/<frozen-test-run> --model deepseek-v4-pro
```

test 结果只能用于最终报告和未来新版本的整体比较，不能再用于逐题修改当前版本。

## 证据边界

- `source_evidence` 是定位提示，不是可以直接复制进回答的标准答案。
- 金标准使用概括性答案要点，避免大段复制课件。
- 当前 PDF/PPT 为课程资料，公开仓库或申请材料中不得附带原文件或大段课件文本，除非已获得授权。
- 如果后续补入 L3、L4、L5、L7 课件，应新增评测版本，不要静默修改当前边界题的预期行为。

## 用户实验的最小扩展

离线评测完成后，可招募 8–12 名学过基础 NLP 的学生做小规模 pilot：

- 每人完成等量的 `tfidf_rag` 与 `graph_rag` 任务，顺序交叉平衡。
- 记录答案正确率、完成时间、对依据的理解、信任评分和访谈反馈。
- 只报告 pilot 证据和局限，不把小样本描述为普遍学习效果。
- 不收集无关个人信息；保存前去标识化并取得参与者知情同意。
