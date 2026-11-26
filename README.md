# RAG_teaching_assistant
基于 RAG 与知识图谱的 NLP 课程学习智能助手

## Learning Agent

系统基于 LangChain 与 LangGraph 提供受控的 Planner–Executor 学习 Agent。
六类学习能力使用 LangChain 结构化工具封装；用户提交学习目标后，Planner
基于工具 Schema 选择下一步行动，LangGraph Executor 调用白名单工具，将
Observation 写入线程级短期会话状态，并循环至完成或触发停止条件。

```text
START -> Plan -> Tool Calling -> Observation -> Re-plan / Finish
                    |                         |
                    +-- Checkpoint / Trace ---+
```

当前工具：

- `course_qa`：基于课程证据和知识图谱回答问题；
- `concept_compare`：辨析易混概念；
- `learning_path`：生成前序学习路径；
- `generate_quiz`：生成结构化练习；
- `learning_report`：生成学习诊断；
- `create_todo`：写入下一步学习任务。

统一入口：

```http
POST /api/agent/run
Content-Type: application/json

{
  "goal": "先解释 Word2Vec 和 CBOW 的关系，再生成两道练习题",
  "max_steps": 4,
  "max_reflections": 1
}
```

Agent返回最终答复、下一步行动、工具调用记录和Planner–Executor Trace。
LangGraph的`InMemorySaver`以`session_id`作为`thread_id`保存线程级Checkpoint，
同一进程中的后续请求可复用最近20条工具Observation；服务重启后该短期状态会
清空。工具白名单、结构化决策、重复调用阻断、最大步数和失败降级共同约束执行。

Planner支持DeepSeek和ECNU模型供应商。有`DEEPSEEK_API_KEY`时默认使用
DeepSeek；否则使用`ECNUGPT_API_KEY`。也可以通过
`AGENT_MODEL_PROVIDER=deepseek|ecnu`固定供应商，通过`DEEPSEEK_MODEL`或
`ECNUGPT_MODEL`固定模型版本。

## Reflection Agent

执行Agent生成结果后，LangGraph将状态路由到独立的评测/反思Agent，检查目标
完成度、课程依据和工具可靠性。评测Agent只负责返回结构化结论和修改要求，不能
直接调用学习工具；执行Agent也不能自行判定最终通过。

```text
Learning Goal
    -> Executor Agent
    -> Reflection Agent
         | pass   -> Final Result
         | revise -> Feedback -> Executor Retry -> Re-review
```

默认最多反思重试一次。工具错误、空回答和未完成状态会在模型评审前被确定性
规则直接退回；模型返回的完成度、依据性或工具可靠性低于阈值时，即使文字结论
写成`pass`也会被改判为`revise`。达到重试上限后仍未通过时，接口返回
`needs_revision`，不会把结果标记为完成。
