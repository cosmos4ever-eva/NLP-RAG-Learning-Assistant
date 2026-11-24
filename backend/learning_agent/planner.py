import json
import re
from collections.abc import Callable
from typing import Any

from .model_provider import call_agent_model
from .schemas import PlannerDecision


TOOL_CATALOG = {
    "course_qa": {
        "purpose": "基于课程证据和知识图谱回答问题",
        "arguments": {"question": "string"},
    },
    "concept_compare": {
        "purpose": "对两个易混或相关概念进行证据化辨析",
        "arguments": {"concept_a": "string", "concept_b": "string"},
    },
    "learning_path": {
        "purpose": "根据知识图谱生成目标概念的前序学习路径",
        "arguments": {"target_entity": "string"},
    },
    "generate_quiz": {
        "purpose": "围绕指定概念生成课程证据支持的练习题",
        "arguments": {
            "concept": "string",
            "num_questions": "integer, default 3",
            "difficulty": "easy | medium | hard",
        },
    },
    "learning_report": {
        "purpose": "根据学习记录、错题和覆盖度生成诊断报告",
        "arguments": {},
    },
    "create_todo": {
        "purpose": "把明确的下一步学习行动写入TodoList",
        "arguments": {"content": "string", "source": "agent"},
    },
}


class ModelPlanner:
    """Convert a learning goal and prior observations into one guarded action."""

    def __init__(self, model_call: Callable[[str], str] = call_agent_model):
        self.model_call = model_call

    def decide(
        self,
        goal: str,
        observations: list[dict[str, Any]],
        tools: list[str],
    ) -> PlannerDecision:
        prompt = self._build_prompt(goal, observations, tools)
        raw = self.model_call(prompt)
        return PlannerDecision.model_validate(self._extract_json(raw))

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        candidate = fenced.group(1) if fenced else None
        if candidate is None:
            start = raw.find("{")
            end = raw.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Planner没有返回JSON决策。")
            candidate = raw[start:end + 1]
        return json.loads(candidate)

    @staticmethod
    def _build_prompt(
        goal: str,
        observations: list[dict[str, Any]],
        tools: list[str],
    ) -> str:
        tool_specs = [
            {"name": name, **TOOL_CATALOG.get(name, {})}
            for name in tools
        ]
        return f"""
你是NLP课程学习Agent的Planner。每次只决定下一步，不要直接虚构工具结果。

学习目标：
{goal}

允许使用的工具：
{json.dumps(tool_specs, ensure_ascii=False)}

已有Observation：
{json.dumps(observations, ensure_ascii=False, default=str)}

如果还需要工具，严格返回：
{{
  "action": "tool",
  "tool": "允许使用的工具名",
  "arguments": {{}},
  "reasoning": "为什么需要这一步"
}}

如果目标已经完成，严格返回：
{{
  "action": "finish",
  "final_answer": "整合工具结果后的最终答复",
  "next_actions": ["可选的下一步学习行动"],
  "reasoning": "为什么可以结束"
}}

约束：
1. 只能选择允许使用的工具。
2. 不要重复已经成功且参数相同的工具调用。
3. 工具失败时可以调整参数、选择其他工具或说明无法完成。
4. 只输出一个JSON对象，不要输出Markdown。
""".strip()
