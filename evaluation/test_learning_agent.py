import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from learning_agent.agent import LearningAgent
from learning_agent.planner import ModelPlanner
from learning_agent.model_provider import call_agent_model, resolve_agent_provider
from learning_agent.schemas import AgentRequest, PlannerDecision
from learning_agent.tools import build_default_tools


class ScriptedPlanner:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.observation_snapshots = []

    def decide(self, goal, observations, tools):
        self.observation_snapshots.append(list(observations))
        return next(self.decisions)


class LearningAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_executes_planned_tool_and_returns_observable_trace(self):
        planner = ScriptedPlanner([
            PlannerDecision(
                action="tool",
                tool="course_qa",
                arguments={"question": "什么是词嵌入？"},
                reasoning="先检索课程证据并回答。",
            ),
            PlannerDecision(
                action="finish",
                final_answer="词嵌入把词映射为稠密向量。",
                reasoning="已有足够课程证据。",
            ),
        ])

        async def course_qa(arguments):
            self.assertEqual("什么是词嵌入？", arguments["question"])
            return {"answer": "词嵌入把词映射为稠密向量。", "sources": ["L5_p3"]}

        agent = LearningAgent(planner=planner, tools={"course_qa": course_qa})
        result = await agent.run(AgentRequest(goal="解释词嵌入", max_steps=3))

        self.assertEqual("completed", result.status)
        self.assertEqual("词嵌入把词映射为稠密向量。", result.final_answer)
        self.assertEqual(1, len(result.tool_calls))
        self.assertEqual("course_qa", result.tool_calls[0].tool)
        self.assertEqual("success", result.tool_calls[0].status)
        self.assertEqual("finish", result.trace[-1].phase)

    async def test_reuses_observations_from_the_same_session(self):
        planner = ScriptedPlanner([
            PlannerDecision(
                action="tool",
                tool="course_qa",
                arguments={"question": "什么是CBOW？"},
            ),
            PlannerDecision(action="finish", final_answer="第一轮完成"),
            PlannerDecision(action="finish", final_answer="第二轮完成"),
        ])

        async def course_qa(arguments):
            return {"answer": "CBOW用上下文预测中心词。"}

        agent = LearningAgent(planner=planner, tools={"course_qa": course_qa})
        await agent.run(AgentRequest(goal="解释CBOW", session_id="learner-1"))
        await agent.run(AgentRequest(goal="根据刚才内容继续", session_id="learner-1"))

        second_run_observations = planner.observation_snapshots[2]
        self.assertEqual(1, len(second_run_observations))
        self.assertEqual("course_qa", second_run_observations[0]["tool"])
        self.assertEqual("success", second_run_observations[0]["status"])

    async def test_planner_failure_is_returned_as_a_safe_agent_result(self):
        class BrokenPlanner:
            def decide(self, goal, observations, tools):
                raise ValueError("invalid planner JSON")

        agent = LearningAgent(planner=BrokenPlanner(), tools={})
        result = await agent.run(AgentRequest(goal="解释HMM"))

        self.assertEqual("failed", result.status)
        self.assertIn("规划失败", result.final_answer)
        self.assertEqual("guardrail", result.trace[-1].phase)
        self.assertNotIn("invalid planner JSON", result.final_answer)

    async def test_duplicate_tool_call_is_blocked(self):
        planner = ScriptedPlanner([
            PlannerDecision(
                action="tool",
                tool="course_qa",
                arguments={"question": "解释HMM"},
            ),
            PlannerDecision(
                action="tool",
                tool="course_qa",
                arguments={"question": "解释HMM"},
            ),
            PlannerDecision(action="finish", final_answer="已完成"),
        ])
        execution_count = 0

        async def course_qa(arguments):
            nonlocal execution_count
            execution_count += 1
            return {"answer": "HMM是隐马尔可夫模型。"}

        agent = LearningAgent(planner=planner, tools={"course_qa": course_qa})
        result = await agent.run(AgentRequest(goal="解释HMM", max_steps=3))

        self.assertEqual(1, execution_count)
        self.assertEqual("error", result.tool_calls[1].status)
        self.assertIn("重复", result.tool_calls[1].error)
        self.assertEqual("completed", result.status)


class ModelPlannerTest(unittest.TestCase):
    def test_parses_a_structured_tool_decision_from_model_output(self):
        raw = """```json
        {
          "action": "tool",
          "tool": "generate_quiz",
          "arguments": {"concept": "Word2Vec", "num_questions": 2},
          "reasoning": "用户要求在解释后进行练习"
        }
        ```"""
        planner = ModelPlanner(model_call=lambda prompt: raw)

        decision = planner.decide(
            goal="解释Word2Vec并出两道题",
            observations=[{"tool": "course_qa", "status": "success"}],
            tools=["course_qa", "generate_quiz"],
        )

        self.assertEqual("tool", decision.action)
        self.assertEqual("generate_quiz", decision.tool)
        self.assertEqual(2, decision.arguments["num_questions"])

    def test_prompt_exposes_tool_purpose_and_argument_schema(self):
        prompt = ModelPlanner._build_prompt(
            goal="学习Word2Vec",
            observations=[],
            tools=["learning_path", "generate_quiz"],
        )

        self.assertIn("target_entity", prompt)
        self.assertIn("num_questions", prompt)
        self.assertIn("前序学习路径", prompt)


class AgentModelProviderTest(unittest.TestCase):
    def test_deepseek_is_preferred_when_both_keys_exist(self):
        with patch.dict(
            "os.environ",
            {"DEEPSEEK_API_KEY": "deepseek-key", "ECNUGPT_API_KEY": "ecnu-key"},
            clear=True,
        ):
            provider = resolve_agent_provider()

        self.assertEqual("deepseek", provider["name"])
        self.assertEqual("deepseek-flash", provider["model"])

    def test_provider_can_be_explicitly_pinned_to_ecnu(self):
        with patch.dict(
            "os.environ",
            {"AGENT_MODEL_PROVIDER": "ecnu", "ECNUGPT_API_KEY": "ecnu-key"},
            clear=True,
        ):
            provider = resolve_agent_provider()

        self.assertEqual("ecnu", provider["name"])

    def test_explicit_ecnu_choice_is_forwarded_to_generation_client(self):
        with patch.dict(
            "os.environ",
            {
                "AGENT_MODEL_PROVIDER": "ecnu",
                "DEEPSEEK_API_KEY": "deepseek-key",
                "ECNUGPT_API_KEY": "ecnu-key",
            },
            clear=True,
        ), patch(
            "learning_agent.model_provider.call_ecnu_gpt",
            return_value="ok",
        ) as mocked:
            result = call_agent_model("plan this")

        self.assertEqual("ok", result)
        mocked.assert_called_once_with("plan this", provider_name="ecnu")


class AgentToolsTest(unittest.IsolatedAsyncioTestCase):
    async def test_learning_path_tool_uses_the_real_course_graph(self):
        tools = build_default_tools()

        result = await tools["learning_path"]({"target_entity": "Word2Vec"})

        self.assertEqual("Word2Vec", result["target"])
        self.assertIn("Word2Vec", result["path"])
        self.assertIn("course_qa", tools)
        self.assertIn("generate_quiz", tools)
        self.assertIn("learning_report", tools)


class LearningAgentRouteTest(unittest.TestCase):
    def test_run_endpoint_returns_agent_trace(self):
        from main import app
        from learning_agent.routes import get_agent

        planner = ScriptedPlanner([
            PlannerDecision(action="finish", final_answer="已完成学习目标")
        ])
        fake_agent = LearningAgent(planner=planner, tools={})
        app.dependency_overrides[get_agent] = lambda: fake_agent
        self.addCleanup(app.dependency_overrides.clear)

        response = TestClient(app).post(
            "/api/agent/run",
            json={"goal": "制定一个NLP学习计划", "max_steps": 2},
        )

        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("completed", body["status"])
        self.assertEqual("已完成学习目标", body["final_answer"])
        self.assertEqual("finish", body["trace"][-1]["phase"])


if __name__ == "__main__":
    unittest.main()
