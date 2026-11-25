import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from learning_agent.reflection import ModelReflectionAgent, ReflectiveLearningAgent
from learning_agent.schemas import (
    AgentRequest,
    AgentResponse,
    ReviewDecision,
    ToolCallRecord,
    TraceEvent,
)


class ScriptedExecutor:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        return next(self.responses)


class ScriptedReviewer:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def review(self, goal, result):
        return next(self.decisions)


def completed_response(answer="词嵌入把词映射为稠密向量。"):
    return AgentResponse(
        session_id="session-1",
        status="completed",
        final_answer=answer,
        trace=[TraceEvent(step=1, phase="finish", detail="completed")],
    )


class ReflectiveLearningAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_passed_review_returns_without_retry(self):
        executor = ScriptedExecutor([completed_response()])
        reviewer = ScriptedReviewer([
            ReviewDecision(
                verdict="pass",
                goal_completion=0.95,
                grounding=0.9,
                tool_reliability=1.0,
                feedback="目标已完成，回答有依据。",
            )
        ])
        agent = ReflectiveLearningAgent(executor=executor, reviewer=reviewer)

        result = await agent.run(AgentRequest(goal="解释词嵌入", max_reflections=1))

        self.assertEqual("completed", result.status)
        self.assertEqual(1, result.attempt_count)
        self.assertEqual("pass", result.review.verdict)
        self.assertEqual(1, len(result.review_history))
        self.assertEqual(1, len(executor.requests))

    async def test_failed_review_feedback_drives_one_revision(self):
        executor = ScriptedExecutor([
            completed_response("只解释了Word2Vec。"),
            completed_response("解释了Word2Vec和CBOW，并生成了两道题。"),
        ])
        reviewer = ScriptedReviewer([
            ReviewDecision(
                verdict="revise",
                goal_completion=0.5,
                grounding=0.8,
                tool_reliability=1.0,
                issues=["缺少CBOW对比和练习题"],
                feedback="补充CBOW对比并生成两道练习题。",
            ),
            ReviewDecision(
                verdict="pass",
                goal_completion=1.0,
                grounding=0.9,
                tool_reliability=1.0,
                feedback="修正后已完成目标。",
            ),
        ])
        agent = ReflectiveLearningAgent(executor=executor, reviewer=reviewer)

        result = await agent.run(AgentRequest(
            goal="解释Word2Vec和CBOW并生成两道题",
            session_id="learner-1",
            max_reflections=1,
        ))

        self.assertEqual("completed", result.status)
        self.assertEqual(2, result.attempt_count)
        self.assertEqual("pass", result.review.verdict)
        self.assertEqual(["revise", "pass"], [item.verdict for item in result.review_history])
        self.assertIn("补充CBOW对比", executor.requests[1].goal)
        self.assertEqual("learner-1:reflection:1", executor.requests[1].session_id)
        self.assertIn("生成了两道题", result.final_answer)

    async def test_retry_limit_returns_needs_revision_instead_of_completed(self):
        executor = ScriptedExecutor([
            completed_response("第一次不完整回答"),
            completed_response("第二次仍不完整"),
        ])
        reviewer = ScriptedReviewer([
            ReviewDecision(
                verdict="revise",
                goal_completion=0.4,
                grounding=0.7,
                tool_reliability=1.0,
                issues=["缺少练习题"],
                feedback="补充练习题。",
            ),
            ReviewDecision(
                verdict="revise",
                goal_completion=0.6,
                grounding=0.8,
                tool_reliability=1.0,
                issues=["题目数量不足"],
                feedback="需要两道题。",
            ),
        ])
        agent = ReflectiveLearningAgent(executor=executor, reviewer=reviewer)

        result = await agent.run(AgentRequest(goal="解释后出两道题", max_reflections=1))

        self.assertEqual("needs_revision", result.status)
        self.assertEqual(2, result.attempt_count)
        self.assertEqual("revise", result.review.verdict)
        self.assertEqual(2, len(result.review_history))

    async def test_unavailable_reviewer_does_not_retry_executor(self):
        executor = ScriptedExecutor([completed_response()])
        reviewer = ModelReflectionAgent(
            model_call=lambda prompt: (_ for _ in ()).throw(RuntimeError("timeout"))
        )
        agent = ReflectiveLearningAgent(executor=executor, reviewer=reviewer)

        result = await agent.run(AgentRequest(goal="解释词嵌入", max_reflections=1))

        self.assertEqual("review_failed", result.status)
        self.assertEqual(1, result.attempt_count)
        self.assertEqual(1, len(executor.requests))


class ModelReflectionAgentTest(unittest.TestCase):
    def test_returns_structured_review_from_model_output(self):
        raw = """```json
        {
          "verdict": "revise",
          "goal_completion": 0.6,
          "grounding": 0.8,
          "tool_reliability": 1.0,
          "issues": ["缺少两道练习题"],
          "feedback": "补充两道有课程依据的题目"
        }
        ```"""
        reviewer = ModelReflectionAgent(model_call=lambda prompt: raw)

        decision = reviewer.review("解释后出两道题", completed_response("只完成了解释"))

        self.assertEqual("revise", decision.verdict)
        self.assertEqual(0.6, decision.goal_completion)
        self.assertEqual(["缺少两道练习题"], decision.issues)

    def test_tool_error_is_rejected_before_model_judging(self):
        result = completed_response("看似完整的回答")
        result.tool_calls = [ToolCallRecord(
            step=1,
            tool="generate_quiz",
            arguments={"concept": "Word2Vec"},
            status="error",
            error="模型超时",
        )]
        reviewer = ModelReflectionAgent(
            model_call=lambda prompt: self.fail("tool error should bypass model judge")
        )

        decision = reviewer.review("解释后生成练习题", result)

        self.assertEqual("revise", decision.verdict)
        self.assertEqual(0.0, decision.tool_reliability)
        self.assertIn("generate_quiz", decision.feedback)

    def test_low_scores_cannot_be_marked_as_pass(self):
        raw = """{
          "verdict": "pass",
          "goal_completion": 0.4,
          "grounding": 0.5,
          "tool_reliability": 1.0,
          "issues": [],
          "feedback": ""
        }"""
        reviewer = ModelReflectionAgent(model_call=lambda prompt: raw)

        decision = reviewer.review("解释并出题", completed_response("只完成解释"))

        self.assertEqual("revise", decision.verdict)
        self.assertTrue(decision.issues)
        self.assertIn("完成度", decision.feedback)

    def test_reviewer_model_failure_returns_unavailable_decision(self):
        def broken_model(prompt):
            raise RuntimeError("provider timeout")

        reviewer = ModelReflectionAgent(model_call=broken_model)

        decision = reviewer.review("解释词嵌入", completed_response())

        self.assertEqual("revise", decision.verdict)
        self.assertFalse(decision.review_available)
        self.assertIn("评测Agent暂不可用", decision.issues)


class ReflectionRouteWiringTest(unittest.TestCase):
    def test_default_agent_uses_reflection_loop(self):
        from learning_agent.routes import get_agent

        self.assertIsInstance(get_agent(), ReflectiveLearningAgent)


if __name__ == "__main__":
    unittest.main()
