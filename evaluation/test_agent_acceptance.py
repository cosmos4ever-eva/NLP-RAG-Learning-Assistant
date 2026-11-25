import unittest
from pathlib import Path

from evaluation.run_agent_acceptance import load_tasks, summarize


ROOT = Path(__file__).resolve().parents[1]


class AgentAcceptanceTest(unittest.TestCase):
    def test_frozen_dataset_has_unique_tasks_and_supported_tools(self):
        tasks = load_tasks(ROOT / "evaluation" / "agent_acceptance_v1.jsonl")
        supported = {
            "course_qa",
            "concept_compare",
            "learning_path",
            "generate_quiz",
            "learning_report",
            "create_todo",
        }

        self.assertEqual(10, len(tasks))
        self.assertEqual(10, len({task["id"] for task in tasks}))
        self.assertTrue(all(set(task["required_tools"]) <= supported for task in tasks))
        self.assertEqual(2, sum(task["expected_refusal"] for task in tasks))

    def test_summary_separates_completion_from_deterministic_task_pass(self):
        rows = [
            {
                "passed": True,
                "status": "completed",
                "required_tools": ["course_qa"],
                "required_tool_recall": 1.0,
                "tool_precision": 1.0,
                "tool_success": True,
                "attempt_count": 1,
                "expected_refusal": False,
                "detected_refusal": False,
                "latency_seconds": 10.0,
            },
            {
                "passed": False,
                "status": "completed",
                "required_tools": ["course_qa"],
                "required_tool_recall": 0.0,
                "tool_precision": 0.0,
                "tool_success": True,
                "attempt_count": 2,
                "expected_refusal": True,
                "detected_refusal": False,
                "latency_seconds": 20.0,
            },
        ]

        summary = summarize(rows)

        self.assertEqual(2, summary["completed_count"])
        self.assertEqual(1, summary["task_pass_count"])
        self.assertEqual(0.5, summary["required_tool_recall"])
        self.assertEqual(1, summary["reflection_trigger_count"])
        self.assertEqual(0, summary["boundary_refusal_count"])
        self.assertEqual(15.0, summary["mean_latency_seconds"])


if __name__ == "__main__":
    unittest.main()
