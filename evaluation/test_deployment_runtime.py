import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment_runtime import backend_process_spec, build_backend_environment


class DeploymentRuntimeTest(unittest.TestCase):
    def test_backend_starts_from_backend_directory(self):
        command, cwd = backend_process_spec(ROOT, 8123)

        self.assertEqual(ROOT / "backend", cwd)
        self.assertEqual("main:app", command[3])
        self.assertEqual("8123", command[-1])

    def test_model_secrets_are_forwarded_to_backend(self):
        values = {
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "DEEPSEEK_MODEL": "deepseek-chat",
            "LEARNING_MODEL_PROVIDER": "deepseek",
            "AGENT_MODEL_PROVIDER": "deepseek",
            "ECNUGPT_API_KEY": "ecnu-secret",
            "ECNUGPT_MODEL": "ecnu-plus",
        }

        environment = build_backend_environment(
            {"EXISTING": "kept"},
            lambda name: values.get(name, ""),
        )

        self.assertEqual("kept", environment["EXISTING"])
        for name, value in values.items():
            self.assertEqual(value, environment[name])

    def test_empty_optional_secrets_do_not_overwrite_existing_environment(self):
        environment = build_backend_environment(
            {"DEEPSEEK_MODEL": "existing-model"},
            lambda _name: "",
        )

        self.assertEqual("existing-model", environment["DEEPSEEK_MODEL"])


if __name__ == "__main__":
    unittest.main()
