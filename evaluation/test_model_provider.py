import unittest
from unittest.mock import patch

from evaluation.run_experiment import build_model_payload, resolve_provider
from graph_rag.pipeline import resolve_generation_provider


class ModelProviderTest(unittest.TestCase):
    def test_product_generation_prefers_deepseek_when_configured(self):
        with patch.dict(
            "os.environ",
            {"DEEPSEEK_API_KEY": "deepseek-key", "ECNUGPT_API_KEY": "ecnu-key"},
            clear=True,
        ):
            provider = resolve_generation_provider()

        self.assertEqual("deepseek", provider["name"])
        self.assertEqual("deepseek-flash", provider["model"])

    def test_deepseek_default_is_non_thinking_flash(self):
        provider = resolve_provider('deepseek')
        payload = build_model_payload('test prompt', provider)

        self.assertEqual('https://api.deepseek.com', provider['base_url'])
        self.assertEqual('DEEPSEEK_API_KEY', provider['key_env'])
        self.assertEqual('deepseek-flash', payload['model'])
        self.assertEqual({'type': 'disabled'}, payload['thinking'])
        self.assertNotIn('temperature', payload)

    def test_ecnu_configuration_remains_available(self):
        provider = resolve_provider('ecnu')
        payload = build_model_payload('test prompt', provider)

        self.assertEqual('ECNUGPT_API_KEY', provider['key_env'])
        self.assertEqual(0.3, payload['temperature'])
        self.assertNotIn('thinking', payload)

    def test_model_override_is_recorded(self):
        provider = resolve_provider('deepseek', 'deepseek-v4-pro')

        self.assertEqual('deepseek-v4-pro', provider['model'])
        self.assertEqual(0.3, provider['temperature'])


if __name__ == '__main__':
    unittest.main()
