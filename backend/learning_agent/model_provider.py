import os

import requests

from graph_rag.pipeline import call_ecnu_gpt


def resolve_agent_provider() -> dict[str, str]:
    requested = os.getenv("AGENT_MODEL_PROVIDER", "").strip().lower()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    ecnu_key = os.getenv("ECNUGPT_API_KEY", "").strip()

    if not requested:
        requested = "deepseek" if deepseek_key else "ecnu"

    if requested == "deepseek":
        if not deepseek_key:
            raise RuntimeError("缺少DEEPSEEK_API_KEY，无法运行Agent Planner。")
        return {
            "name": "deepseek",
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            "api_key": deepseek_key,
        }

    if requested == "ecnu":
        if not ecnu_key:
            raise RuntimeError("缺少ECNUGPT_API_KEY，无法运行Agent Planner。")
        return {
            "name": "ecnu",
            "model": os.getenv("ECNUGPT_MODEL", "ecnu-plus"),
            "api_key": ecnu_key,
        }

    raise RuntimeError(f"不支持的Agent模型供应商：{requested}")


def call_agent_model(prompt: str) -> str:
    provider = resolve_agent_provider()
    if provider["name"] == "ecnu":
        return call_ecnu_gpt(prompt, provider_name="ecnu")

    payload = {
        "model": provider["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1200,
    }
    if provider["model"] == "deepseek-flash":
        payload["thinking"] = {"type": "disabled"}
    else:
        payload["temperature"] = 0.1

    response = requests.post(
        f"{provider['base_url']}/chat/completions",
        headers={
            "Authorization": f"Bearer {provider['api_key']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
