"""Pure deployment helpers shared by Streamlit entrypoint and tests."""
from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import requests


MODEL_CONFIGURATION_NAMES = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "LEARNING_MODEL_PROVIDER",
    "AGENT_MODEL_PROVIDER",
    "ECNUGPT_API_KEY",
    "ECNUGPT_MODEL",
)


def build_local_api_session() -> requests.Session:
    """Create a session that never routes localhost traffic through cloud proxies."""
    session = requests.Session()
    session.trust_env = False
    return session


def build_backend_environment(
    base_environment: Mapping[str, str],
    secret_loader: Callable[[str], str],
) -> dict[str, str]:
    """Merge non-empty deployment secrets without erasing existing values."""
    environment = dict(base_environment)
    for name in ("NO_PROXY", "no_proxy"):
        existing_hosts = [
            host.strip() for host in environment.get(name, "").split(",") if host.strip()
        ]
        for local_host in ("127.0.0.1", "localhost"):
            if local_host not in existing_hosts:
                existing_hosts.append(local_host)
        environment[name] = ",".join(existing_hosts)
    for name in MODEL_CONFIGURATION_NAMES:
        value = secret_loader(name).strip()
        if value:
            environment[name] = value
    environment.setdefault("ECNUGPT_MODEL", "ecnu-plus")
    return environment


def backend_process_spec(root: Path, port: int) -> tuple[list[str], Path]:
    """Return the import-safe Uvicorn command and its working directory."""
    return (
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        root / "backend",
    )
