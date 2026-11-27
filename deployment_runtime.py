"""Pure deployment helpers shared by Streamlit entrypoint and tests."""
from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

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

_LOCAL_API_CLIENT: Any | None = None


def build_local_api_session() -> requests.Session:
    """Create a session that never routes localhost traffic through cloud proxies."""
    session = requests.Session()
    session.trust_env = False
    return session


def install_local_api_client(client: Any | None) -> None:
    """Install an in-process API client for single-container deployments."""
    global _LOCAL_API_CLIENT
    _LOCAL_API_CLIENT = client


def get_local_api_client() -> Any:
    """Return the injected in-process client or a proxy-free HTTP session."""
    return _LOCAL_API_CLIENT or build_local_api_session()


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
