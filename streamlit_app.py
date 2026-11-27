"""Streamlit Community Cloud entrypoint for the original frontend.

The original UI keeps calling the FastAPI routes. On Community Cloud this
entrypoint connects it to the same FastAPI app in-process, avoiding a second
localhost server inside the Streamlit sandbox.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import streamlit as st

from deployment_runtime import build_backend_environment, install_local_api_client


ROOT = Path(__file__).resolve().parent
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
API_URL = f"http://127.0.0.1:{BACKEND_PORT}"

st.set_page_config(
    page_title="NLP Graph-RAG Teaching Assistant",
    page_icon="",
    layout="wide",
)
os.environ["STREAMLIT_PAGE_CONFIGURED"] = "1"


def _secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def start_backend() -> TestClient:
    """Connect the Streamlit run to the in-process FastAPI application.

    TestClient must not be stored in ``st.cache_resource``. Streamlit 1.41 can
    finish the script immediately after replaying that cached resource, which
    leaves Community Cloud with a healthy process but an empty page.
    """
    environment = build_backend_environment(os.environ, _secret)
    os.environ.update(environment)
    backend_directory = str(ROOT / "backend")
    if backend_directory not in sys.path:
        sys.path.insert(0, backend_directory)
    from main import app as backend_app

    client = TestClient(backend_app)
    health = client.get("/health")
    health.raise_for_status()
    install_local_api_client(client)
    return client


start_backend()
os.environ["API_BASE"] = API_URL

# Importing the existing module renders the original Streamlit frontend.
from frontend import app as original_frontend  # noqa: E402,F401
