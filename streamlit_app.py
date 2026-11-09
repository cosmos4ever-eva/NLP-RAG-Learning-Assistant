"""Streamlit Community Cloud entrypoint for the original frontend.

The original UI keeps calling the FastAPI routes. This entrypoint starts that
API in the same container before importing the existing Streamlit app.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import streamlit as st


ROOT = Path(__file__).resolve().parent
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
API_URL = f"http://127.0.0.1:{BACKEND_PORT}"


def _secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


@st.cache_resource
def start_backend() -> subprocess.Popen[str]:
    environment = os.environ.copy()
    environment["ECNUGPT_API_KEY"] = _secret("ECNUGPT_API_KEY")
    environment["ECNUGPT_MODEL"] = _secret("ECNUGPT_MODEL") or "ecnu-plus"
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for _ in range(30):
        try:
            response = requests.get(f"{API_URL}/health", timeout=1)
            if response.ok:
                return process
        except requests.RequestException:
            time.sleep(0.25)
    process.terminate()
    raise RuntimeError("FastAPI backend did not become ready")


start_backend()
os.environ["API_BASE"] = API_URL

# Importing the existing module renders the original Streamlit frontend.
from frontend import app as original_frontend  # noqa: E402,F401
