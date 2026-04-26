"""Shared OpenAI helpers for LexScan services."""
from __future__ import annotations

import importlib.util
import os
import re
from functools import lru_cache
from pathlib import Path

_ENV_PATH = Path(__file__).with_name(".env")
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{1,127}$")


def load_local_env() -> None:
    """Best-effort .env loader for local development."""
    if not _ENV_PATH.exists():
        return

    try:
        raw = _ENV_PATH.read_text(encoding="utf-8")
    except OSError:
        return

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def has_openai_package() -> bool:
    return importlib.util.find_spec("openai") is not None


def openai_api_key() -> str:
    load_local_env()
    return os.getenv("OPENAI_API_KEY", "").strip()


def is_valid_model_name(model_name: str) -> bool:
    cleaned = (model_name or "").strip()
    if not cleaned or "=" in cleaned or any(ch.isspace() for ch in cleaned):
        return False
    return bool(_MODEL_NAME_RE.match(cleaned))


def model_from_env(env_name: str, fallback: str) -> str:
    load_local_env()
    configured = os.getenv(env_name, "").strip()
    if is_valid_model_name(configured):
        return configured
    return fallback


@lru_cache(maxsize=4)
def get_openai_client(timeout: float = 30.0):
    api_key = openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai>=1.30 is required for OpenAI-backed features.") from exc

    return OpenAI(api_key=api_key, timeout=timeout)
