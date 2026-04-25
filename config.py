"""Shared configuration for the RAG-based second-stage review layer."""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_ENV_PATH = Path(__file__).with_name(".env")
_MODEL_NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9._:-]{1,127}$"


def _load_local_env() -> None:
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


def _env_flag(name: str, default: bool) -> bool:
    _load_local_env()
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def has_openai_package() -> bool:
    return importlib.util.find_spec("openai") is not None


def openai_api_key() -> str:
    _load_local_env()
    return os.getenv("OPENAI_API_KEY", "").strip()


def review_llm_available() -> bool:
    return bool(openai_api_key()) and has_openai_package()


def _is_valid_model_name(model_name: str) -> bool:
    cleaned = (model_name or "").strip()
    if not cleaned or "=" in cleaned or any(ch.isspace() for ch in cleaned):
        return False

    import re

    return bool(re.match(_MODEL_NAME_RE, cleaned))


def review_model_name() -> str:
    _load_local_env()
    configured = os.getenv("OPENAI_REVIEW_MODEL", "").strip()
    if _is_valid_model_name(configured):
        return configured
    return "gpt-4o-mini"


@lru_cache(maxsize=1)
def review_client():
    api_key = openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai>=1.30 is required for second-stage review.") from exc

    return OpenAI(api_key=api_key, timeout=45.0)


CLAUSE_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "agreement date": ("Agreement Date",),
    "anti-assignment": ("Anti-Assignment",),
    "arbitration": ("Arbitration",),
    "assignment": ("Anti-Assignment",),
    "audit rights": ("Audit Rights",),
    "change of control": ("Change Of Control",),
    "covenant not to sue": ("Covenant Not To Sue",),
    "dispute resolution": ("Dispute Resolution",),
    "document name": ("Document Name",),
    "effective date": ("Effective Date",),
    "exclusivity": ("Exclusivity",),
    "expiration": ("Expiration Date",),
    "expiration date": ("Expiration Date",),
    "force majeure": ("Force Majeure",),
    "governing law": ("Governing Law",),
    "indemnification": ("Indemnification",),
    "indemnity": ("Indemnification",),
    "insurance": ("Insurance",),
    "ip ownership": ("IP Ownership Assignment", "Joint IP Ownership"),
    "ip ownership assignment": ("IP Ownership Assignment",),
    "jurisdiction": ("Dispute Resolution", "Governing Law"),
    "license grant": ("License Grant",),
    "limitation of liability": ("Cap On Liability", "Uncapped Liability"),
    "most favored nation": ("Most Favored Nation",),
    "mfn": ("Most Favored Nation",),
    "non-compete": ("Non-Compete",),
    "non-solicit": ("No-Solicit Of Customers", "No-Solicit Of Employees"),
    "parties": ("Parties",),
    "renewal": ("Renewal Term",),
    "renewal term": ("Renewal Term",),
    "source code escrow": ("Source Code Escrow",),
    "termination": ("Termination For Convenience",),
    "termination for convenience": ("Termination For Convenience",),
}

MANUAL_HARD_NEGATIVE_LABELS: dict[str, tuple[str, ...]] = {
    "Governing Law": ("Jurisdiction",),
    "Jurisdiction": ("Governing Law",),
    "Termination": ("Expiration", "Renewal"),
    "Termination For Convenience": ("Expiration", "Renewal"),
    "Non-Compete": ("Non-Solicit", "Exclusivity"),
    "Exclusivity": ("Non-Compete", "Most Favored Nation"),
    "Indemnification": ("Limitation of Liability",),
    "Limitation of Liability": ("Indemnification",),
    "Cap On Liability": ("Indemnification",),
    "IP Ownership": ("License Grant",),
    "IP Ownership Assignment": ("License Grant",),
    "License Grant": ("IP Ownership",),
    "Change of Control": ("Assignment",),
    "Change Of Control": ("Assignment",),
    "Assignment": ("Change of Control",),
    "Anti-Assignment": ("Change of Control",),
}


def expand_label_alias(label: str, known_labels: set[str] | None = None) -> list[str]:
    """Expand a user-facing alias to one or more concrete CUAD labels."""
    cleaned = (label or "").strip()
    if not cleaned:
        return []

    if known_labels and cleaned in known_labels:
        return [cleaned]

    key = cleaned.lower()
    if key in CLAUSE_ALIAS_MAP:
        expanded = [item for item in CLAUSE_ALIAS_MAP[key] if not known_labels or item in known_labels]
        if expanded:
            return expanded

    if known_labels:
        matches = [item for item in known_labels if item.lower() == key]
        if matches:
            return matches

    return [cleaned]


def normalize_label(label: str, known_labels: set[str] | None = None) -> str:
    expanded = expand_label_alias(label, known_labels=known_labels)
    return expanded[0] if expanded else label


@dataclass(frozen=True)
class ReviewConfig:
    """Runtime settings for the second-stage review pipeline."""

    data_dir: str = "data/cuad"
    rag_embedding_backend: str = field(
        default_factory=lambda: os.getenv("LEXSCAN_RAG_EMBEDDING_BACKEND", "tfidf").strip().lower()
    )
    rag_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "LEXSCAN_RAG_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )
    rag_index_cache_path: str = "checkpoints/rag_index.joblib"
    review_model: str = field(default_factory=review_model_name)
    split_seed: int = field(default_factory=lambda: int(os.getenv("LEXSCAN_REVIEW_SPLIT_SEED", "42")))
    retriever_top_positives: int = field(default_factory=lambda: int(os.getenv("LEXSCAN_RAG_TOP_POSITIVES", "3")))
    retriever_top_negatives: int = field(default_factory=lambda: int(os.getenv("LEXSCAN_RAG_TOP_NEGATIVES", "2")))
    model_top_k: int = field(default_factory=lambda: int(os.getenv("LEXSCAN_REVIEW_TOP_K_LABELS", "3")))
    example_snippet_chars: int = field(default_factory=lambda: int(os.getenv("LEXSCAN_RAG_SNIPPET_CHARS", "320")))
    enable_second_stage_review: bool = field(default_factory=lambda: _env_flag("LEXSCAN_ENABLE_SECOND_STAGE_REVIEW", True))


def get_review_config() -> ReviewConfig:
    _load_local_env()
    return ReviewConfig()
