"""Shared configuration for the RAG-based second-stage review layer."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from openai_utils import (
    get_openai_client,
    has_openai_package,
    is_valid_model_name,
    load_local_env,
    model_from_env,
    openai_api_key,
)


def _env_flag(name: str, default: bool) -> bool:
    load_local_env()
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def review_llm_available() -> bool:
    return bool(openai_api_key()) and has_openai_package()


def review_model_name() -> str:
    return model_from_env("OPENAI_REVIEW_MODEL", "gpt-4o-mini")


def chat_model_name() -> str:
    return model_from_env("LEXSCAN_CHAT_MODEL", "gpt-4o-mini")


@lru_cache(maxsize=1)
def review_client():
    if not openai_api_key():
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return get_openai_client(timeout=45.0)


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
    enable_chatbot: bool = field(default_factory=lambda: _env_flag("LEXSCAN_ENABLE_CHATBOT", True))
    chat_model: str = field(default_factory=chat_model_name)
    chat_top_k: int = field(default_factory=lambda: int(os.getenv("LEXSCAN_CHAT_TOP_K", "4")))
    chat_chunk_chars: int = field(default_factory=lambda: int(os.getenv("LEXSCAN_CHAT_CHUNK_CHARS", "900")))
    chat_chunk_overlap: int = field(default_factory=lambda: int(os.getenv("LEXSCAN_CHAT_CHUNK_OVERLAP", "160")))
    chat_history_turns: int = field(default_factory=lambda: int(os.getenv("LEXSCAN_CHAT_HISTORY_TURNS", "6")))
    chat_min_score: float = field(default_factory=lambda: float(os.getenv("LEXSCAN_CHAT_MIN_SCORE", "0.12")))
    max_rag_cache_mb: int = field(default_factory=lambda: int(os.getenv("LEXSCAN_MAX_RAG_CACHE_MB", "768")))


def get_review_config() -> ReviewConfig:
    load_local_env()
    return ReviewConfig()
