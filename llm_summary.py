"""OpenAI-backed contract summary helpers for the LexScan Flask app."""
from __future__ import annotations

import importlib.util
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_CHARS_TYPE = 3_000
_MAX_CHARS_SUMMARY = 6_000
_SUMMARY_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[\.\)])\s*")
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{1,127}$")
_ENV_PATH = Path(__file__).with_name(".env")


def _load_local_env() -> None:
    """Best-effort parse of a local .env file for development convenience."""
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


def _summary_model() -> str:
    _load_local_env()
    configured = os.getenv("OPENAI_CONTRACT_SUMMARY_MODEL", "").strip()
    if _is_valid_model_name(configured):
        return configured
    return "gpt-4o"


def _summary_model_candidates() -> list[str]:
    configured = _summary_model()
    candidates = [
        configured,
        "gpt-4o",
        "gpt-4o-mini",
    ]
    output: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        cleaned = item.strip()
        if not _is_valid_model_name(cleaned) or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def _api_key() -> str:
    _load_local_env()
    return os.getenv("OPENAI_API_KEY", "").strip()


def _has_openai_package() -> bool:
    return importlib.util.find_spec("openai") is not None


def is_summary_enabled() -> bool:
    return bool(_api_key()) and _has_openai_package()


def _is_valid_model_name(model_name: str) -> bool:
    cleaned = (model_name or "").strip()
    if not cleaned or "=" in cleaned or any(ch.isspace() for ch in cleaned):
        return False
    return bool(_MODEL_NAME_RE.match(cleaned))


def summary_unavailable_payload(
    message: str | None = None,
    *,
    status: str = "disabled",
) -> dict:
    return {
        "available": False,
        "status": status,
        "message": message or "AI contract summary is unavailable.",
        "doc_type": "",
        "bullets": [],
        "model": "",
    }


@lru_cache(maxsize=1)
def _client():
    api_key = _api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI package is not installed. Add openai>=1.30 to enable summaries."
        ) from exc

    return OpenAI(api_key=api_key, timeout=30.0)


def _is_invalid_model_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "invalid model id" in text or "model_not_found" in text or "does not exist" in text


def _chat_completion_with_model_fallback(
    *,
    messages: list[dict[str, str]],
    temperature: float,
) -> tuple[str, str]:
    last_exc: Exception | None = None
    for model_name in _summary_model_candidates():
        try:
            resp = _client().chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=messages,
            )
            return (resp.choices[0].message.content or "").strip(), model_name
        except Exception as exc:
            last_exc = exc
            if _is_invalid_model_error(exc):
                logger.warning("Summary model %s failed; trying fallback model.", model_name)
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No valid OpenAI summary model candidates are configured.")


def classify_document_type(text: str) -> tuple[str, str]:
    """Return document type as a short label, e.g. 'Software License Agreement'."""
    content, model_name = _chat_completion_with_model_fallback(
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal assistant. Identify the contract document type "
                    "in 3-6 words (for example: 'Non-Disclosure Agreement', "
                    "'Software License Agreement', 'Employment Contract'). "
                    "Reply with only the document type."
                ),
            },
            {
                "role": "user",
                "content": f"Contract excerpt:\n\n{text[:_MAX_CHARS_TYPE]}",
            },
        ],
    )
    return content or "Unknown Document Type", model_name


def summarize_contract(text: str, doc_type: str) -> tuple[str, str]:
    """Return a plain-English bullet-point summary aimed at non-lawyers."""
    return _chat_completion_with_model_fallback(
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal assistant that explains contracts in plain English "
                    "for people without a legal background. Be concise and avoid jargon."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Summarize this {doc_type} in plain English using 4-6 bullet points. "
                    "Cover: (1) parties involved, (2) key obligations of each party, "
                    "(3) payment terms or duration if present, (4) termination conditions, "
                    "(5) any unusual or one-sided terms worth noting.\n\n"
                    f"Contract:\n\n{text[:_MAX_CHARS_SUMMARY]}"
                ),
            },
        ],
    )


def summarize_contract_direct(text: str) -> tuple[str, str]:
    """Fallback path: ask for the document type and bullet summary in one direct prompt."""
    return _chat_completion_with_model_fallback(
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal assistant. Read the contract excerpt and respond with:"
                    "\nDocument Type: <3-6 word label>"
                    "\n- 4 to 6 short plain-English bullet points for a non-lawyer."
                    "\nDo not add any extra sections."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize this contract in plain English. Cover the parties, core obligations, "
                    "payment or duration if present, termination conditions, and unusual or one-sided terms.\n\n"
                    f"Contract:\n\n{text[:_MAX_CHARS_SUMMARY]}"
                ),
            },
        ],
    )


def parse_summary_bullets(summary: str) -> list[str]:
    """Normalize common markdown/numbered list output into plain bullet text."""
    summary = summary.replace("\u2022", "- ")
    bullets: list[str] = []
    current: list[str] = []

    for raw_line in summary.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                bullets.append(" ".join(current).strip())
                current = []
            continue

        if _SUMMARY_BULLET_RE.match(line):
            if current:
                bullets.append(" ".join(current).strip())
            current = [_SUMMARY_BULLET_RE.sub("", line).strip()]
            continue

        if current:
            current.append(line)
        else:
            current = [line]

    if current:
        bullets.append(" ".join(current).strip())

    cleaned = [b.strip(" -*") for b in bullets if b.strip(" -*")]
    if cleaned:
        return cleaned[:6]

    fallback = [
        part.strip(" -*")
        for part in re.split(r"(?<=[.!?])\s+", summary.strip())
        if part.strip(" -*")
    ]
    return fallback[:6]


def parse_direct_summary_output(summary: str) -> tuple[str, list[str]]:
    doc_type = "Unknown Document Type"
    bullet_lines: list[str] = []

    for raw_line in summary.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("document type:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate:
                doc_type = candidate
            continue
        bullet_lines.append(line)

    bullets = parse_summary_bullets("\n".join(bullet_lines))
    return doc_type, bullets


def build_contract_summary(text: str) -> dict:
    """Return a UI-friendly OpenAI summary payload without breaking classification."""
    if not _api_key():
        return summary_unavailable_payload(
            "Set OPENAI_API_KEY to enable the AI contract summary."
        )

    if not _has_openai_package():
        return summary_unavailable_payload(
            "Install openai>=1.30 to enable the AI contract summary."
        )

    try:
        doc_type, model_name = classify_document_type(text)
        summary_text, summary_model = summarize_contract(text, doc_type)
        bullets = parse_summary_bullets(summary_text)
        effective_model = summary_model or model_name
    except Exception:
        logger.exception("Two-step OpenAI contract summary failed; trying direct fallback")
        try:
            fallback_text, effective_model = summarize_contract_direct(text)
            doc_type, bullets = parse_direct_summary_output(fallback_text)
        except Exception:
            logger.exception("OpenAI contract summary failed")
            return summary_unavailable_payload(
                "The contract was classified, but the AI summary could not be generated.",
                status="error",
            )

    if not bullets:
        return summary_unavailable_payload(
            "The AI summary request returned no usable content.",
            status="error",
        )

    return {
        "available": True,
        "status": "ready",
        "message": "",
        "doc_type": doc_type,
        "bullets": bullets,
        "model": effective_model,
    }
