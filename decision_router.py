"""Deterministic routing rules for second-stage legal clause review."""
from __future__ import annotations

from typing import Any


def route_review_decision(
    *,
    original_model_label: str,
    model_confidence: float,
    presence: str | None,
    supports_label: str | None,
    better_label_if_any: str | None,
    reason: str,
) -> dict[str, Any]:
    """Apply the required deterministic routing rules to a reviewed clause."""
    presence = (presence or "UNCERTAIN").upper()
    supports_label = (supports_label or "UNCERTAIN").upper()
    better_label = better_label_if_any or None

    if model_confidence >= 0.80 and supports_label == "YES":
        return {
            "final_decision": "ACCEPT",
            "final_label": original_model_label,
            "reason": reason,
        }

    if supports_label == "NO" and better_label:
        return {
            "final_decision": "RERANK_LABEL",
            "final_label": better_label,
            "reason": reason,
        }

    if presence == "ABSENT" or supports_label == "NO":
        return {
            "final_decision": "REJECT",
            "final_label": None,
            "reason": reason,
        }

    if presence == "UNCERTAIN" or supports_label == "UNCERTAIN":
        return {
            "final_decision": "HUMAN_REVIEW",
            "final_label": original_model_label,
            "reason": reason,
        }

    return {
        "final_decision": "HUMAN_REVIEW",
        "final_label": original_model_label,
        "reason": reason,
    }
