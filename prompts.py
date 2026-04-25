"""Prompt builders for the legal clause review agents."""
from __future__ import annotations

import json
from typing import Any


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": candidate.get("candidate_label"),
        "model_confidence": candidate.get("model_confidence"),
        "definition": candidate.get("clause_definition"),
        "positive_examples": [
            {
                "source_contract_id": item.get("source_contract_id"),
                "snippet_text": item.get("snippet_text"),
            }
            for item in candidate.get("retrieved_positives", [])
        ],
        "hard_negative_examples": [
            {
                "label": item.get("clause_label"),
                "source_contract_id": item.get("source_contract_id"),
                "snippet_text": item.get("snippet_text"),
            }
            for item in candidate.get("retrieved_hard_negatives", [])
        ],
    }


def presence_system_prompt() -> str:
    return (
        "You are a legal review assistant. Decide only whether the TARGET clause is present "
        "in the provided contract snippet. Use the clause definition and retrieved evidence. "
        "Do not invent labels or facts. Respond with JSON only using exactly this schema: "
        '{"presence":"PRESENT|ABSENT|UNCERTAIN","reason":"...","confidence":0.0}'
    )


def presence_user_prompt(review_packet: dict[str, Any], target_label: str) -> str:
    target = next(
        (item for item in review_packet.get("candidate_reviews", []) if item.get("candidate_label") == target_label),
        {},
    )
    payload = {
        "target_label": target_label,
        "snippet": review_packet.get("snippet", ""),
        "candidate_labels": review_packet.get("candidate_labels", []),
        "target_evidence": _compact_candidate(target),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def evidence_system_prompt() -> str:
    return (
        "You are a legal review assistant. Check whether the provided snippet actually supports "
        "the TARGET clause label. Use the clause definition, retrieved positive examples, hard negative "
        "examples, and the alternative candidate labels. Only choose better_label_if_any from the provided "
        "candidate labels. If unsure, return UNCERTAIN. Respond with JSON only using exactly this schema: "
        '{"supports_label":"YES|NO|UNCERTAIN","better_label_if_any":null,"reason":"...","confidence":0.0}'
    )


def evidence_user_prompt(review_packet: dict[str, Any], target_label: str) -> str:
    compact = [_compact_candidate(item) for item in review_packet.get("candidate_reviews", [])]
    payload = {
        "target_label": target_label,
        "snippet": review_packet.get("snippet", ""),
        "candidate_labels": review_packet.get("candidate_labels", []),
        "candidate_evidence": compact,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def strict_retry_message(schema_name: str) -> str:
    return (
        f"Your previous {schema_name} response was not valid JSON. Reply again with JSON only, "
        "no markdown, no commentary, no code fences, and no extra keys."
    )
