"""Agent implementations for retrieval and LLM-based legal review."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from config import ReviewConfig, get_review_config, review_client, review_llm_available
from prompts import (
    evidence_system_prompt,
    evidence_user_prompt,
    presence_system_prompt,
    presence_user_prompt,
    strict_retry_message,
)
from retriever import RagRetriever


class CandidateRetrieverAgent:
    """Stage A: fetch clause definitions, positives, and hard negatives."""

    def __init__(self, retriever: RagRetriever, config: ReviewConfig | None = None) -> None:
        self.retriever = retriever
        self.config = config or get_review_config()

    def run(
        self,
        *,
        snippet: str,
        original_model_label: str,
        original_model_confidence: float,
        candidate_labels: list[dict[str, Any]],
    ) -> dict[str, Any]:
        candidate_reviews: list[dict[str, Any]] = []
        model_label_names = [item["label"] for item in candidate_labels]

        for candidate in candidate_labels:
            label = candidate["label"]
            candidate_reviews.append(
                {
                    "snippet": snippet,
                    "candidate_label": label,
                    "model_confidence": round(float(candidate.get("confidence", 0.0)), 3),
                    "clause_definition": self.retriever.get_definition(label),
                    "retrieved_positives": self.retriever.retrieve_positive_examples(
                        label,
                        snippet,
                        top_k=self.config.retriever_top_positives,
                    ),
                    "retrieved_hard_negatives": self.retriever.retrieve_hard_negatives(
                        label,
                        snippet,
                        top_k=self.config.retriever_top_negatives,
                        model_top_labels=model_label_names,
                    ),
                }
            )

        return {
            "snippet": snippet,
            "original_model_label": original_model_label,
            "original_model_confidence": round(float(original_model_confidence), 3),
            "candidate_labels": model_label_names,
            "candidate_reviews": candidate_reviews,
        }


@dataclass
class AgentResult:
    payload: dict[str, Any] | None
    valid_json: bool
    raw_response: str


class _JsonReviewAgent:
    """Shared helper for strict JSON-only legal review agents."""

    agent_name = "review"

    def __init__(self, config: ReviewConfig | None = None) -> None:
        self.config = config or get_review_config()

    def available(self) -> bool:
        return review_llm_available()

    def _messages(self, review_packet: dict[str, Any], target_label: str) -> list[dict[str, str]]:
        raise NotImplementedError

    def run(self, review_packet: dict[str, Any], target_label: str) -> AgentResult:
        if not self.available():
            return AgentResult(payload=None, valid_json=False, raw_response="")

        messages = self._messages(review_packet, target_label)
        return self._request_json(messages)

    def _request_json(self, messages: list[dict[str, str]]) -> AgentResult:
        client = review_client()
        raw_content = ""
        for attempt in range(2):
            response = client.chat.completions.create(
                model=self.config.review_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            raw_content = response.choices[0].message.content or ""
            parsed = _parse_json_object(raw_content)
            if parsed is not None:
                return AgentResult(payload=parsed, valid_json=True, raw_response=raw_content)

            if attempt == 0:
                messages = messages + [
                    {
                        "role": "user",
                        "content": strict_retry_message(self.agent_name),
                    }
                ]

        return AgentResult(payload=None, valid_json=False, raw_response=raw_content)


class ClausePresenceAgent(_JsonReviewAgent):
    """Stage B: determine whether the target clause is present."""

    agent_name = "presence"

    def _messages(self, review_packet: dict[str, Any], target_label: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": presence_system_prompt()},
            {"role": "user", "content": presence_user_prompt(review_packet, target_label)},
        ]


class EvidenceSupportAgent(_JsonReviewAgent):
    """Stage C: verify that the snippet supports the target label."""

    agent_name = "evidence"

    def _messages(self, review_packet: dict[str, Any], target_label: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": evidence_system_prompt()},
            {"role": "user", "content": evidence_user_prompt(review_packet, target_label)},
        ]


def _parse_json_object(text: str) -> dict[str, Any] | None:
    candidate = (text or "").strip()
    if not candidate:
        return None

    if candidate.startswith("```"):
        parts = [part for part in candidate.split("```") if part.strip()]
        if parts:
            candidate = parts[-1].strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None
