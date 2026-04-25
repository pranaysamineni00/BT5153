"""Orchestration for the RAG-based second-stage legal clause review pipeline."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from agents import CandidateRetrieverAgent, ClausePresenceAgent, EvidenceSupportAgent
from config import ReviewConfig, get_review_config, review_llm_available
from decision_router import route_review_decision
from retriever import RagRetriever, build_confusion_map_from_validation, get_default_retriever


class SecondStageReviewPipeline:
    """Run retrieval, LLM review, and deterministic routing on classifier outputs."""

    def __init__(
        self,
        classifier: Any,
        *,
        config: ReviewConfig | None = None,
        retriever: RagRetriever | None = None,
    ) -> None:
        self.classifier = classifier
        self.config = config or get_review_config()
        self.retriever = retriever
        self.candidate_retriever = (
            CandidateRetrieverAgent(retriever, config=self.config)
            if retriever is not None
            else None
        )
        self.presence_agent = ClausePresenceAgent(config=self.config)
        self.support_agent = EvidenceSupportAgent(config=self.config)

    def _get_retriever(self) -> RagRetriever:
        if self.retriever is None:
            self.retriever = get_default_retriever(
                config=self.config,
                confusion_map=self._classifier_confusion_map(),
            )
        if self.candidate_retriever is None:
            self.candidate_retriever = CandidateRetrieverAgent(self.retriever, config=self.config)
        return self.retriever

    def _get_candidate_retriever(self) -> CandidateRetrieverAgent:
        if self.candidate_retriever is None:
            self.candidate_retriever = CandidateRetrieverAgent(self._get_retriever(), config=self.config)
        return self.candidate_retriever

    def _classifier_confusion_map(self) -> dict[str, list[str]]:
        logits = getattr(self.classifier, "validation_logits", None)
        labels = getattr(self.classifier, "validation_labels", None)
        id_to_clause = getattr(self.classifier, "id_to_clause", None)
        thresholds = getattr(self.classifier, "thresholds", None)

        if logits is None or labels is None or not id_to_clause:
            return {}

        try:
            return build_confusion_map_from_validation(logits, labels, id_to_clause, thresholds=thresholds)
        except Exception:
            return {}

    def review_contract(
        self,
        text: str,
        classification_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Review every detected snippet from the supervised classifier."""
        base_result = classification_result or self.classifier.classify(text)
        clauses = list(base_result.get("clauses", []))

        if not self.config.enable_second_stage_review:
            return {
                "review_status": "DISABLED",
                "items": [],
                "decision_counts": _decision_counts([]),
            }

        reviews = [self.review_prediction(item) for item in clauses]
        return {
            "review_status": _overall_review_status(reviews),
            "items": reviews,
            "decision_counts": _decision_counts(reviews),
            "review_model": self.config.review_model if review_llm_available() else "",
            "rag_source": "CUAD train split only",
        }

    def review_prediction(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """Review a single detected clause/snippet."""
        original_label = prediction.get("clause", "")
        snippet = prediction.get("snippet", "") or ""
        model_confidence = float(prediction.get("confidence", 0.0))

        candidate_labels = self.classifier.get_top_candidate_labels(
            snippet,
            top_k=self.config.model_top_k,
            ensure_labels=[original_label],
        )
        if not review_llm_available():
            return _llm_skipped_output(prediction, candidate_labels)

        # Stage A: retrieve label definitions, positives, and hard negatives.
        stage_a = self._get_candidate_retriever().run(
            snippet=snippet,
            original_model_label=original_label,
            original_model_confidence=model_confidence,
            candidate_labels=candidate_labels,
        )
        primary_candidate = _find_candidate_packet(stage_a, original_label)

        # Stage B/C: ask the LLM whether the clause is present and whether the snippet supports it.
        presence_result = self.presence_agent.run(stage_a, original_label)
        support_result = self.support_agent.run(stage_a, original_label)

        if not presence_result.valid_json or not support_result.valid_json:
            return _json_failure_output(prediction, stage_a, primary_candidate)

        presence_payload = presence_result.payload or {}
        support_payload = support_result.payload or {}
        # Final routing stays deterministic so the review layer remains inspectable.
        route = route_review_decision(
            original_model_label=original_label,
            model_confidence=model_confidence,
            presence=presence_payload.get("presence"),
            supports_label=support_payload.get("supports_label"),
            better_label_if_any=_validated_better_label(
                support_payload.get("better_label_if_any"),
                stage_a.get("candidate_labels", []),
                original_label,
            ),
            reason=_merge_reasons(
                presence_payload.get("reason", ""),
                support_payload.get("reason", ""),
            ),
        )

        return {
            "snippet": snippet,
            "original_model_label": original_label,
            "model_confidence": round(model_confidence, 3),
            "presence": (presence_payload.get("presence") or "UNCERTAIN").upper(),
            "supports_label": (support_payload.get("supports_label") or "UNCERTAIN").upper(),
            "final_decision": route["final_decision"],
            "final_label": route["final_label"],
            "reason": route["reason"],
            "retrieved_evidence": {
                "positive_examples": primary_candidate.get("retrieved_positives", []),
                "hard_negative_examples": primary_candidate.get("retrieved_hard_negatives", []),
            },
            "candidate_labels": stage_a.get("candidate_labels", []),
            "review_status": "READY",
            "better_label_if_any": _validated_better_label(
                support_payload.get("better_label_if_any"),
                stage_a.get("candidate_labels", []),
                original_label,
            ),
            "presence_confidence": float(presence_payload.get("confidence", 0.0) or 0.0),
            "support_confidence": float(support_payload.get("confidence", 0.0) or 0.0),
        }


def review_contract_predictions(
    text: str,
    classifier: Any,
    classification_result: dict[str, Any] | None = None,
    *,
    config: ReviewConfig | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for the default second-stage review flow."""
    pipeline = SecondStageReviewPipeline(classifier, config=config)
    return pipeline.review_contract(text, classification_result=classification_result)


def compare_baseline_vs_review(
    review_outputs: list[dict[str, Any]],
    true_labels: list[str | list[str] | set[str]],
    label_space: list[str] | None = None,
) -> dict[str, Any]:
    """Compare supervised baseline labels against reviewed final decisions."""
    if len(review_outputs) != len(true_labels):
        raise ValueError("review_outputs and true_labels must have the same length")

    normalized_true = [_normalize_truth(item) for item in true_labels]
    baseline_preds = [[item["original_model_label"]] if item.get("original_model_label") else [] for item in review_outputs]
    reviewed_preds = [
        [item["final_label"]]
        if item.get("final_decision") in {"ACCEPT", "RERANK_LABEL"} and item.get("final_label")
        else []
        for item in review_outputs
    ]

    if label_space is None:
        labels = set()
        for group in normalized_true + baseline_preds + reviewed_preds:
            labels.update(group)
        label_space = sorted(labels)

    baseline_metrics = _compute_label_metrics(normalized_true, baseline_preds, label_space)
    reviewed_metrics = _compute_label_metrics(normalized_true, reviewed_preds, label_space)
    baseline_fp = _count_false_positives(normalized_true, baseline_preds)
    reviewed_fp = _count_false_positives(normalized_true, reviewed_preds)

    return {
        "baseline": baseline_metrics,
        "reviewed": reviewed_metrics,
        "decision_counts": _decision_counts(review_outputs),
        "false_positive_reduction": baseline_fp - reviewed_fp,
    }


def _compute_label_metrics(
    true_labels: list[list[str]],
    predicted_labels: list[list[str]],
    label_space: list[str],
) -> dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    if not label_space:
        return {"precision": 0.0, "recall": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}

    y_true = np.zeros((len(true_labels), len(label_space)), dtype=int)
    y_pred = np.zeros((len(predicted_labels), len(label_space)), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(label_space)}

    for row_idx, labels in enumerate(true_labels):
        for label in labels:
            if label in label_to_idx:
                y_true[row_idx, label_to_idx[label]] = 1

    for row_idx, labels in enumerate(predicted_labels):
        for label in labels:
            if label in label_to_idx:
                y_pred[row_idx, label_to_idx[label]] = 1

    return {
        "precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def _count_false_positives(true_labels: list[list[str]], predicted_labels: list[list[str]]) -> int:
    total = 0
    for truth, preds in zip(true_labels, predicted_labels):
        truth_set = set(truth)
        total += sum(1 for label in preds if label not in truth_set)
    return total


def _normalize_truth(item: str | list[str] | set[str]) -> list[str]:
    if isinstance(item, str):
        return [item] if item else []
    if isinstance(item, set):
        return sorted(item)
    return [label for label in item if label]


def _find_candidate_packet(stage_a: dict[str, Any], label: str) -> dict[str, Any]:
    for item in stage_a.get("candidate_reviews", []):
        if item.get("candidate_label") == label:
            return item
    return {}


def _merge_reasons(presence_reason: str, support_reason: str) -> str:
    parts = [part.strip() for part in [presence_reason, support_reason] if part and part.strip()]
    return " ".join(parts).strip()


def _validated_better_label(
    better_label: Any,
    candidate_labels: list[str],
    original_model_label: str,
) -> str | None:
    if not better_label or better_label == original_model_label:
        return None
    return better_label if better_label in set(candidate_labels) else None


def _llm_skipped_output(
    prediction: dict[str, Any],
    candidate_labels: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "snippet": prediction.get("snippet", ""),
        "original_model_label": prediction.get("clause"),
        "model_confidence": round(float(prediction.get("confidence", 0.0)), 3),
        "presence": "UNCERTAIN",
        "supports_label": "UNCERTAIN",
        "final_decision": "ACCEPT",
        "final_label": prediction.get("clause"),
        "reason": "LLM review skipped because OPENAI_API_KEY or the OpenAI package is unavailable.",
        "retrieved_evidence": {
            "positive_examples": [],
            "hard_negative_examples": [],
        },
        "candidate_labels": [item["label"] for item in candidate_labels],
        "review_status": "LLM_SKIPPED",
    }


def _json_failure_output(
    prediction: dict[str, Any],
    stage_a: dict[str, Any],
    primary_candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "snippet": prediction.get("snippet", ""),
        "original_model_label": prediction.get("clause"),
        "model_confidence": round(float(prediction.get("confidence", 0.0)), 3),
        "presence": "UNCERTAIN",
        "supports_label": "UNCERTAIN",
        "final_decision": "HUMAN_REVIEW",
        "final_label": prediction.get("clause"),
        "reason": "LLM review returned invalid JSON twice; escalated to human review.",
        "retrieved_evidence": {
            "positive_examples": primary_candidate.get("retrieved_positives", []),
            "hard_negative_examples": primary_candidate.get("retrieved_hard_negatives", []),
        },
        "candidate_labels": stage_a.get("candidate_labels", []),
        "review_status": "HUMAN_REVIEW",
    }


def _decision_counts(review_outputs: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(item.get("final_decision", "") for item in review_outputs if item.get("final_decision"))
    return {
        "ACCEPT": counts.get("ACCEPT", 0),
        "REJECT": counts.get("REJECT", 0),
        "RERANK_LABEL": counts.get("RERANK_LABEL", 0),
        "HUMAN_REVIEW": counts.get("HUMAN_REVIEW", 0),
    }


def _overall_review_status(review_outputs: list[dict[str, Any]]) -> str:
    statuses = {item.get("review_status", "") for item in review_outputs}
    if not review_outputs:
        return "READY"
    if statuses == {"READY"}:
        return "READY"
    if "HUMAN_REVIEW" in statuses:
        return "PARTIAL"
    if "LLM_SKIPPED" in statuses:
        return "LLM_SKIPPED"
    return "PARTIAL"
