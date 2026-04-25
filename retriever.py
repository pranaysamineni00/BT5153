"""Retrieval helpers for the second-stage RAG review layer."""
from __future__ import annotations

import random
from typing import Any

import numpy as np

from config import ReviewConfig, expand_label_alias, get_review_config, normalize_label
from evaluation import probabilities_from_logits, tune_per_clause_thresholds
from rag_index import RagExample, RagIndex, build_rag_index

_DEFAULT_RETRIEVER: "RagRetriever | None" = None


def build_confusion_map_from_validation(
    logits: np.ndarray,
    labels: np.ndarray,
    id_to_clause: dict[int, str],
    thresholds: dict[str, float] | None = None,
    top_k: int = 3,
) -> dict[str, list[str]]:
    """Derive likely confusions from validation predictions."""
    if logits.size == 0 or labels.size == 0:
        return {}

    thresholds = thresholds or tune_per_clause_thresholds(logits, labels, id_to_clause)
    probs = probabilities_from_logits(logits)
    preds = np.zeros_like(probs, dtype=int)
    for idx, clause_name in id_to_clause.items():
        preds[:, idx] = (probs[:, idx] >= float(thresholds.get(clause_name, 0.5))).astype(int)

    confusion_scores: dict[str, dict[str, float]] = {
        clause: {} for clause in id_to_clause.values()
    }

    for row_idx in range(labels.shape[0]):
        true_indices = np.where(labels[row_idx].astype(int) == 1)[0].tolist()
        pred_indices = np.where(preds[row_idx] == 1)[0].tolist()

        for pred_idx in pred_indices:
            pred_label = id_to_clause[pred_idx]
            if pred_idx not in true_indices:
                for true_idx in true_indices:
                    true_label = id_to_clause[true_idx]
                    confusion_scores[pred_label][true_label] = confusion_scores[pred_label].get(true_label, 0.0) + 1.0

        for true_idx in true_indices:
            true_label = id_to_clause[true_idx]
            for pred_idx in pred_indices:
                if pred_idx == true_idx:
                    continue
                pred_label = id_to_clause[pred_idx]
                if labels[row_idx, pred_idx] == 0:
                    confusion_scores[true_label][pred_label] = confusion_scores[true_label].get(pred_label, 0.0) + 1.0

    return {
        label: [
            other_label
            for other_label, _ in sorted(other_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ]
        for label, other_scores in confusion_scores.items()
        if other_scores
    }


class RagRetriever:
    """Query positive examples and hard negatives from the train-split RAG index."""

    def __init__(self, rag_index: RagIndex, config: ReviewConfig | None = None) -> None:
        self.rag_index = rag_index
        self.config = config or get_review_config()
        self.known_labels = set(self.rag_index.entry_ids_by_label)

    def get_definition(self, label: str) -> str:
        canonical = normalize_label(label, known_labels=self.known_labels)
        return self.rag_index.label_definitions.get(canonical, "")

    def get_hard_negative_labels(
        self,
        label: str,
        model_top_labels: list[str] | None = None,
    ) -> list[str]:
        """Choose hard-negative labels using manual maps, model hints, confusions, or fallback."""
        canonical = normalize_label(label, known_labels=self.known_labels)

        manual = []
        raw_manual = self.rag_index.hard_negative_entry_ids_by_label.get(canonical)
        if raw_manual:
            manual_labels = []
            for other_label, entry_ids in self.rag_index.entry_ids_by_label.items():
                if any(item in set(raw_manual) for item in entry_ids):
                    manual_labels.append(other_label)
            manual = manual_labels
        if manual:
            return _dedupe_preserve([item for item in manual if item != canonical])[:3]

        if model_top_labels:
            model_candidates: list[str] = []
            for item in model_top_labels:
                model_candidates.extend(expand_label_alias(item, known_labels=self.known_labels))
            model_candidates = [item for item in model_candidates if item != canonical]
            if model_candidates:
                return _dedupe_preserve(model_candidates)[:3]

        confusion_candidates = list(self.rag_index.confusion_map.get(canonical, []))
        if confusion_candidates:
            confusion_candidates = [
                normalize_label(item, known_labels=self.known_labels)
                for item in confusion_candidates
                if normalize_label(item, known_labels=self.known_labels) != canonical
            ]
            confusion_candidates = _dedupe_preserve(confusion_candidates)
            if confusion_candidates:
                return confusion_candidates[:3]

        semantic = [
            item
            for item in self.rag_index.label_neighbor_map.get(canonical, [])
            if item != canonical
        ]
        if semantic:
            return semantic[:3]

        fallback_pool = [item for item in sorted(self.known_labels) if item != canonical]
        rng = random.Random(self.config.split_seed + len(canonical))
        rng.shuffle(fallback_pool)
        return fallback_pool[:3]

    def retrieve_positive_examples(
        self,
        label: str,
        query_snippet: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        canonical = normalize_label(label, known_labels=self.known_labels)
        indices = self.rag_index.entry_ids_by_label.get(canonical, [])
        matches = self.rag_index.search_entries(query_snippet, indices, top_k)
        return [_example_to_dict(item) for item in matches]

    def retrieve_hard_negatives(
        self,
        label: str,
        query_snippet: str,
        top_k: int = 2,
        model_top_labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        candidate_labels = self.get_hard_negative_labels(label, model_top_labels=model_top_labels)
        candidate_indices: list[int] = []
        for other_label in candidate_labels:
            candidate_indices.extend(self.rag_index.entry_ids_by_label.get(other_label, []))
        matches = self.rag_index.search_entries(query_snippet, candidate_indices, top_k)
        return [_example_to_dict(item) for item in matches]


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _example_to_dict(example: RagExample) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "clause_label": example.clause_label,
        "clause_definition": example.clause_definition,
        "source_contract_id": example.source_contract_id,
        "snippet_text": example.snippet_text,
        "answer_text": example.answer_text,
        "question": example.question,
    }


def get_default_retriever(
    *,
    config: ReviewConfig | None = None,
    confusion_map: dict[str, list[str]] | None = None,
    force_rebuild: bool = False,
) -> RagRetriever:
    global _DEFAULT_RETRIEVER

    config = config or get_review_config()
    if _DEFAULT_RETRIEVER is not None and not force_rebuild and confusion_map is None:
        return _DEFAULT_RETRIEVER

    index = build_rag_index(config=config, confusion_map=confusion_map, force_rebuild=force_rebuild)
    _DEFAULT_RETRIEVER = RagRetriever(index, config=config)
    return _DEFAULT_RETRIEVER


def retrieve_positive_examples(
    label: str,
    query_snippet: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    return get_default_retriever().retrieve_positive_examples(label, query_snippet, top_k=top_k)


def retrieve_hard_negatives(
    label: str,
    query_snippet: str,
    top_k: int = 2,
    model_top_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    return get_default_retriever().retrieve_hard_negatives(
        label,
        query_snippet,
        top_k=top_k,
        model_top_labels=model_top_labels,
    )


def get_hard_negative_labels(
    label: str,
    model_top_labels: list[str] | None = None,
) -> list[str]:
    return get_default_retriever().get_hard_negative_labels(
        label,
        model_top_labels=model_top_labels,
    )
