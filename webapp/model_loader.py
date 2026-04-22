from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, Union

import numpy as np
import streamlit as st

CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "tfidf_lr_artifacts.joblib"


# best_threshold is either a single float (global) or a dict {clause_name: float}
ThresholdT = Union[float, dict]


class ClassifierArtifacts(NamedTuple):
    # For sklearn path `model` is the TF-IDF pipeline with .predict_proba(texts).
    # For Legal-BERT path `model` is a HuggingFace model and `tokenizer` is set.
    model: Any
    best_threshold: ThresholdT
    id_to_clause: dict
    # Populated only for the Legal-BERT (V8) path — None for sklearn (V1-V7).
    tokenizer: Any = None
    backend: str = "sklearn"
    max_length: int = 512
    stride: int = 128


@st.cache_resource(show_spinner="Loading clause classifier…")
def load_classifier() -> ClassifierArtifacts:
    if not CHECKPOINT_PATH.exists():
        st.error(
            f"Checkpoint not found at `{CHECKPOINT_PATH}`. "
            "Run `save_checkpoint.py` (or the notebook export cell) first."
        )
        st.stop()

    import joblib
    data = joblib.load(CHECKPOINT_PATH)

    # V8 Legal-BERT artifact: has model_state_dict + tokenizer_name keys
    if "model_state_dict" in data:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer_name = data["tokenizer_name"]
        num_labels = int(data["num_labels"])
        id_to_clause = {int(k): v for k, v in data["id_to_clause"].items()}

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            tokenizer_name,
            num_labels=num_labels,
            id2label=id_to_clause,
            label2id={v: k for k, v in id_to_clause.items()},
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
        model.load_state_dict(data["model_state_dict"])
        model.eval()

        thr = data["best_threshold"]
        if not isinstance(thr, dict):
            thr = float(thr)

        return ClassifierArtifacts(
            model=model,
            best_threshold=thr,
            id_to_clause=id_to_clause,
            tokenizer=tokenizer,
            backend="legal_bert",
            max_length=int(data.get("max_length", 512)),
            stride=int(data.get("stride", 128)),
        )

    # V1-V7 sklearn pipeline
    thr = data["best_threshold"]
    if not isinstance(thr, dict):
        thr = float(thr)

    return ClassifierArtifacts(
        model=data["model"],
        best_threshold=thr,
        id_to_clause=data["id_to_clause"],
        tokenizer=None,
        backend="sklearn",
    )


def _predict_proba_legal_bert(text: str, artifacts: ClassifierArtifacts) -> np.ndarray:
    """Run Legal-BERT chunked inference and max-pool to contract-level probabilities.

    Returns a 1-D array of shape (num_labels,).
    """
    import torch

    tokenizer = artifacts.tokenizer
    model = artifacts.model

    chunks = tokenizer(
        text,
        max_length=artifacts.max_length,
        stride=artifacts.stride,
        truncation=True,
        padding="max_length",
        return_overflowing_tokens=True,
        return_tensors="pt",
    )

    input_ids = chunks["input_ids"]
    attention_mask = chunks["attention_mask"]
    token_type_ids = chunks.get("token_type_ids")

    num_labels = len(artifacts.id_to_clause)
    if input_ids.shape[0] == 0:
        return np.zeros(num_labels, dtype=np.float32)

    # Batch in groups of 8 to keep CPU memory sane for long contracts
    batch = 8
    all_probs = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], batch):
            end = start + batch
            kwargs = {
                "input_ids": input_ids[start:end],
                "attention_mask": attention_mask[start:end],
            }
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids[start:end]
            logits = model(**kwargs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    chunk_probs = np.concatenate(all_probs, axis=0)  # [n_chunks, num_labels]
    # Max-pool across chunks → contract-level probability per clause
    return chunk_probs.max(axis=0)


def predict_clauses(text: str, artifacts: ClassifierArtifacts) -> list[str]:
    """Return list of flagged clause type names for a single contract text.

    Handles two backends:
      - "sklearn"     → TF-IDF pipeline .predict_proba(texts) (V1-V7)
      - "legal_bert"  → HuggingFace model, chunk + max-pool (V8)

    Thresholds:
      - float  → global log-odds threshold (sklearn V1-V4 only)
      - dict   → {clause_name: probability_threshold} per clause (V5+, V8)
    """
    if artifacts.backend == "legal_bert":
        proba = _predict_proba_legal_bert(text, artifacts)
    else:
        proba = artifacts.model.predict_proba([text])[0]  # shape (num_labels,)

    thr = artifacts.best_threshold
    flagged: list[str] = []

    if isinstance(thr, dict):
        # Per-clause probability thresholds (V5+, V8)
        for i, clause in artifacts.id_to_clause.items():
            clause_thr = thr.get(clause, 0.5)
            if proba[i] >= clause_thr:
                flagged.append(clause)
    else:
        # Global log-odds threshold (V1-V4 sklearn only)
        eps = 1e-7
        p = np.clip(proba, eps, 1 - eps)
        logits = np.log(p / (1 - p))
        flagged = [
            artifacts.id_to_clause[i]
            for i, logit in enumerate(logits)
            if logit >= thr
        ]

    return flagged
