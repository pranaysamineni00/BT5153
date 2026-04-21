from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import streamlit as st

CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "tfidf_lr_artifacts.joblib"


class ClassifierArtifacts(NamedTuple):
    model: object           # _TfIdfPipeline with .predict_proba(texts)
    best_threshold: float
    id_to_clause: dict[int, str]


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
    return ClassifierArtifacts(
        model=data["model"],
        best_threshold=float(data["best_threshold"]),
        id_to_clause=data["id_to_clause"],
    )


def predict_clauses(text: str, artifacts: ClassifierArtifacts) -> list[str]:
    """Return list of flagged clause type names for a single contract text."""
    proba = artifacts.model.predict_proba([text])          # shape (1, num_labels)
    scores = proba[0]
    eps = 1e-7
    p = np.clip(scores, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    flagged = [
        artifacts.id_to_clause[i]
        for i, logit in enumerate(logits)
        if logit >= artifacts.best_threshold
    ]
    return flagged
