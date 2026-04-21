from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))


def precision_at_recall_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_target: float = 0.80,
) -> float:
    """Precision at the highest threshold that achieves recall >= recall_target.

    Returns 0.0 if the class has no positives or recall_target is unreachable.
    """
    if y_true.sum() == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = recall >= recall_target
    if not mask.any():
        return 0.0
    return float(precision[mask].max())


def tune_per_clause_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
    id_to_clause: dict[int, str],
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """Find the threshold per clause that maximises per-clause F1 on the given logits.

    Returns dict mapping clause_name -> best_threshold (between 0 and 1).
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    probs = _sigmoid(logits)
    best: dict[str, float] = {}

    for clause_id, clause_name in id_to_clause.items():
        y_true = labels[:, clause_id].astype(int)
        y_score = probs[:, clause_id]
        best_f1 = -1.0
        best_t = 0.5
        for t in thresholds:
            preds = (y_score >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        best[clause_name] = best_t
    return best


def compute_aggregate_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[int | str, float],
    id_to_clause: dict[int, str] | None = None,
) -> dict[str, float]:
    """Macro/micro F1, precision, recall using per-clause thresholds.

    thresholds keys may be clause_name (str) or clause_id (int).
    """
    probs = _sigmoid(logits)
    num_labels = logits.shape[1]
    preds = np.zeros_like(probs, dtype=int)

    for i in range(num_labels):
        key = id_to_clause[i] if (id_to_clause and i in id_to_clause) else i
        t = thresholds.get(key, 0.5)
        preds[:, i] = (probs[:, i] >= t).astype(int)

    int_labels = labels.astype(int)
    return {
        "macro_f1":        float(f1_score(int_labels, preds, average="macro",  zero_division=0)),
        "micro_f1":        float(f1_score(int_labels, preds, average="micro",  zero_division=0)),
        "micro_precision": float(precision_score(int_labels, preds, average="micro", zero_division=0)),
        "micro_recall":    float(recall_score(int_labels, preds, average="micro",    zero_division=0)),
    }


def compute_per_clause_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[str, float],
    id_to_clause: dict[int, str],
    n_positive_train: dict[int, int],
) -> pd.DataFrame:
    """Per-clause breakdown: precision, recall, F1, P@80R, AUPR, threshold, n_positive_train.

    Returns a DataFrame sorted by F1 descending with one row per clause type.
    """
    probs = _sigmoid(logits)
    rows = []
    for clause_id, clause_name in id_to_clause.items():
        y_true = labels[:, clause_id].astype(int)
        y_score = probs[:, clause_id]
        t = thresholds.get(clause_name, 0.5)
        preds = (y_score >= t).astype(int)

        rows.append({
            "clause_type":            clause_name,
            "threshold":              round(t, 2),
            "precision":              float(precision_score(y_true, preds, zero_division=0)),
            "recall":                 float(recall_score(y_true, preds, zero_division=0)),
            "f1":                     float(f1_score(y_true, preds, zero_division=0)),
            "precision_at_80_recall": precision_at_recall_threshold(y_true, y_score, 0.80),
            "aupr":                   float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else 0.0,
            "support_test":           int(y_true.sum()),
            "n_positive_train":       n_positive_train.get(clause_id, 0),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[str, float],
    id_to_clause: dict[int, str],
    title: str = "Predicted vs True Clause Co-occurrence",
    save_path: str | None = None,
) -> None:
    """Heatmap: for each true-positive clause (row), what rate is each clause predicted (col)?

    Rows are indexed by true positive samples for each clause. Values show prediction rates.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    probs = _sigmoid(logits)
    num_labels = logits.shape[1]
    clause_names = [id_to_clause[i] for i in range(num_labels)]
    preds = np.zeros_like(probs, dtype=int)
    for i in range(num_labels):
        t = thresholds.get(clause_names[i], 0.5)
        preds[:, i] = (probs[:, i] >= t).astype(int)

    int_labels = labels.astype(int)
    matrix = np.zeros((num_labels, num_labels), dtype=float)
    for true_idx in range(num_labels):
        true_positive_mask = int_labels[:, true_idx] == 1
        if true_positive_mask.sum() == 0:
            continue
        for pred_idx in range(num_labels):
            matrix[true_idx, pred_idx] = preds[true_positive_mask, pred_idx].mean()

    short_names = [n[:20] for n in clause_names]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        matrix,
        xticklabels=short_names,
        yticklabels=short_names,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Rate predicted positive"},
    )
    ax.set_xlabel("Predicted clause")
    ax.set_ylabel("True clause (positive samples only)")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_model_comparison(
    results: dict[str, pd.DataFrame],
    metric: str = "f1",
    save_path: str | None = None,
) -> None:
    """Grouped bar chart: all models x clause types for a given metric column."""
    if not results:
        raise ValueError("results dict is empty — nothing to plot")

    import matplotlib.pyplot as plt

    model_names = list(results.keys())
    combined = None
    for model_name, df in results.items():
        sub = df[["clause_type", metric]].rename(columns={metric: model_name})
        combined = sub if combined is None else combined.merge(sub, on="clause_type", how="outer")

    combined = combined.set_index("clause_type").fillna(0)
    combined.plot(kind="barh", figsize=(14, 10), colormap="tab10")
    plt.xlabel(metric.upper())
    plt.title(f"Per-Clause {metric.upper()} — All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
