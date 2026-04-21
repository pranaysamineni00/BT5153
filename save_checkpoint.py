"""Export the trained TF-IDF + LR model artifacts to checkpoints/tfidf_lr_artifacts.joblib.

Run this once after Section 3 of the notebook has trained the TF-IDF model:

    python save_checkpoint.py

Or paste the body into a notebook cell and run it there (e.g., on Colab after
tfidf_artifacts is available in memory).
"""

from __future__ import annotations

from pathlib import Path

import joblib

# ── Replace this with however tfidf_artifacts is named in your notebook ────────
# In the notebook, Section 3 produces: tfidf_artifacts = train_tfidf_lr(...)
# Make sure that variable is in scope before running this script.

try:
    artifacts = tfidf_artifacts  # noqa: F821  (notebook variable)
except NameError:
    raise RuntimeError(
        "Variable 'tfidf_artifacts' not found. "
        "Run this script inside the notebook after training, or assign the variable manually:\n"
        "    artifacts = tfidf_artifacts"
    )

out_dir = Path(__file__).parent / "checkpoints"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "tfidf_lr_artifacts.joblib"

joblib.dump(
    {
        "model": artifacts.model,
        "best_threshold": artifacts.best_threshold,
        "id_to_clause": artifacts.id_to_clause,
    },
    out_path,
)

print(f"Saved to {out_path}")
print(f"  Clauses: {len(artifacts.id_to_clause)}")
print(f"  Threshold: {artifacts.best_threshold:.4f}")
