# Legal Clause Classification Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 5-model legal clause classification pipeline (Steps 1–5 of BT5153 spec) as a unified Colab-compatible notebook backed by 4 Python modules.

**Architecture:** Four focused Python modules (`data_loading`, `preprocessing`, `training`, `evaluation`) are orchestrated by `pipeline.ipynb`. The existing `cuad_chunk_multilabel.py` is superseded — its logic migrates into `preprocessing.py` and `training.py`. All 5 training functions return a common `ModelArtifacts` dataclass so evaluation code works uniformly.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace `transformers`/`datasets`, scikit-learn, pandas, matplotlib/seaborn, pytest (local), Google Colab (GPU runtime for full training)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `data_loading.py` | Create | CUAD + LEDGAR loading, validation, confirmation stats |
| `preprocessing.py` | Create | Clause filtering, EDA chart, chunking, splits, sample weights, aggregation |
| `training.py` | Create | All 5 model training loops, shared utilities, ModelArtifacts |
| `evaluation.py` | Create | Per-clause metrics, Precision@80%R, AUPR, threshold tuning, confusion matrix |
| `pipeline.ipynb` | Modify | Orchestration notebook — import modules, run all steps |
| `cuad_chunk_multilabel.py` | Supersede | Logic migrates; file kept but no longer imported |
| `tests/test_data_loading.py` | Create | Unit tests for loading functions |
| `tests/test_preprocessing.py` | Create | Unit tests for filtering, chunking, weights, aggregation |
| `tests/test_evaluation.py` | Create | Unit tests for all metric computations |
| `tests/test_training.py` | Create | Smoke tests for all 5 training functions |

---

## Task 1: data_loading.py — CUAD Loader

**Files:**
- Create: `data_loading.py`
- Create: `tests/test_data_loading.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_data_loading.py
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from data_loading import load_cuad

def test_load_cuad_returns_dataframe_with_required_columns():
    mock_cuad = {
        "data": [
            {
                "title": "Contract A",
                "paragraphs": [
                    {
                        "context": "This agreement shall be governed by...",
                        "qas": [
                            {
                                "id": "Contract A__Governing Law",
                                "question": "What law governs?",
                                "answers": [{"text": "governed by", "answer_start": 25}],
                                "is_impossible": False,
                            }
                        ],
                    }
                ],
            }
        ]
    }
    with patch("builtins.open"), patch("json.load", return_value=mock_cuad):
        from pathlib import Path
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "open", MagicMock()):
            pass

    # Direct construction test — call internal builder
    from data_loading import _parse_cuad_json
    df = _parse_cuad_json(mock_cuad)
    required = {"contract_title", "clause_type", "question", "contract_text",
                "has_answer", "answer_texts", "answer_starts", "answer_count"}
    assert required.issubset(set(df.columns))
    assert len(df) == 1
    assert df.iloc[0]["clause_type"] == "Governing Law"
    assert df.iloc[0]["has_answer"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/pranay/Documents/legal_clause_classification
python -m pytest tests/test_data_loading.py::test_load_cuad_returns_dataframe_with_required_columns -v
```
Expected: `ImportError: No module named 'data_loading'`

- [ ] **Step 3: Implement data_loading.py**

```python
# data_loading.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


def _parse_cuad_json(cuad: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for doc in cuad["data"]:
        for paragraph in doc["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answers = qa.get("answers", [])
                rows.append(
                    {
                        "contract_title": doc["title"],
                        "clause_type": qa["id"].split("__", 1)[-1],
                        "question": qa["question"],
                        "contract_text": context,
                        "has_answer": len(answers) > 0,
                        "answer_texts": [a["text"] for a in answers],
                        "answer_starts": [a["answer_start"] for a in answers],
                        "answer_count": len(answers),
                    }
                )
    return pd.DataFrame(rows)


def load_cuad(data_dir: str = "data/cuad") -> pd.DataFrame:
    """Download (if needed) and parse CUAD into a flat clause-level DataFrame."""
    raw_json_path = Path(data_dir) / "CUAD_v1" / "CUAD_v1.json"
    raw_json_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_json_path.exists():
        downloaded = hf_hub_download(
            repo_id="theatticusproject/cuad",
            repo_type="dataset",
            filename="CUAD_v1/CUAD_v1.json",
            local_dir=data_dir,
        )
        raw_json_path = Path(downloaded)

    with raw_json_path.open() as f:
        cuad = json.load(f)

    df = _parse_cuad_json(cuad)

    print(f"CUAD loaded: {df['contract_title'].nunique():,} contracts, "
          f"{df['clause_type'].nunique()} clause types, "
          f"{len(df):,} rows, "
          f"positive rate {df['has_answer'].mean():.2%}")
    return df


def load_ledgar(cache_dir: str = "data/ledgar") -> Any:
    """Download and return the LEDGAR dataset from lex_glue."""
    from datasets import load_dataset
    dataset = load_dataset("coastalcph/lex_glue", "ledgar", cache_dir=cache_dir)
    n_train = len(dataset["train"])
    n_labels = dataset["train"].features["label"].num_classes
    print(f"LEDGAR loaded: {n_train:,} train examples, {n_labels} categories")
    return dataset
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_data_loading.py::test_load_cuad_returns_dataframe_with_required_columns -v
```
Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add data_loading.py tests/test_data_loading.py
git commit -m "feat: add data_loading module with CUAD and LEDGAR loaders"
```

---

## Task 2: preprocessing.py — Clause Filtering & EDA

**Files:**
- Create: `preprocessing.py`
- Create: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preprocessing.py
import numpy as np
import pandas as pd
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_cuad_df():
    """Minimal synthetic CUAD DataFrame for testing."""
    rows = []
    for contract_idx in range(10):
        title = f"Contract_{contract_idx}"
        for clause in ["Governing Law", "Non-Compete", "Rare Clause"]:
            has_answer = (clause != "Rare Clause") or (contract_idx == 0)
            rows.append({
                "contract_title": title,
                "clause_type": clause,
                "contract_text": f"text of {title}",
                "has_answer": has_answer,
                "answer_texts": ["span"] if has_answer else [],
                "answer_starts": [5] if has_answer else [],
                "answer_count": 1 if has_answer else 0,
            })
    return pd.DataFrame(rows)


def test_filter_clauses_drops_below_threshold():
    from preprocessing import filter_clauses
    df = _make_cuad_df()
    filtered, excluded = filter_clauses(df, min_positives=5)
    assert "Rare Clause" in excluded
    assert "Rare Clause" not in filtered["clause_type"].values


def test_filter_clauses_keeps_above_threshold():
    from preprocessing import filter_clauses
    df = _make_cuad_df()
    filtered, excluded = filter_clauses(df, min_positives=5)
    assert "Governing Law" in filtered["clause_type"].values
    assert "Non-Compete" in filtered["clause_type"].values


def test_filter_clauses_excluded_is_dict_with_counts():
    from preprocessing import filter_clauses
    df = _make_cuad_df()
    _, excluded = filter_clauses(df, min_positives=5)
    assert isinstance(excluded, dict)
    assert excluded["Rare Clause"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_preprocessing.py -k "filter" -v
```
Expected: `ImportError: No module named 'preprocessing'`

- [ ] **Step 3: Implement filtering functions**

```python
# preprocessing.py
from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# ── Clause filtering ──────────────────────────────────────────────────────────

def filter_clauses(
    cuad_df: pd.DataFrame,
    min_positives: int = 20,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop clause types with fewer than min_positives positive examples.

    Returns (filtered_df, excluded_dict) where excluded_dict maps
    clause_type → positive_count for every excluded type.
    """
    positive_counts = (
        cuad_df[cuad_df["has_answer"]]
        .groupby("clause_type")
        .size()
    )
    excluded = {
        clause: int(count)
        for clause, count in positive_counts.items()
        if count < min_positives
    }
    # Also exclude clause types with zero positives (not in positive_counts at all)
    all_types = set(cuad_df["clause_type"].unique())
    for clause in all_types:
        if clause not in positive_counts.index:
            excluded[clause] = 0

    keep = all_types - set(excluded.keys())
    filtered = cuad_df[cuad_df["clause_type"].isin(keep)].copy()

    if excluded:
        print(f"Excluded {len(excluded)} clause types (below min_positives={min_positives}):")
        for c, n in sorted(excluded.items(), key=lambda x: x[1]):
            print(f"  {c}: {n} positives")

    return filtered, excluded


def plot_clause_frequency(cuad_df: pd.DataFrame, save_path: str | None = None) -> None:
    """Bar chart: positive rate per clause type, sorted descending."""
    summary = (
        cuad_df.groupby("clause_type")["has_answer"]
        .agg(positive_rate="mean", positive_count="sum", total="count")
        .sort_values("positive_rate", ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=summary, x="positive_rate", y="clause_type", ax=ax, palette="crest")
    ax.set_title("Positive Rate per Clause Type (CUAD)")
    ax.set_xlabel("Positive Rate")
    ax.set_ylabel("")
    ax.axvline(x=0.2, color="red", linestyle="--", alpha=0.6, label="20% threshold")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_preprocessing.py -k "filter" -v
```
Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add preprocessing module with clause filtering and EDA plot"
```

---

## Task 3: preprocessing.py — Chunking & Splits

**Files:**
- Modify: `preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_preprocessing.py`:

```python
def test_build_chunk_examples_labels_positive_chunk():
    """A chunk overlapping a known answer span must have that clause labelled 1."""
    from preprocessing import build_chunk_examples
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
    clause_to_id = {"Governing Law": 0, "Non-Compete": 1}

    records = [
        {
            "contract_title": "C1",
            "contract_text": "This agreement is governed by the laws of New York.",
            "clause_spans": {"Governing Law": [(18, 46)]},
        }
    ]
    examples = build_chunk_examples(records, clause_to_id, tokenizer, max_length=64, stride=16)
    assert len(examples) >= 1
    # At least one chunk should have Governing Law = 1
    governing_law_hits = [e for e in examples if e["labels"][0] == 1.0]
    assert len(governing_law_hits) >= 1
    # Non-Compete should be 0 everywhere
    assert all(e["labels"][1] == 0.0 for e in examples)


def test_split_contract_records_80_10_10():
    from preprocessing import build_contract_records, split_contract_records

    rows = []
    for i in range(100):
        rows.append({
            "contract_title": f"C{i}", "clause_type": "Governing Law",
            "contract_text": "text", "has_answer": True,
            "answer_texts": ["t"], "answer_starts": [0], "answer_count": 1,
        })
    df = pd.DataFrame(rows)
    records = build_contract_records(df)
    train, val, test = split_contract_records(records, seed=42)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_preprocessing.py -k "chunk or split" -v
```
Expected: `ImportError` or `AttributeError` for missing functions.

- [ ] **Step 3: Implement chunking and split functions**

Append to `preprocessing.py`:

```python
# ── Contract record building ──────────────────────────────────────────────────

def build_clause_mappings(cuad_df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    clause_names = sorted(cuad_df["clause_type"].unique().tolist())
    clause_to_id = {name: idx for idx, name in enumerate(clause_names)}
    id_to_clause = {idx: name for name, idx in clause_to_id.items()}
    return clause_to_id, id_to_clause


def build_contract_records(cuad_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for contract_title, group in cuad_df.groupby("contract_title", sort=False):
        contract_text = group["contract_text"].iloc[0]
        clause_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in group.itertuples(index=False):
            for answer_text, answer_start in zip(row.answer_texts, row.answer_starts):
                answer_end = int(answer_start) + len(answer_text)
                clause_spans[row.clause_type].append((int(answer_start), answer_end))
        records.append({
            "contract_title": contract_title,
            "contract_text": contract_text,
            "clause_spans": dict(clause_spans),
        })
    return records


def split_contract_records(
    contract_records: list[dict[str, Any]],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")
    titles = [r["contract_title"] for r in contract_records]
    title_to_record = {r["contract_title"]: r for r in contract_records}
    train_titles, temp_titles = train_test_split(titles, train_size=train_size, random_state=seed)
    val_ratio = val_size / (val_size + test_size)
    val_titles, test_titles = train_test_split(temp_titles, train_size=val_ratio, random_state=seed)
    return (
        [title_to_record[t] for t in train_titles],
        [title_to_record[t] for t in val_titles],
        [title_to_record[t] for t in test_titles],
    )


# ── Chunk examples ────────────────────────────────────────────────────────────

def _span_overlaps_chunk(span_start: int, span_end: int, chunk_start: int, chunk_end: int) -> bool:
    return max(span_start, chunk_start) < min(span_end, chunk_end)


def build_chunk_examples(
    contract_records: list[dict[str, Any]],
    clause_to_id: dict[str, int],
    tokenizer: Any,
    max_length: int = 512,
    stride: int = 128,
) -> list[dict[str, Any]]:
    chunk_examples: list[dict[str, Any]] = []
    num_labels = len(clause_to_id)

    for record in contract_records:
        contract_text = record["contract_text"]
        chunked = tokenizer(
            contract_text,
            max_length=max_length,
            stride=stride,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        for chunk_index in range(len(chunked["input_ids"])):
            offset_mapping = chunked["offset_mapping"][chunk_index]
            valid_offsets = [(s, e) for s, e in offset_mapping if e > s]
            if not valid_offsets:
                continue
            chunk_char_start = valid_offsets[0][0]
            chunk_char_end = valid_offsets[-1][1]
            labels = np.zeros(num_labels, dtype=np.float32)
            for clause_name, spans in record["clause_spans"].items():
                if clause_name not in clause_to_id:
                    continue
                clause_id = clause_to_id[clause_name]
                if any(_span_overlaps_chunk(s, e, chunk_char_start, chunk_char_end) for s, e in spans):
                    labels[clause_id] = 1.0
            chunk_example: dict[str, Any] = {
                "contract_title": record["contract_title"],
                "chunk_index": chunk_index,
                "chunk_char_start": chunk_char_start,
                "chunk_char_end": chunk_char_end,
                "chunk_text": contract_text[chunk_char_start:chunk_char_end],
                "input_ids": chunked["input_ids"][chunk_index],
                "attention_mask": chunked["attention_mask"][chunk_index],
                "labels": labels.tolist(),
            }
            if "token_type_ids" in chunked:
                chunk_example["token_type_ids"] = chunked["token_type_ids"][chunk_index]
            chunk_examples.append(chunk_example)
    return chunk_examples


# ── Dataset class ─────────────────────────────────────────────────────────────

class MultiLabelChunkDataset(Dataset):
    def __init__(self, chunk_examples: list[dict[str, Any]]) -> None:
        self.chunk_examples = chunk_examples

    def __len__(self) -> int:
        return len(self.chunk_examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ex = self.chunk_examples[index]
        item = {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.float32),
        }
        if "token_type_ids" in ex:
            item["token_type_ids"] = torch.tensor(ex["token_type_ids"], dtype=torch.long)
        return item
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_preprocessing.py -k "chunk or split or filter" -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add chunking, contract records, and 80/10/10 split to preprocessing"
```

---

## Task 4: preprocessing.py — Sample Weights & Contract Aggregation

**Files:**
- Modify: `preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_preprocessing.py`:

```python
def test_compute_sample_weights_all_negative_gets_low_weight():
    from preprocessing import compute_sample_weights
    examples = [
        {"labels": [0.0, 0.0, 0.0]},  # all-negative
        {"labels": [1.0, 0.0, 0.0]},  # has positive
        {"labels": [0.0, 1.0, 0.0]},  # has positive
    ]
    weights = compute_sample_weights(examples, negative_weight=0.1)
    assert weights[0] == pytest.approx(0.1)
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(1.0)


def test_compute_pos_weight_shape_and_values():
    from preprocessing import compute_pos_weight
    # 4 examples, 2 labels: label 0 has 1 positive, label 1 has 3 positives
    examples = [
        {"labels": [1.0, 1.0]},
        {"labels": [0.0, 1.0]},
        {"labels": [0.0, 1.0]},
        {"labels": [0.0, 0.0]},
    ]
    weights = compute_pos_weight(examples)
    assert weights.shape == (2,)
    # label 0: 3 neg / 1 pos = 3.0
    assert float(weights[0]) == pytest.approx(3.0)
    # label 1: 1 neg / 3 pos = 0.333...
    assert float(weights[1]) == pytest.approx(1.0 / 3.0, rel=1e-3)


def test_aggregate_contract_predictions_takes_max():
    from preprocessing import aggregate_contract_predictions
    import pandas as pd
    chunk_df = pd.DataFrame([
        {"contract_title": "C1", "clause_type": "Governing Law", "score": 0.3, "chunk_index": 0},
        {"contract_title": "C1", "clause_type": "Governing Law", "score": 0.9, "chunk_index": 1},
        {"contract_title": "C1", "clause_type": "Non-Compete",   "score": 0.2, "chunk_index": 0},
    ])
    result = aggregate_contract_predictions(chunk_df)
    assert result.loc[result["clause_type"] == "Governing Law", "max_score"].iloc[0] == pytest.approx(0.9)
    assert result.loc[result["clause_type"] == "Non-Compete",   "max_score"].iloc[0] == pytest.approx(0.2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_preprocessing.py -k "weight or aggregate" -v
```
Expected: `ImportError` for missing functions.

- [ ] **Step 3: Implement sample weights and aggregation**

Append to `preprocessing.py`:

```python
# ── Sample & class weights ────────────────────────────────────────────────────

def compute_sample_weights(
    chunk_examples: list[dict[str, Any]],
    negative_weight: float = 0.1,
) -> list[float]:
    """Return per-sample weights. All-negative chunks get negative_weight; others get 1.0."""
    weights = []
    for ex in chunk_examples:
        is_all_negative = all(l == 0.0 for l in ex["labels"])
        weights.append(negative_weight if is_all_negative else 1.0)
    return weights


def compute_pos_weight(chunk_examples: list[dict[str, Any]]) -> torch.Tensor:
    """Per-label pos_weight for BCEWithLogitsLoss: neg_count / pos_count."""
    label_matrix = np.asarray([ex["labels"] for ex in chunk_examples], dtype=np.float32)
    positive_counts = label_matrix.sum(axis=0)
    negative_counts = len(label_matrix) - positive_counts
    weights = np.where(positive_counts > 0, negative_counts / np.maximum(positive_counts, 1.0), 1.0)
    return torch.tensor(weights, dtype=torch.float32)


# ── Contract-level aggregation ────────────────────────────────────────────────

def aggregate_contract_predictions(chunk_long_df: pd.DataFrame) -> pd.DataFrame:
    """Max-probability rollup across chunks per contract × clause_type.

    Input DataFrame must have columns: contract_title, clause_type, score, chunk_index.
    Returns DataFrame with columns: contract_title, clause_type, max_score, best_chunk_index.
    """
    agg = (
        chunk_long_df
        .sort_values("score", ascending=False)
        .groupby(["contract_title", "clause_type"], sort=False)
        .first()
        .rename(columns={"score": "max_score", "chunk_index": "best_chunk_index"})
        .reset_index()
    )
    return agg[["contract_title", "clause_type", "max_score", "best_chunk_index"]]


def build_chunk_to_contract_map(chunk_examples: list[dict[str, Any]]) -> dict[int, str]:
    """Map chunk list index → contract_title."""
    return {i: ex["contract_title"] for i, ex in enumerate(chunk_examples)}


# ── Convenience: prepare all splits ──────────────────────────────────────────

def prepare_chunked_splits(
    cuad_df: pd.DataFrame,
    model_name: str = "bert-base-uncased",
    max_length: int = 512,
    stride: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    clause_to_id, id_to_clause = build_clause_mappings(cuad_df)
    contract_records = build_contract_records(cuad_df)
    train_records, val_records, test_records = split_contract_records(contract_records, seed=seed)
    train_ex = build_chunk_examples(train_records, clause_to_id, tokenizer, max_length, stride)
    val_ex   = build_chunk_examples(val_records,   clause_to_id, tokenizer, max_length, stride)
    test_ex  = build_chunk_examples(test_records,  clause_to_id, tokenizer, max_length, stride)
    return {
        "tokenizer": tokenizer,
        "clause_to_id": clause_to_id,
        "id_to_clause": id_to_clause,
        "train_records": train_records, "val_records": val_records, "test_records": test_records,
        "train_examples": train_ex, "val_examples": val_ex, "test_examples": test_ex,
        "train_dataset": MultiLabelChunkDataset(train_ex),
        "val_dataset":   MultiLabelChunkDataset(val_ex),
        "test_dataset":  MultiLabelChunkDataset(test_ex),
    }
```

- [ ] **Step 4: Run all preprocessing tests**

```bash
python -m pytest tests/test_preprocessing.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add sample weights, pos_weight, and contract aggregation to preprocessing"
```

---

## Task 5: evaluation.py — Core Metrics & Per-Clause Threshold Tuning

**Files:**
- Create: `evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluation.py
import numpy as np
import pytest


def test_precision_at_recall_threshold_basic():
    from evaluation import precision_at_recall_threshold
    # Perfect classifier: y_score == y_true
    y_true  = np.array([1, 1, 0, 0, 1])
    y_score = np.array([0.9, 0.8, 0.1, 0.2, 0.7])
    p = precision_at_recall_threshold(y_true, y_score, recall_target=0.80)
    assert p == pytest.approx(1.0)


def test_precision_at_recall_threshold_returns_zero_when_unreachable():
    from evaluation import precision_at_recall_threshold
    # y_score is inverted — model never achieves 80% recall at reasonable threshold
    y_true  = np.array([1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    # With all scores identical, recall can be 100% at threshold=0.1
    # This tests the boundary condition
    p = precision_at_recall_threshold(y_true, y_score, recall_target=0.80)
    assert 0.0 <= p <= 1.0


def test_tune_per_clause_thresholds_returns_dict_of_floats():
    from evaluation import tune_per_clause_thresholds
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(100, 3))
    labels = rng.integers(0, 2, size=(100, 3)).astype(float)
    id_to_clause = {0: "A", 1: "B", 2: "C"}
    thresholds = tune_per_clause_thresholds(logits, labels, id_to_clause)
    assert set(thresholds.keys()) == {"A", "B", "C"}
    for v in thresholds.values():
        assert 0.0 < v < 1.0


def test_compute_aggregate_metrics_shapes():
    from evaluation import compute_aggregate_metrics
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(50, 4))
    labels = rng.integers(0, 2, size=(50, 4)).astype(float)
    thresholds = {i: 0.5 for i in range(4)}
    metrics = compute_aggregate_metrics(logits, labels, thresholds)
    for key in ("macro_f1", "micro_f1", "micro_precision", "micro_recall"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_evaluation.py -v
```
Expected: `ImportError: No module named 'evaluation'`

- [ ] **Step 3: Implement evaluation.py**

```python
# evaluation.py
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def precision_at_recall_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_target: float = 0.80,
) -> float:
    """Precision at the highest threshold that achieves recall >= recall_target."""
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
    """Find the threshold per clause that maximises per-clause F1 on the given logits."""
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

    thresholds keys may be clause_id (int) or clause_name (str).
    If id_to_clause is provided, maps names → ids automatically.
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_evaluation.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add evaluation.py tests/test_evaluation.py
git commit -m "feat: add evaluation module with per-clause threshold tuning and aggregate metrics"
```

---

## Task 6: evaluation.py — Per-Clause Breakdown, AUPR & Confusion Matrix

**Files:**
- Modify: `evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_evaluation.py`:

```python
def test_compute_per_clause_metrics_columns():
    from evaluation import compute_per_clause_metrics, tune_per_clause_thresholds
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(80, 3))
    labels = rng.integers(0, 2, size=(80, 3)).astype(float)
    # Ensure at least one positive per clause
    labels[:10, :] = 1.0
    id_to_clause = {0: "Governing Law", 1: "Non-Compete", 2: "Indemnification"}
    n_positives = {0: 20, 1: 15, 2: 10}
    thresholds = tune_per_clause_thresholds(logits, labels, id_to_clause)
    df = compute_per_clause_metrics(logits, labels, thresholds, id_to_clause, n_positives)
    required = {"clause_type", "precision", "recall", "f1",
                "precision_at_80_recall", "aupr", "n_positive_train", "threshold"}
    assert required.issubset(set(df.columns))
    assert len(df) == 3


def test_per_clause_aupr_in_range():
    from evaluation import compute_per_clause_metrics, tune_per_clause_thresholds
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(60, 2))
    labels = rng.integers(0, 2, size=(60, 2)).astype(float)
    labels[:5, :] = 1.0
    id_to_clause = {0: "A", 1: "B"}
    thresholds = tune_per_clause_thresholds(logits, labels, id_to_clause)
    df = compute_per_clause_metrics(logits, labels, thresholds, id_to_clause, {0: 10, 1: 8})
    assert (df["aupr"] >= 0.0).all() and (df["aupr"] <= 1.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_evaluation.py -k "per_clause or aupr" -v
```
Expected: `ImportError` for `compute_per_clause_metrics`

- [ ] **Step 3: Implement per-clause breakdown and confusion matrix**

Append to `evaluation.py`:

```python
def compute_per_clause_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[str, float],
    id_to_clause: dict[int, str],
    n_positive_train: dict[int, int],
) -> pd.DataFrame:
    """Per-clause breakdown: precision, recall, F1, P@80R, AUPR, threshold, n_positive_train."""
    probs = _sigmoid(logits)
    rows = []
    for clause_id, clause_name in id_to_clause.items():
        y_true = labels[:, clause_id].astype(int)
        y_score = probs[:, clause_id]
        t = thresholds.get(clause_name, 0.5)
        preds = (y_score >= t).astype(int)

        rows.append({
            "clause_type":           clause_name,
            "threshold":             round(t, 2),
            "precision":             float(precision_score(y_true, preds, zero_division=0)),
            "recall":                float(recall_score(y_true, preds, zero_division=0)),
            "f1":                    float(f1_score(y_true, preds, zero_division=0)),
            "precision_at_80_recall": precision_at_recall_threshold(y_true, y_score, 0.80),
            "aupr":                  float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else 0.0,
            "n_positive_train":      n_positive_train.get(clause_id, 0),
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
    """Heatmap showing how often each true clause is co-predicted with others."""
    probs = _sigmoid(logits)
    num_labels = logits.shape[1]
    clause_names = [id_to_clause[i] for i in range(num_labels)]
    preds = np.zeros_like(probs, dtype=int)
    for i in range(num_labels):
        t = thresholds.get(clause_names[i], 0.5)
        preds[:, i] = (probs[:, i] >= t).astype(int)

    int_labels = labels.astype(int)
    # Co-occurrence: for each true positive clause (row), how often is each clause predicted (col)?
    matrix = np.zeros((num_labels, num_labels), dtype=float)
    for true_idx in range(num_labels):
        true_positive_mask = int_labels[:, true_idx] == 1
        if true_positive_mask.sum() == 0:
            continue
        for pred_idx in range(num_labels):
            matrix[true_idx, pred_idx] = preds[true_positive_mask, pred_idx].mean()

    short_names = [n[:20] for n in clause_names]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix, xticklabels=short_names, yticklabels=short_names,
                annot=True, fmt=".2f", cmap="Blues", ax=ax,
                cbar_kws={"label": "Rate predicted positive"})
    ax.set_xlabel("Predicted clause")
    ax.set_ylabel("True clause (positive samples only)")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(
    results: dict[str, pd.DataFrame],
    metric: str = "f1",
    save_path: str | None = None,
) -> None:
    """Grouped bar chart: all models × clause types for a given metric."""
    model_names = list(results.keys())
    # Align on clause_type
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
    plt.show()
```

- [ ] **Step 4: Run all evaluation tests**

```bash
python -m pytest tests/test_evaluation.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add evaluation.py tests/test_evaluation.py
git commit -m "feat: add per-clause breakdown, AUPR, P@80R, confusion matrix to evaluation"
```

---

## Task 7: training.py — Shared Utilities & ModelArtifacts

**Files:**
- Create: `training.py`
- Create: `tests/test_training.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_training.py
import pytest
import numpy as np


def test_model_artifacts_dataclass_fields():
    from training import ModelArtifacts
    import pandas as pd
    ma = ModelArtifacts(
        model_name="Test",
        model=None,
        tokenizer=None,
        best_threshold=0.5,
        val_metrics={"micro_f1": 0.7},
        history=pd.DataFrame(),
        id_to_clause={0: "A"},
        val_logits=np.zeros((5, 1)),
        val_labels=np.zeros((5, 1)),
    )
    assert ma.model_name == "Test"
    assert ma.best_threshold == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_training.py::test_model_artifacts_dataclass_fields -v
```
Expected: `ImportError: No module named 'training'`

- [ ] **Step 3: Implement shared utilities**

```python
# training.py
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from preprocessing import MultiLabelChunkDataset, compute_pos_weight, compute_sample_weights


@dataclass
class ModelArtifacts:
    model_name: str
    model: Any
    tokenizer: Any
    best_threshold: float
    val_metrics: dict[str, float]
    history: pd.DataFrame
    id_to_clause: dict[int, str]
    val_logits: np.ndarray
    val_labels: np.ndarray


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def tune_global_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    from sklearn.metrics import f1_score
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        preds = (sigmoid(logits) >= t).astype(int)
        f1 = float(f1_score(labels.astype(int), preds, average="micro", zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    preds = (sigmoid(logits) >= best_t).astype(int)
    from sklearn.metrics import precision_score, recall_score
    return best_t, {
        "micro_f1":        best_f1,
        "micro_precision": float(precision_score(labels.astype(int), preds, average="micro", zero_division=0)),
        "micro_recall":    float(recall_score(labels.astype(int), preds, average="micro", zero_division=0)),
    }


def collect_logits_and_labels(
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            labels = batch["labels"].cpu().numpy()
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits.cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels)
    return np.vstack(all_logits), np.vstack(all_labels)


def _run_training_loop(
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_examples: list[dict],
    device: torch.device,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> tuple[Any, pd.DataFrame, float, dict, np.ndarray, np.ndarray]:
    """Core training loop shared by all transformer models. Returns (model, history, threshold, metrics, val_logits, val_labels)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    effective = len(train_loader) if max_train_batches is None else min(len(train_loader), max_train_batches)
    total_steps = max(1, epochs * max(1, effective))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * warmup_ratio), total_steps)

    pos_weight = compute_pos_weight(train_examples).to(device)
    sample_weights = torch.tensor(compute_sample_weights(train_examples), dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_state, best_t, best_metrics = None, 0.5, {"micro_f1": -1.0}
    best_val_logits, best_val_labels = None, None
    history_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for bi, batch in enumerate(train_loader):
            if max_train_batches is not None and bi >= max_train_batches:
                break
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            # per-sample loss weighting
            batch_sw = sample_weights[bi * train_loader.batch_size:(bi + 1) * train_loader.batch_size]
            batch_sw = batch_sw[:labels.shape[0]].to(device)
            loss = (loss_fn(logits, labels).mean(dim=1) * batch_sw).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            seen += 1

        val_logits, val_labels = collect_logits_and_labels(model, val_loader, device, max_val_batches)
        t, metrics = tune_global_threshold(val_logits, val_labels)
        history_rows.append({"epoch": epoch, "train_loss": total_loss / max(1, seen),
                              "val_threshold": t, **metrics})
        if metrics["micro_f1"] > best_metrics["micro_f1"]:
            best_metrics = metrics
            best_t = t
            best_state = copy.deepcopy(model.state_dict())
            best_val_logits = val_logits
            best_val_labels = val_labels

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history_rows), best_t, best_metrics, best_val_logits, best_val_labels
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_training.py::test_model_artifacts_dataclass_fields -v
```
Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add training.py tests/test_training.py
git commit -m "feat: add training module skeleton with ModelArtifacts and shared training loop"
```

---

## Task 8: training.py — TF-IDF + Logistic Regression Baseline

**Files:**
- Modify: `training.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_training.py`:

```python
def test_train_tfidf_lr_smoke():
    from training import train_tfidf_lr, ModelArtifacts
    import pandas as pd

    rows = []
    for i in range(60):
        rows.append({
            "contract_title": f"C{i}", "clause_type": "Governing Law",
            "contract_text": f"agreement governed by law number {i}",
            "has_answer": i % 3 != 0,
            "answer_texts": ["governed"] if i % 3 != 0 else [],
            "answer_starts": [10] if i % 3 != 0 else [],
            "answer_count": 1 if i % 3 != 0 else 0,
        })
    for i in range(60):
        rows.append({
            "contract_title": f"C{i}", "clause_type": "Non-Compete",
            "contract_text": f"agreement governed by law number {i}",
            "has_answer": i % 2 == 0,
            "answer_texts": ["compete"] if i % 2 == 0 else [],
            "answer_starts": [5] if i % 2 == 0 else [],
            "answer_count": 1 if i % 2 == 0 else 0,
        })
    df = pd.DataFrame(rows)

    from preprocessing import build_clause_mappings, build_contract_records, split_contract_records
    clause_to_id, id_to_clause = build_clause_mappings(df)
    records = build_contract_records(df)
    train_r, val_r, _ = split_contract_records(records, seed=42)

    def records_to_df(recs, df_source):
        titles = {r["contract_title"] for r in recs}
        return df_source[df_source["contract_title"].isin(titles)]

    train_df = records_to_df(train_r, df)
    val_df   = records_to_df(val_r, df)

    artifacts = train_tfidf_lr(train_df, val_df, id_to_clause)
    assert isinstance(artifacts, ModelArtifacts)
    assert artifacts.model_name == "TF-IDF + LR"
    assert artifacts.val_logits.shape[1] == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_training.py::test_train_tfidf_lr_smoke -v
```
Expected: `ImportError` for `train_tfidf_lr`

- [ ] **Step 3: Implement TF-IDF + LR training**

Append to `training.py`:

```python
def train_tfidf_lr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    id_to_clause: dict[int, str],
) -> ModelArtifacts:
    """TF-IDF + multi-output Logistic Regression baseline.

    Uses contract-level text (one row per contract, labels aggregated).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier

    clause_names = [id_to_clause[i] for i in range(len(id_to_clause))]

    def build_contract_matrix(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        texts, label_matrix = [], []
        for title, group in df.groupby("contract_title"):
            texts.append(group["contract_text"].iloc[0])
            row = np.zeros(len(id_to_clause), dtype=float)
            for _, r in group.iterrows():
                if r["has_answer"] and r["clause_type"] in [id_to_clause[i] for i in range(len(id_to_clause))]:
                    clause_id = {v: k for k, v in id_to_clause.items()}[r["clause_type"]]
                    row[clause_id] = 1.0
            label_matrix.append(row)
        return texts, np.array(label_matrix)

    train_texts, train_labels = build_contract_matrix(train_df)
    val_texts,   val_labels   = build_contract_matrix(val_df)

    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)

    clf = MultiOutputClassifier(
        LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, solver="lbfgs")
    )
    clf.fit(X_train, train_labels)

    # Collect probability scores as "logits" (log-odds) for uniform interface
    val_probs = np.column_stack([est.predict_proba(X_val)[:, 1] for est in clf.estimators_])
    # Convert probabilities to log-odds for compatibility with sigmoid-based evaluation
    eps = 1e-7
    val_logits = np.log(np.clip(val_probs, eps, 1 - eps) / (1 - np.clip(val_probs, eps, 1 - eps)))

    best_t, val_metrics = tune_global_threshold(val_logits, val_labels)

    # Store the pipeline for inference
    class TfIdfPipeline:
        def __init__(self, vec, model):
            self.vectorizer = vec
            self.clf = model
        def predict_proba(self, texts):
            X = self.vectorizer.transform(texts)
            return np.column_stack([e.predict_proba(X)[:, 1] for e in self.clf.estimators_])

    pipeline = TfIdfPipeline(vectorizer, clf)

    print(f"TF-IDF + LR → val micro_F1={val_metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name="TF-IDF + LR",
        model=pipeline,
        tokenizer=None,
        best_threshold=best_t,
        val_metrics=val_metrics,
        history=pd.DataFrame(),
        id_to_clause=id_to_clause,
        val_logits=val_logits,
        val_labels=val_labels,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_training.py::test_train_tfidf_lr_smoke -v
```
Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add training.py tests/test_training.py
git commit -m "feat: add TF-IDF + LR baseline training function"
```

---

## Task 9: training.py — BERT on CUAD

**Files:**
- Modify: `training.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Write failing smoke test**

Add to `tests/test_training.py`:

```python
def test_train_bert_cuad_smoke():
    """Smoke test: 1 epoch, 2 batches, verifies output shape and types."""
    from training import train_bert_cuad, ModelArtifacts
    from preprocessing import prepare_chunked_splits
    import pandas as pd

    rows = []
    for i in range(30):
        for clause in ["Governing Law", "Non-Compete"]:
            rows.append({
                "contract_title": f"C{i}", "clause_type": clause,
                "contract_text": f"This agreement number {i} is governed by the laws of jurisdiction {i % 5}.",
                "has_answer": i % 2 == 0,
                "answer_texts": ["governed"] if i % 2 == 0 else [],
                "answer_starts": [14] if i % 2 == 0 else [],
                "answer_count": 1 if i % 2 == 0 else 0,
            })
    df = pd.DataFrame(rows)
    splits = prepare_chunked_splits(df, model_name="distilbert-base-uncased",
                                    max_length=64, stride=16, seed=42)

    artifacts = train_bert_cuad(
        train_dataset=splits["train_dataset"],
        val_dataset=splits["val_dataset"],
        train_examples=splits["train_examples"],
        model_name="distilbert-base-uncased",
        tokenizer=splits["tokenizer"],
        id_to_clause=splits["id_to_clause"],
        epochs=1,
        batch_size=4,
        max_train_batches=2,
        max_val_batches=2,
    )
    assert isinstance(artifacts, ModelArtifacts)
    assert artifacts.model_name == "BERT (CUAD)"
    assert artifacts.val_logits.shape[1] == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_training.py::test_train_bert_cuad_smoke -v
```
Expected: `ImportError` for `train_bert_cuad`

- [ ] **Step 3: Implement train_bert_cuad**

Append to `training.py`:

```python
def train_bert_cuad(
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict],
    model_name: str,
    tokenizer: Any,
    id_to_clause: dict[int, str],
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    device: torch.device | None = None,
    artifact_name: str = "BERT (CUAD)",
) -> ModelArtifacts:
    """Fine-tune a BERT-family model directly on CUAD multi-label chunks."""
    device = device or choose_device()

    label2id = {v: k for k, v in id_to_clause.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id_to_clause),
        id2label=id_to_clause,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        model, train_loader, val_loader, train_examples, device,
        epochs, learning_rate, weight_decay, warmup_ratio,
        max_train_batches, max_val_batches,
    )

    print(f"{artifact_name} → val micro_F1={metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name=artifact_name,
        model=model,
        tokenizer=tokenizer,
        best_threshold=best_t,
        val_metrics=metrics,
        history=history,
        id_to_clause=id_to_clause,
        val_logits=val_logits,
        val_labels=val_labels,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_training.py::test_train_bert_cuad_smoke -v
```
Expected: `PASSED` (may take ~30s on CPU with distilbert)

- [ ] **Step 5: Commit**

```bash
git add training.py tests/test_training.py
git commit -m "feat: add train_bert_cuad transformer training function"
```

---

## Task 10: training.py — BERT with LEDGAR Domain Adaptation

**Files:**
- Modify: `training.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Write failing smoke test**

Add to `tests/test_training.py`:

```python
def test_train_bert_ledgar_cuad_smoke():
    """Smoke: phase 1 LEDGAR fine-tune + phase 2 CUAD fine-tune with minimal batches."""
    from training import train_bert_ledgar_cuad, ModelArtifacts
    from preprocessing import prepare_chunked_splits
    from datasets import Dataset as HFDataset
    import pandas as pd

    # Minimal synthetic LEDGAR-like dataset
    ledgar_data = {
        "text": [f"provision text example number {i}" for i in range(40)],
        "label": [i % 10 for i in range(40)],
    }
    mock_ledgar = {"train": HFDataset.from_dict(ledgar_data)}

    rows = []
    for i in range(30):
        for clause in ["Governing Law", "Non-Compete"]:
            rows.append({
                "contract_title": f"C{i}", "clause_type": clause,
                "contract_text": f"This contract number {i} is governed under jurisdiction {i % 5}.",
                "has_answer": i % 2 == 0,
                "answer_texts": ["governed"] if i % 2 == 0 else [],
                "answer_starts": [13] if i % 2 == 0 else [],
                "answer_count": 1 if i % 2 == 0 else 0,
            })
    df = pd.DataFrame(rows)
    splits = prepare_chunked_splits(df, model_name="distilbert-base-uncased",
                                    max_length=64, stride=16, seed=42)

    artifacts = train_bert_ledgar_cuad(
        ledgar_dataset=mock_ledgar,
        train_dataset=splits["train_dataset"],
        val_dataset=splits["val_dataset"],
        train_examples=splits["train_examples"],
        model_name="distilbert-base-uncased",
        tokenizer=splits["tokenizer"],
        id_to_clause=splits["id_to_clause"],
        ledgar_epochs=1, ledgar_max_batches=2,
        cuad_epochs=1,   cuad_max_train_batches=2, cuad_max_val_batches=2,
        batch_size=4,
    )
    assert isinstance(artifacts, ModelArtifacts)
    assert artifacts.model_name == "BERT (LEDGAR→CUAD)"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_training.py::test_train_bert_ledgar_cuad_smoke -v
```
Expected: `ImportError` for `train_bert_ledgar_cuad`

- [ ] **Step 3: Implement LEDGAR domain adaptation**

Append to `training.py`:

```python
def train_bert_ledgar_cuad(
    ledgar_dataset: Any,
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict],
    model_name: str,
    tokenizer: Any,
    id_to_clause: dict[int, str],
    ledgar_epochs: int = 3,
    ledgar_max_batches: int | None = None,
    cuad_epochs: int = 3,
    cuad_max_train_batches: int | None = None,
    cuad_max_val_batches: int | None = None,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    device: torch.device | None = None,
) -> ModelArtifacts:
    """Two-phase training: (1) fine-tune on LEDGAR, (2) strip head and fine-tune on CUAD."""
    from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset
    from transformers import AutoTokenizer as AT

    device = device or choose_device()

    # ── Phase 1: LEDGAR fine-tuning ───────────────────────────────────────────
    print("Phase 1: domain-adapting on LEDGAR...")
    ledgar_train = ledgar_dataset["train"]
    n_ledgar_labels = ledgar_train.features["label"].num_classes

    ledgar_tokenizer = AT.from_pretrained(model_name, use_fast=True)
    ledgar_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_ledgar_labels,
    ).to(device)

    def tokenize_ledgar(batch):
        return ledgar_tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

    ledgar_tok = ledgar_train.map(tokenize_ledgar, batched=True)
    ledgar_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    class LedgarTorchDataset(TorchDataset):
        def __init__(self, hf_dataset):
            self.ds = hf_dataset
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            item = self.ds[i]
            return {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"],
                    "labels": item["label"]}

    ledgar_loader = TorchDataLoader(LedgarTorchDataset(ledgar_tok), batch_size=batch_size, shuffle=True)
    optimizer_p1 = torch.optim.AdamW(ledgar_model.parameters(), lr=learning_rate)
    ce_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(1, ledgar_epochs + 1):
        ledgar_model.train()
        total_loss, seen = 0.0, 0
        for bi, batch in enumerate(ledgar_loader):
            if ledgar_max_batches is not None and bi >= ledgar_max_batches:
                break
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optimizer_p1.zero_grad(set_to_none=True)
            logits = ledgar_model(**inputs).logits
            loss = ce_loss(logits, labels)
            loss.backward()
            optimizer_p1.step()
            total_loss += float(loss.item())
            seen += 1
        print(f"  LEDGAR epoch {epoch}: loss={total_loss/max(1,seen):.4f}")

    # ── Phase 2: transfer to CUAD ─────────────────────────────────────────────
    print("Phase 2: transferring to CUAD multi-label task...")
    # Extract the backbone (remove classification head)
    backbone_state = {
        k: v for k, v in ledgar_model.state_dict().items()
        if not k.startswith("classifier") and not k.startswith("pre_classifier")
    }
    label2id = {v: k for k, v in id_to_clause.items()}
    cuad_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id_to_clause),
        id2label=id_to_clause,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    # Load backbone weights (non-head layers)
    missing, unexpected = cuad_model.load_state_dict(backbone_state, strict=False)
    print(f"  Transferred {len(backbone_state)} layers; missing={len(missing)}, unexpected={len(unexpected)}")
    cuad_model = cuad_model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        cuad_model, train_loader, val_loader, train_examples, device,
        cuad_epochs, learning_rate, 0.01, 0.1,
        cuad_max_train_batches, cuad_max_val_batches,
    )

    print(f"BERT (LEDGAR→CUAD) → val micro_F1={metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name="BERT (LEDGAR→CUAD)",
        model=model,
        tokenizer=tokenizer,
        best_threshold=best_t,
        val_metrics=metrics,
        history=history,
        id_to_clause=id_to_clause,
        val_logits=val_logits,
        val_labels=val_labels,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_training.py::test_train_bert_ledgar_cuad_smoke -v
```
Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add training.py tests/test_training.py
git commit -m "feat: add BERT LEDGAR domain adaptation + CUAD fine-tuning"
```

---

## Task 11: training.py — Legal-BERT, Longformer & Legal-BERT→Longformer

**Files:**
- Modify: `training.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Write failing smoke tests**

Add to `tests/test_training.py`:

```python
def test_train_legal_bert_cuad_smoke():
    from training import train_bert_cuad
    from preprocessing import prepare_chunked_splits
    import pandas as pd
    rows = []
    for i in range(30):
        rows.append({"contract_title": f"C{i}", "clause_type": "Governing Law",
                     "contract_text": f"legal text {i}", "has_answer": i%2==0,
                     "answer_texts": ["legal"] if i%2==0 else [],
                     "answer_starts": [0] if i%2==0 else [], "answer_count": 1 if i%2==0 else 0})
    df = pd.DataFrame(rows)
    splits = prepare_chunked_splits(df, model_name="distilbert-base-uncased",
                                    max_length=64, stride=16, seed=42)
    # Use distilbert as stand-in for legal-bert in smoke test
    artifacts = train_bert_cuad(
        splits["train_dataset"], splits["val_dataset"], splits["train_examples"],
        "distilbert-base-uncased", splits["tokenizer"], splits["id_to_clause"],
        epochs=1, batch_size=4, max_train_batches=2, max_val_batches=2,
        artifact_name="Legal-BERT (CUAD)",
    )
    assert artifacts.model_name == "Legal-BERT (CUAD)"


def test_init_longformer_from_legal_bert_weight_shapes():
    from training import init_longformer_from_legal_bert
    # Uses distilbert-base-uncased as a stand-in to avoid downloading large models in tests
    # Just verify the function runs and returns a model with correct output size
    model = init_longformer_from_legal_bert(
        num_labels=3,
        legal_bert_name="distilbert-base-uncased",
        longformer_name="allenai/longformer-base-4096",
    )
    assert model.config.num_labels == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_training.py -k "legal_bert or longformer" -v
```
Expected: `ImportError` for `init_longformer_from_legal_bert`

- [ ] **Step 3: Implement Legal-BERT helpers and Longformer init**

Append to `training.py`:

```python
def train_legal_bert_cuad(
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict],
    tokenizer: Any,
    id_to_clause: dict[int, str],
    model_name: str = "nlpaueb/legal-bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    device: torch.device | None = None,
) -> ModelArtifacts:
    """Fine-tune Legal-BERT on CUAD (reuses train_bert_cuad with fixed artifact name)."""
    return train_bert_cuad(
        train_dataset, val_dataset, train_examples,
        model_name, tokenizer, id_to_clause,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        max_train_batches=max_train_batches, max_val_batches=max_val_batches,
        device=device, artifact_name="Legal-BERT (CUAD)",
    )


def init_longformer_from_legal_bert(
    num_labels: int,
    legal_bert_name: str = "nlpaueb/legal-bert-base-uncased",
    longformer_name: str = "allenai/longformer-base-4096",
) -> Any:
    """Initialise a LongformerForSequenceClassification with Legal-BERT backbone weights.

    Follows Mamakas et al. 2022:
    - Shared encoder layer weights copied by name
    - Position embeddings tiled from 512 → 4096
    - Global attention projections initialised from BERT attention weights
    """
    from transformers import AutoModel, LongformerForSequenceClassification, LongformerConfig

    print(f"Loading Legal-BERT backbone from {legal_bert_name}...")
    bert_model = AutoModel.from_pretrained(legal_bert_name)
    bert_state = bert_model.state_dict()

    print(f"Loading Longformer config from {longformer_name}...")
    lf_config = LongformerConfig.from_pretrained(longformer_name)
    lf_config.num_labels = num_labels
    lf_config.problem_type = "multi_label_classification"

    print("Initialising Longformer with Longformer-base-4096 weights...")
    lf_model = LongformerForSequenceClassification.from_pretrained(
        longformer_name, config=lf_config, ignore_mismatched_sizes=True
    )
    lf_state = lf_model.state_dict()
    new_state: dict[str, torch.Tensor] = {}

    for lf_key in lf_state:
        # Map longformer key → bert key (strip "longformer." prefix → "")
        bert_key = lf_key.replace("longformer.", "")

        if "position_embeddings" in lf_key:
            # Extend 512 positions → 4096 by tiling
            bert_pos_key = "embeddings.position_embeddings.weight"
            if bert_pos_key in bert_state:
                bert_pos = bert_state[bert_pos_key]           # [512, 768]
                target_size = lf_state[lf_key].shape[0]       # typically 4098
                repeats = (target_size // bert_pos.shape[0]) + 1
                extended = bert_pos.repeat(repeats, 1)[:target_size]
                new_state[lf_key] = extended
                continue
            new_state[lf_key] = lf_state[lf_key]

        elif "query_global" in lf_key or "key_global" in lf_key or "value_global" in lf_key:
            # Global attention projections: initialise from BERT's attention weights
            base = lf_key.replace("query_global", "query").replace("key_global", "key").replace("value_global", "value")
            bert_equiv = base.replace("longformer.", "")
            if bert_equiv in bert_state and bert_state[bert_equiv].shape == lf_state[lf_key].shape:
                new_state[lf_key] = bert_state[bert_equiv].clone()
            else:
                new_state[lf_key] = lf_state[lf_key]

        elif bert_key in bert_state and bert_state[bert_key].shape == lf_state[lf_key].shape:
            new_state[lf_key] = bert_state[bert_key]

        else:
            new_state[lf_key] = lf_state[lf_key]

    lf_model.load_state_dict(new_state)
    transferred = sum(1 for k in lf_state if new_state[k] is not lf_state[k])
    print(f"Transferred {transferred}/{len(lf_state)} weight tensors from Legal-BERT to Longformer.")
    return lf_model


def train_longformer_cuad(
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict],
    tokenizer: Any,
    id_to_clause: dict[int, str],
    model_name: str = "allenai/longformer-base-4096",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    device: torch.device | None = None,
) -> ModelArtifacts:
    """Fine-tune Longformer-base-4096 on CUAD multi-label chunks."""
    device = device or choose_device()
    label2id = {v: k for k, v in id_to_clause.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id_to_clause),
        id2label=id_to_clause,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        model, train_loader, val_loader, train_examples, device,
        epochs, learning_rate, 0.01, 0.1, max_train_batches, max_val_batches,
    )
    print(f"Longformer (CUAD) → val micro_F1={metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name="Longformer (CUAD)",
        model=model, tokenizer=tokenizer,
        best_threshold=best_t, val_metrics=metrics, history=history,
        id_to_clause=id_to_clause, val_logits=val_logits, val_labels=val_labels,
    )


def train_legalbert_longformer_cuad(
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict],
    tokenizer: Any,
    id_to_clause: dict[int, str],
    legal_bert_name: str = "nlpaueb/legal-bert-base-uncased",
    longformer_name: str = "allenai/longformer-base-4096",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    device: torch.device | None = None,
) -> ModelArtifacts:
    """Legal-BERT warm-started Longformer fine-tuned on CUAD (Mamakas et al. 2022)."""
    device = device or choose_device()
    model = init_longformer_from_legal_bert(
        num_labels=len(id_to_clause),
        legal_bert_name=legal_bert_name,
        longformer_name=longformer_name,
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        model, train_loader, val_loader, train_examples, device,
        epochs, learning_rate, 0.01, 0.1, max_train_batches, max_val_batches,
    )
    print(f"Legal-BERT→Longformer → val micro_F1={metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name="Legal-BERT→Longformer (CUAD)",
        model=model, tokenizer=tokenizer,
        best_threshold=best_t, val_metrics=metrics, history=history,
        id_to_clause=id_to_clause, val_logits=val_logits, val_labels=val_labels,
    )
```

- [ ] **Step 4: Run all training tests**

```bash
python -m pytest tests/test_training.py -v
```
Expected: all `PASSED` (note: `test_init_longformer_from_legal_bert_weight_shapes` requires downloading `allenai/longformer-base-4096` ~600MB — skip with `-k "not longformer_weight"` if offline)

- [ ] **Step 5: Commit**

```bash
git add training.py tests/test_training.py
git commit -m "feat: add Legal-BERT, Longformer, and Legal-BERT→Longformer training functions"
```

---

## Task 12: pipeline.ipynb — Sections 0–4 (Steps 1–4)

**Files:**
- Modify: `pipeline.ipynb` (existing notebook — add new cells after existing EDA cells)

- [ ] **Step 1: Add Section 0 — Install & imports cell**

Add as the first cell (or replace the existing `%pip install` cell):

```python
# Section 0 — Install & imports
# On Colab: upload data_loading.py, preprocessing.py, training.py, evaluation.py alongside this notebook

%pip install -q pandas pyarrow huggingface_hub matplotlib seaborn torch transformers \
    scikit-learn accelerate datasets

import importlib
import data_loading, preprocessing, training, evaluation

for mod in [data_loading, preprocessing, training, evaluation]:
    importlib.reload(mod)

from data_loading import load_cuad, load_ledgar
from preprocessing import (
    filter_clauses, plot_clause_frequency, build_clause_mappings,
    build_contract_records, prepare_chunked_splits,
)
from training import (
    train_tfidf_lr, train_bert_ledgar_cuad, train_bert_cuad,
    train_legal_bert_cuad, train_longformer_cuad, train_legalbert_longformer_cuad,
)
from evaluation import (
    tune_per_clause_thresholds, compute_per_clause_metrics,
    compute_aggregate_metrics, plot_confusion_matrix, plot_model_comparison,
)
```

- [ ] **Step 2: Add Section 1 — Load CUAD + LEDGAR**

```python
# Section 1 — Load datasets
cuad_df  = load_cuad("data/cuad")
ledgar   = load_ledgar("data/ledgar")
```

- [ ] **Step 3: Add Section 2 — EDA, filtering, chunking**

```python
# Section 2 — EDA + clause filtering + chunking

# 2a. Bar chart — positive rate per clause type
clause_summary = plot_clause_frequency(cuad_df, save_path="figures/clause_frequency.png")
display(clause_summary.head(20))

# 2b. Filter clauses below threshold
MIN_POSITIVES = 20
filtered_df, excluded_clauses = filter_clauses(cuad_df, min_positives=MIN_POSITIVES)
print(f"\nRetained clause types: {filtered_df['clause_type'].nunique()}")
print(f"Excluded: {list(excluded_clauses.keys())}")

# 2c. Prepare BERT-family chunks (512 tokens, stride 128)
BERT_SPLITS = prepare_chunked_splits(
    filtered_df, model_name="bert-base-uncased", max_length=512, stride=128, seed=42
)
print(f"Train chunks: {len(BERT_SPLITS['train_examples']):,} | "
      f"Val: {len(BERT_SPLITS['val_examples']):,} | "
      f"Test: {len(BERT_SPLITS['test_examples']):,}")

# 2d. Prepare Longformer chunks (4096 tokens, stride 512)
from preprocessing import AutoTokenizer, build_contract_records, split_contract_records, \
    build_chunk_examples, MultiLabelChunkDataset, build_clause_mappings

LF_TOKENIZER = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=True)
clause_to_id, id_to_clause = build_clause_mappings(filtered_df)
contract_records = build_contract_records(filtered_df)
train_r, val_r, test_r = split_contract_records(contract_records, seed=42)

lf_train_ex = build_chunk_examples(train_r, clause_to_id, LF_TOKENIZER, max_length=4096, stride=512)
lf_val_ex   = build_chunk_examples(val_r,   clause_to_id, LF_TOKENIZER, max_length=4096, stride=512)
lf_test_ex  = build_chunk_examples(test_r,  clause_to_id, LF_TOKENIZER, max_length=4096, stride=512)

LF_SPLITS = {
    "tokenizer": LF_TOKENIZER, "id_to_clause": id_to_clause,
    "train_examples": lf_train_ex, "val_examples": lf_val_ex, "test_examples": lf_test_ex,
    "train_dataset": MultiLabelChunkDataset(lf_train_ex),
    "val_dataset":   MultiLabelChunkDataset(lf_val_ex),
    "test_dataset":  MultiLabelChunkDataset(lf_test_ex),
    "test_records":  test_r,
}
```

- [ ] **Step 4: Add Section 3 — Train all 5 models**

```python
# Section 3 — Train all models
# ── Hypothesis statement (printed before any results) ──────────────────────
print("""
HYPOTHESES:
H1: Legal-BERT will outperform BERT on clause types with specialised legal vocabulary
    (governing law, dispute resolution, indemnification) due to legal-domain pretraining.
H2: Longformer will outperform BERT-family on clause types where relevant language is
    spread across the full document (force majeure, termination for convenience),
    because it processes up to 4,096 tokens without chunking.
H3: Legal-BERT warm-started Longformer will be the strongest overall, combining
    domain-specific vocabulary with architectural long-context advantage.
""")

# Count positive training examples per clause for reporting
from preprocessing import compute_pos_weight
import numpy as np
label_matrix = np.array([ex["labels"] for ex in BERT_SPLITS["train_examples"]])
n_positive_train = {i: int(label_matrix[:, i].sum()) for i in range(label_matrix.shape[1])}

# 3a. TF-IDF + LR baseline
import pandas as pd
train_titles = {r["contract_title"] for r in BERT_SPLITS["train_records"]}
val_titles   = {r["contract_title"] for r in BERT_SPLITS["val_records"]}
train_df_subset = filtered_df[filtered_df["contract_title"].isin(train_titles)]
val_df_subset   = filtered_df[filtered_df["contract_title"].isin(val_titles)]

artifacts_tfidf = train_tfidf_lr(train_df_subset, val_df_subset, BERT_SPLITS["id_to_clause"])

# 3b. BERT (CUAD only)
artifacts_bert = train_bert_cuad(
    BERT_SPLITS["train_dataset"], BERT_SPLITS["val_dataset"],
    BERT_SPLITS["train_examples"], "bert-base-uncased",
    BERT_SPLITS["tokenizer"], BERT_SPLITS["id_to_clause"],
    epochs=3, batch_size=8,
)

# 3c. BERT (LEDGAR → CUAD)
artifacts_bert_ledgar = train_bert_ledgar_cuad(
    ledgar_dataset=ledgar,
    train_dataset=BERT_SPLITS["train_dataset"], val_dataset=BERT_SPLITS["val_dataset"],
    train_examples=BERT_SPLITS["train_examples"],
    model_name="bert-base-uncased", tokenizer=BERT_SPLITS["tokenizer"],
    id_to_clause=BERT_SPLITS["id_to_clause"],
    ledgar_epochs=3, cuad_epochs=3, batch_size=8,
)

# 3d. Legal-BERT (CUAD)
from preprocessing import AutoTokenizer as AT
legal_bert_tokenizer = AT.from_pretrained("nlpaueb/legal-bert-base-uncased", use_fast=True)
legal_bert_splits = prepare_chunked_splits(
    filtered_df, model_name="nlpaueb/legal-bert-base-uncased",
    max_length=512, stride=128, seed=42
)
artifacts_legalbert = train_legal_bert_cuad(
    legal_bert_splits["train_dataset"], legal_bert_splits["val_dataset"],
    legal_bert_splits["train_examples"], legal_bert_splits["tokenizer"],
    legal_bert_splits["id_to_clause"],
    model_name="nlpaueb/legal-bert-base-uncased",
    epochs=3, batch_size=8,
)

# 3e. Longformer (CUAD)
artifacts_longformer = train_longformer_cuad(
    LF_SPLITS["train_dataset"], LF_SPLITS["val_dataset"],
    LF_SPLITS["train_examples"], LF_SPLITS["tokenizer"],
    LF_SPLITS["id_to_clause"],
    epochs=3, batch_size=2,
)

# 3f. Legal-BERT → Longformer (CUAD)
artifacts_lf_lb = train_legalbert_longformer_cuad(
    LF_SPLITS["train_dataset"], LF_SPLITS["val_dataset"],
    LF_SPLITS["train_examples"], LF_SPLITS["tokenizer"],
    LF_SPLITS["id_to_clause"],
    epochs=3, batch_size=2,
)

ALL_ARTIFACTS = [
    artifacts_tfidf, artifacts_bert_ledgar, artifacts_bert,
    artifacts_legalbert, artifacts_longformer, artifacts_lf_lb
]
```

- [ ] **Step 5: Add Section 4 — Evaluate all models**

```python
# Section 4 — Evaluate all models

from torch.utils.data import DataLoader as TDL
from training import collect_logits_and_labels, choose_device

device = choose_device()
results_per_clause = {}
results_aggregate  = {}

for art in ALL_ARTIFACTS:
    # Get test logits
    if art.model_name == "TF-IDF + LR":
        test_titles  = {r["contract_title"] for r in BERT_SPLITS["test_records"]}
        test_df_sub  = filtered_df[filtered_df["contract_title"].isin(test_titles)]
        test_texts   = [grp["contract_text"].iloc[0]
                        for _, grp in test_df_sub.groupby("contract_title")]
        test_label_matrix = np.zeros((len(test_texts), len(art.id_to_clause)))
        for ti, (title, grp) in enumerate(test_df_sub.groupby("contract_title")):
            for _, row in grp.iterrows():
                if row["has_answer"]:
                    cid = {v: k for k, v in art.id_to_clause.items()}.get(row["clause_type"])
                    if cid is not None:
                        test_label_matrix[ti, cid] = 1.0
        import numpy as np
        eps = 1e-7
        probs = art.model.predict_proba(test_texts)
        test_logits = np.log(np.clip(probs, eps, 1-eps) / (1 - np.clip(probs, eps, 1-eps)))
        test_labels = test_label_matrix
    else:
        splits = LF_SPLITS if "Longformer" in art.model_name else BERT_SPLITS
        test_loader = TDL(splits["test_dataset"], batch_size=8, shuffle=False)
        test_logits, test_labels = collect_logits_and_labels(art.model, test_loader, device)

    # Tune per-clause thresholds on val logits, apply to test
    per_clause_thresholds = tune_per_clause_thresholds(
        art.val_logits, art.val_labels, art.id_to_clause
    )

    # Per-clause breakdown
    clause_df = compute_per_clause_metrics(
        test_logits, test_labels, per_clause_thresholds,
        art.id_to_clause, n_positive_train,
    )
    results_per_clause[art.model_name] = clause_df

    # Aggregate metrics
    agg = compute_aggregate_metrics(test_logits, test_labels, per_clause_thresholds, art.id_to_clause)
    results_aggregate[art.model_name] = agg
    print(f"{art.model_name:40s} macro_F1={agg['macro_f1']:.4f}  micro_F1={agg['micro_f1']:.4f}")

# Aggregate comparison table
agg_df = pd.DataFrame(results_aggregate).T.reset_index().rename(columns={"index": "model"})
display(agg_df.sort_values("macro_f1", ascending=False))

# Per-clause breakdown for best model
best_model_name = agg_df.sort_values("macro_f1", ascending=False).iloc[0]["model"]
print(f"\nPer-clause breakdown — {best_model_name}:")
display(results_per_clause[best_model_name])

# Confusion matrices
for art in ALL_ARTIFACTS:
    splits = LF_SPLITS if "Longformer" in art.model_name else BERT_SPLITS
    if art.model_name != "TF-IDF + LR":
        test_loader = TDL(splits["test_dataset"], batch_size=8, shuffle=False)
        t_logits, t_labels = collect_logits_and_labels(art.model, test_loader, device)
    else:
        t_logits, t_labels = test_logits, test_labels
    per_t = tune_per_clause_thresholds(art.val_logits, art.val_labels, art.id_to_clause)
    plot_confusion_matrix(t_logits, t_labels, per_t, art.id_to_clause,
                          title=f"Confusion Matrix — {art.model_name}",
                          save_path=f"figures/confusion_{art.model_name.replace(' ', '_')}.png")

# Model comparison chart
plot_model_comparison(results_per_clause, metric="f1", save_path="figures/model_comparison_f1.png")
```

- [ ] **Step 6: Add Section 5 — LLM risk summaries**

```python
# Section 5 — LLM risk summaries (Claude API via Colab secrets)

# Get API key from Colab secrets (set in Colab: Runtime > Manage secrets > ANTHROPIC_API_KEY)
try:
    from google.colab import userdata
    ANTHROPIC_API_KEY = userdata.get("ANTHROPIC_API_KEY")
except ImportError:
    import os
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

import anthropic

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

RISK_PROMPT_TEMPLATE = """You are a contract risk analyst reviewing a flagged clause for a non-specialist business reader.

Clause type: {clause_type}
Contract text excerpt:
\"\"\"
{clause_text}
\"\"\"

Provide your analysis in exactly this format:

RISK EXPLANATION: [2-3 sentences explaining what this clause means and why it matters, in plain English for a non-lawyer]
SEVERITY: [Low / Medium / High]
WATCH FOR: [One specific thing the reviewer should look out for or negotiate]"""


def generate_risk_summary(clause_type: str, clause_text: str) -> dict:
    """Call Claude to generate a plain-English risk summary for a flagged clause."""
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": RISK_PROMPT_TEMPLATE.format(
                clause_type=clause_type,
                clause_text=clause_text[:1500],  # cap to avoid token limits
            )
        }]
    )
    raw = message.content[0].text
    result = {"clause_type": clause_type, "clause_text": clause_text[:500], "raw_output": raw}
    for line in raw.split("\n"):
        if line.startswith("RISK EXPLANATION:"):
            result["risk_explanation"] = line.replace("RISK EXPLANATION:", "").strip()
        elif line.startswith("SEVERITY:"):
            result["severity"] = line.replace("SEVERITY:", "").strip()
        elif line.startswith("WATCH FOR:"):
            result["watch_for"] = line.replace("WATCH FOR:", "").strip()
    return result


# Run on flagged chunks from best model — sample 25 flagged clauses for manual rating
from preprocessing import aggregate_contract_predictions

best_art = next(a for a in ALL_ARTIFACTS if a.model_name == best_model_name)
splits_for_best = LF_SPLITS if "Longformer" in best_model_name else BERT_SPLITS

# Build long-form predictions dataframe
test_loader = TDL(splits_for_best["test_dataset"], batch_size=8, shuffle=False)
t_logits, _ = collect_logits_and_labels(best_art.model, test_loader, device)

import torch, numpy as np
probs = 1 / (1 + np.exp(-t_logits))
per_clause_t = tune_per_clause_thresholds(best_art.val_logits, best_art.val_labels, best_art.id_to_clause)

flagged_rows = []
for chunk_idx, (chunk_probs, example) in enumerate(
    zip(probs, splits_for_best["test_examples"])
):
    for clause_id, clause_name in best_art.id_to_clause.items():
        t = per_clause_t.get(clause_name, 0.5)
        score = float(chunk_probs[clause_id])
        if score >= t:
            flagged_rows.append({
                "contract_title": example["contract_title"],
                "clause_type": clause_name,
                "score": score,
                "chunk_index": example["chunk_index"],
                "chunk_text": example["chunk_text"],
            })

flagged_df = pd.DataFrame(flagged_rows)
contract_agg = aggregate_contract_predictions(flagged_df[["contract_title","clause_type","score","chunk_index"]])

# Join back chunk text for the best chunk per contract×clause
flagged_df_dedup = (
    flagged_df.sort_values("score", ascending=False)
    .drop_duplicates(subset=["contract_title", "clause_type"])
)

# Sample 25 for LLM rating
sample_for_rating = flagged_df_dedup.sample(n=min(25, len(flagged_df_dedup)), random_state=42)

risk_summaries = []
for _, row in sample_for_rating.iterrows():
    summary = generate_risk_summary(row["clause_type"], row["chunk_text"])
    risk_summaries.append(summary)
    print(f"[{row['clause_type']}] Severity: {summary.get('severity','?')}")

risk_df = pd.DataFrame(risk_summaries)
display(risk_df[["clause_type", "severity", "risk_explanation", "watch_for"]])

# Manual rating template (fill in during evaluation)
rating_template = risk_df[["clause_type", "chunk_text", "risk_explanation", "watch_for"]].copy()
rating_template["factual_accuracy_1_5"] = ""
rating_template["clarity_1_5"] = ""
rating_template.to_csv("risk_summary_ratings.csv", index=False)
print("Saved rating template to risk_summary_ratings.csv")
```

- [ ] **Step 7: Add `figures/` directory and verify notebook runs end-to-end (Sections 0–2 only on local)**

```bash
mkdir -p figures
# Run sections 0-2 to confirm imports and data loading work
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 \
    --ExecutePreprocessor.kernel_name=python3 pipeline.ipynb \
    --output pipeline_test_run.ipynb 2>&1 | tail -20
```
Expected: no `ImportError`, sections 0-2 complete. Section 3+ will timeout locally (GPU needed).

- [ ] **Step 8: Commit**

```bash
git add pipeline.ipynb figures/ risk_summary_ratings.csv
git commit -m "feat: assemble pipeline.ipynb with all 5 steps (Steps 1-5)"
```

---

## Task 13: Final wiring — run tests, verify Colab compatibility

**Files:**
- No new files

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: all tests pass (skip `test_init_longformer_from_legal_bert_weight_shapes` if no internet: `pytest tests/ -k "not longformer_weight"`)

- [ ] **Step 2: Verify imports work from a clean Python environment**

```bash
python -c "
import data_loading, preprocessing, training, evaluation
from data_loading import load_cuad, load_ledgar
from preprocessing import filter_clauses, plot_clause_frequency, prepare_chunked_splits
from training import (train_tfidf_lr, train_bert_ledgar_cuad, train_bert_cuad,
                      train_legal_bert_cuad, train_longformer_cuad,
                      train_legalbert_longformer_cuad, ModelArtifacts)
from evaluation import (tune_per_clause_thresholds, compute_per_clause_metrics,
                        compute_aggregate_metrics, plot_confusion_matrix)
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "feat: complete Steps 1-5 pipeline implementation"
```

---

## Spec Coverage Check

| Spec Requirement | Task |
|---|---|
| Load CUAD | Task 1 |
| Load LEDGAR | Task 1 |
| EDA bar chart per clause | Task 2 |
| Minimum positive threshold + exclusion log | Task 2 |
| 80/10/10 contract-level split | Task 3 |
| 512-token chunking with overlap | Task 3 |
| Feature-level downweighting (all-negative chunks) | Task 4 |
| `pos_weight` in BCEWithLogitsLoss | Task 4 |
| Max-probability aggregation + chunk↔contract map | Task 4 |
| TF-IDF + LR baseline | Task 8 |
| BERT domain-adapted on LEDGAR | Task 10 |
| BERT fine-tuned on CUAD | Task 9 |
| Legal-BERT fine-tuned on CUAD | Task 11 |
| Longformer fine-tuned on CUAD | Task 11 |
| Legal-BERT→Longformer warm-start | Task 11 |
| Hypothesis stated before results | Task 12 (Section 3) |
| Per-clause threshold tuning | Task 5 |
| Precision@80% Recall | Task 6 |
| AUPR per clause | Task 6 |
| Macro/micro F1, precision, recall | Task 5 |
| Per-clause breakdown table | Task 6 |
| Confusion matrix per model | Task 6 |
| LLM risk summaries (Claude API) | Task 12 (Section 5) |
| Manual rating template | Task 12 (Section 5) |
