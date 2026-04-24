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


def _tfidf_proba_col(est, X):
    """Return class-1 probability; DummyClassifier may only have one class."""
    proba = est.predict_proba(X)
    if len(est.classes_) == 1:
        return np.full(X.shape[0], float(est.classes_[0]))
    return proba[:, 1]


class _TfIdfPipeline:
    """Wraps a TF-IDF vectorizer + per-label estimators for inference.

    predict_proba returns raw probabilities in [0, 1] — NOT log-odds.
    Convert to log-odds before passing to sigmoid-based evaluation if needed:
        p = np.clip(pipeline.predict_proba(texts), 1e-7, 1-1e-7)
        logits = np.log(p / (1 - p))
    """

    def __init__(self, vec, ests):
        self.vectorizer = vec
        self.estimators_ = ests

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return np.column_stack([_tfidf_proba_col(e, X) for e in self.estimators_])


@dataclass
class ModelArtifacts:
    """Unified container returned by every training function."""
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
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))


def _tune_global_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Find the global threshold maximising micro-F1; return (threshold, metrics_dict)."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        preds = (_sigmoid(logits) >= t).astype(int)
        f1 = float(f1_score(labels.astype(int), preds, average="micro", zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    preds = (_sigmoid(logits) >= best_t).astype(int)
    return best_t, {
        "micro_f1":        best_f1,
        "micro_precision": float(precision_score(labels.astype(int), preds, average="micro", zero_division=0)),
        "micro_recall":    float(recall_score(labels.astype(int), preds, average="micro",    zero_division=0)),
    }


def collect_logits_and_labels(
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and return (logits, labels) as numpy arrays."""
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
    """Shared training loop for all transformer models.

    Returns (model, history_df, best_threshold, best_val_metrics, best_val_logits, best_val_labels).
    Applies pos_weight (BCEWithLogitsLoss) and per-sample downweighting for all-negative chunks.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    effective = len(train_loader) if max_train_batches is None else min(len(train_loader), max_train_batches)
    total_steps = max(1, epochs * max(1, effective))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * warmup_ratio), total_steps)

    pos_weight = compute_pos_weight(train_examples).to(device)
    sample_weights = torch.tensor(compute_sample_weights(train_examples), dtype=torch.float32)
    # reduction="none" so we can apply per-sample weights manually
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    # Mixed precision: ~1.5-2x speedup on CUDA (T4/V100/A100). No-op on CPU/MPS.
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state: dict | None = None
    best_t = 0.5
    best_metrics: dict[str, float] = {"micro_f1": -1.0}
    best_val_logits: np.ndarray | None = None
    best_val_labels: np.ndarray | None = None
    history_rows: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        sample_offset = 0  # tracks position in sample_weights across batches

        for bi, batch in enumerate(train_loader):
            if max_train_batches is not None and bi >= max_train_batches:
                break
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**inputs).logits
                # per-sample loss weighting: down-weight all-negative chunks
                batch_size_actual = labels.shape[0]
                batch_sw = sample_weights[sample_offset:sample_offset + batch_size_actual].to(device)
                loss = (loss_fn(logits, labels).mean(dim=1) * batch_sw).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.item())
            seen += 1
            sample_offset += batch_size_actual

        val_logits, val_labels = collect_logits_and_labels(model, val_loader, device, max_val_batches)
        t, metrics = _tune_global_threshold(val_logits, val_labels)
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


def train_tfidf_lr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    id_to_clause: dict[int, str],
) -> ModelArtifacts:
    """TF-IDF + multi-output Logistic Regression baseline (contract-level, not chunk-level).

    Uses full contract text per contract (one row per contract after groupby).
    Returns ModelArtifacts with logits stored as log-odds for compatibility
    with sigmoid-based evaluation functions.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier

    name_to_id = {v: k for k, v in id_to_clause.items()}

    def _build_contract_matrix(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        texts, label_rows = [], []
        for title, group in df.groupby("contract_title"):
            # All rows for a contract share the same full contract text; take first.
            texts.append(group["contract_text"].iloc[0])
            row = np.zeros(len(id_to_clause), dtype=float)
            for r in group.itertuples(index=False):
                if r.has_answer and r.clause_type in name_to_id:
                    row[name_to_id[r.clause_type]] = 1.0
            label_rows.append(row)
        return texts, np.array(label_rows)

    train_texts, train_labels = _build_contract_matrix(train_df)
    val_texts,   val_labels   = _build_contract_matrix(val_df)

    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)

    # Fit a per-label classifier. Labels with only one class in the training split
    # (all-positive or all-negative) cannot be fit by LR — use DummyClassifier instead.
    estimators = []
    for i in range(train_labels.shape[1]):
        col = train_labels[:, i]
        if len(np.unique(col)) < 2:
            est = DummyClassifier(strategy="most_frequent")
        else:
            est = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, solver="lbfgs")
        est.fit(X_train, col)
        estimators.append(est)

    # Collect class-1 probabilities; convert to log-odds for uniform interface with _sigmoid
    eps = 1e-7
    val_probs = np.column_stack([_tfidf_proba_col(est, X_val) for est in estimators])
    p = np.clip(val_probs, eps, 1 - eps)
    val_logits = np.log(p / (1 - p))

    best_t, val_metrics = _tune_global_threshold(val_logits, val_labels)

    pipeline = _TfIdfPipeline(vectorizer, estimators)

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
    """Fine-tune a BERT-family model directly on CUAD multi-label chunks.

    The artifact_name parameter lets Legal-BERT reuse this function with a
    different name (see train_legal_bert_cuad in Task 11).
    """
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

    _pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=(2 if _pin else 0), pin_memory=_pin, persistent_workers=_pin)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=(2 if _pin else 0), pin_memory=_pin, persistent_workers=_pin)

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
    """Two-phase training: (1) fine-tune on LEDGAR multi-class, (2) transfer to CUAD multi-label.

    Phase 1 uses LEDGAR labels to warm the model on legal language.
    Phase 2 strips the LEDGAR classification head, attaches a new multi-label head,
    and fine-tunes on CUAD using the shared _run_training_loop.
    """
    device = device or choose_device()

    # ── Phase 1: LEDGAR fine-tuning ───────────────────────────────────────────
    print("Phase 1: domain-adapting on LEDGAR...")
    ledgar_train = ledgar_dataset["train"]
    n_ledgar_labels = ledgar_train.features["label"].num_classes

    ledgar_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_ledgar_labels,
    ).to(device)

    def _tokenize_ledgar(batch: dict) -> dict:
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=512
        )

    ledgar_tok = ledgar_train.map(_tokenize_ledgar, batched=True)
    ledgar_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    from torch.utils.data import DataLoader as TorchDataLoader
    _pin = device.type == "cuda"
    ledgar_loader = TorchDataLoader(
        _LedgarDataset(ledgar_tok), batch_size=batch_size, shuffle=True,
        num_workers=(2 if _pin else 0), pin_memory=_pin, persistent_workers=_pin,
    )
    optimizer_p1 = torch.optim.AdamW(ledgar_model.parameters(), lr=learning_rate, weight_decay=0.01)
    ce_loss = torch.nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler_p1 = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, ledgar_epochs + 1):
        ledgar_model.train()
        total_loss, seen = 0.0, 0
        for bi, batch in enumerate(ledgar_loader):
            if ledgar_max_batches is not None and bi >= ledgar_max_batches:
                break
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optimizer_p1.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = ledgar_model(**inputs).logits
                loss = ce_loss(logits, labels)
            scaler_p1.scale(loss).backward()
            scaler_p1.step(optimizer_p1)
            scaler_p1.update()
            total_loss += float(loss.item())
            seen += 1
        print(f"  LEDGAR epoch {epoch}: loss={total_loss / max(1, seen):.4f}")

    # ── Phase 2: transfer backbone to CUAD ───────────────────────────────────
    print("Phase 2: transferring backbone to CUAD multi-label task...")
    # Extract backbone weights before freeing the LEDGAR model to save VRAM
    backbone_state = {
        k: v for k, v in ledgar_model.state_dict().items()
        if not k.startswith("classifier") and not k.startswith("pre_classifier")
    }

    del ledgar_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    label2id = {v: k for k, v in id_to_clause.items()}
    cuad_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id_to_clause),
        id2label=id_to_clause,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    missing, unexpected = cuad_model.load_state_dict(backbone_state, strict=False)
    print(f"  Transferred {len(backbone_state)} layers; "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    cuad_model = cuad_model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=(2 if _pin else 0), pin_memory=_pin, persistent_workers=_pin)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=(2 if _pin else 0), pin_memory=_pin, persistent_workers=_pin)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        cuad_model, train_loader, val_loader, train_examples, device,
        epochs=cuad_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_train_batches=cuad_max_train_batches,
        max_val_batches=cuad_max_val_batches,
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

    # Gradient checkpointing: recomputes activations on the backward pass instead
    # of storing them, trading ~15% extra compute for ~4x less activation memory.
    # Critical for Longformer at 4096 tokens — without this the T4/A100 runs OOM.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    _pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=(2 if _pin else 0), pin_memory=_pin, persistent_workers=_pin)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=(2 if _pin else 0), pin_memory=_pin, persistent_workers=_pin)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        model, train_loader, val_loader, train_examples, device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
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

    # Gradient checkpointing: same rationale as train_longformer_cuad — prevents
    # OOM on 4096-token sequences without changing the training methodology.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    _pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=(2 if _pin else 0), pin_memory=_pin, persistent_workers=_pin)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=(2 if _pin else 0), pin_memory=_pin, persistent_workers=_pin)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        model, train_loader, val_loader, train_examples, device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    print(f"Legal-BERT→Longformer → val micro_F1={metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name="Legal-BERT→Longformer (CUAD)",
        model=model, tokenizer=tokenizer,
        best_threshold=best_t, val_metrics=metrics, history=history,
        id_to_clause=id_to_clause, val_logits=val_logits, val_labels=val_labels,
    )


def train_longformer_ledgar_cuad(
    ledgar_dataset: Any,
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict],
    tokenizer: Any,
    id_to_clause: dict[int, str],
    val_examples: list[dict] | None = None,
    longformer_name: str = "allenai/longformer-base-4096",
    ledgar_epochs: int = 2,
    ledgar_max_batches: int | None = None,
    ledgar_batch_size: int = 32,
    cuad_epochs: int = 3,
    cuad_max_train_batches: int | None = None,
    cuad_max_val_batches: int | None = None,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    device: torch.device | None = None,
) -> ModelArtifacts:
    """Two-phase training: (1) fine-tune Longformer-base on LEDGAR, (2) transfer to CUAD.

    Unlike train_legalbert_longformer_cuad, this does NOT copy Legal-BERT weights with
    512→4096 position-embedding tiling. It domain-adapts Longformer's native position
    embeddings on LEDGAR, keeping all 4096 positions properly trained.
    Both phases use local-only attention (no global_attention_mask), consistent with
    train_longformer_cuad and the CUAD test evaluation in Section 4.
    """
    from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

    device = device or choose_device()
    _pin = device.type == "cuda"

    # ── Phase 1: LEDGAR fine-tuning ───────────────────────────────────────────
    print("Phase 1: domain-adapting Longformer on LEDGAR...", flush=True)
    ledgar_train = ledgar_dataset["train"]
    n_ledgar_labels = ledgar_train.features["label"].num_classes

    ledgar_model = AutoModelForSequenceClassification.from_pretrained(
        longformer_name, num_labels=n_ledgar_labels,
    ).to(device)
    ledgar_model.config.use_cache = False
    ledgar_model.gradient_checkpointing_enable()

    def _tokenize_ledgar(batch: dict) -> dict:
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=512
        )

    print(f"  Tokenizing {len(ledgar_train):,} LEDGAR examples...", flush=True)
    ledgar_tok = ledgar_train.map(_tokenize_ledgar, batched=True, num_proc=4)
    ledgar_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    print(f"  Tokenization complete.", flush=True)

    class _LFLedgarDataset(TorchDataset):
        def __init__(self, hf_ds):
            self.ds = hf_ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, i):
            item = self.ds[i]
            return {
                "input_ids":      item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels":         item["label"],
            }

    # ledgar_batch_size decouples Phase 1 batch size from Phase 2:
    # Phase 1 uses 512-token sequences so a much larger batch is safe on H100.
    ledgar_loader = TorchDataLoader(
        _LFLedgarDataset(ledgar_tok), batch_size=ledgar_batch_size, shuffle=True,
        num_workers=2, pin_memory=_pin, persistent_workers=True,
    )
    optimizer_p1 = torch.optim.AdamW(ledgar_model.parameters(), lr=learning_rate, weight_decay=0.01)
    ce_loss = torch.nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler_p1 = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, ledgar_epochs + 1):
        ledgar_model.train()
        total_loss, seen = 0.0, 0
        for bi, batch in enumerate(ledgar_loader):
            if ledgar_max_batches is not None and bi >= ledgar_max_batches:
                break
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optimizer_p1.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = ledgar_model(**inputs).logits
                loss = ce_loss(logits, labels)
            scaler_p1.scale(loss).backward()
            scaler_p1.step(optimizer_p1)
            scaler_p1.update()
            total_loss += float(loss.item())
            seen += 1
        print(f"  LEDGAR epoch {epoch}: loss={total_loss / max(1, seen):.4f}", flush=True)

    # ── Phase 2: transfer backbone to CUAD ───────────────────────────────────
    print("Phase 2: transferring Longformer backbone to CUAD multi-label task...", flush=True)
    backbone_state = {
        k: v for k, v in ledgar_model.state_dict().items()
        if not k.startswith("classifier")
    }
    del ledgar_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    label2id = {v: k for k, v in id_to_clause.items()}
    cuad_model = AutoModelForSequenceClassification.from_pretrained(
        longformer_name,
        num_labels=len(id_to_clause),
        id2label=id_to_clause,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    missing, unexpected = cuad_model.load_state_dict(backbone_state, strict=False)
    print(f"  Transferred {len(backbone_state)} layers; "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    cuad_model = cuad_model.to(device)
    cuad_model.config.use_cache = False
    cuad_model.gradient_checkpointing_enable()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=_pin, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=_pin, persistent_workers=True)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        cuad_model, train_loader, val_loader, train_examples, val_examples or [], device,
        epochs=cuad_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_train_batches=cuad_max_train_batches,
        max_val_batches=cuad_max_val_batches,
    )
    print(f"Longformer (LEDGAR→CUAD) → val micro_F1={metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name="Longformer (LEDGAR→CUAD)",
        model=model, tokenizer=tokenizer,
        best_threshold=best_t, val_metrics=metrics, history=history,
        id_to_clause=id_to_clause, val_logits=val_logits, val_labels=val_labels,
    )


class _LedgarDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]
        return {
            "input_ids":      item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels":         item["label"],
        }


# ═════════════════════════════════════════════════════════════════════════════
# Performance-tuning variants — CPU-friendly upgrades over train_tfidf_lr
# ═════════════════════════════════════════════════════════════════════════════


def _build_contract_matrix(df: pd.DataFrame, id_to_clause: dict[int, str]) -> tuple[list[str], np.ndarray]:
    """Shared helper: one row per contract, full text + multi-label vector."""
    name_to_id = {v: k for k, v in id_to_clause.items()}
    texts, label_rows = [], []
    for _, group in df.groupby("contract_title"):
        texts.append(group["contract_text"].iloc[0])
        row = np.zeros(len(id_to_clause), dtype=float)
        for r in group.itertuples(index=False):
            if r.has_answer and r.clause_type in name_to_id:
                row[name_to_id[r.clause_type]] = 1.0
        label_rows.append(row)
    return texts, np.array(label_rows)


class _HybridPipeline:
    """Like _TfIdfPipeline but supports a FeatureUnion (word + char TF-IDF)."""

    def __init__(self, vectorizer, estimators):
        self.vectorizer = vectorizer
        self.estimators_ = estimators

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return np.column_stack([_tfidf_proba_col(e, X) for e in self.estimators_])


def train_tfidf_lr_v2(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    id_to_clause: dict[int, str],
    ngram_word: tuple[int, int] = (1, 3),
    ngram_char: tuple[int, int] = (3, 5),
    max_features_word: int = 100_000,
    max_features_char: int = 50_000,
    tune_C: bool = False,
    calibrate: bool = False,
    artifact_name: str = "TF-IDF + LR (v2)",
    verbose: bool = True,
) -> ModelArtifacts:
    """Enhanced TF-IDF + LR: word+char n-grams, per-clause C tuning, optional calibration."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import FeatureUnion
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GridSearchCV

    train_texts, train_labels = _build_contract_matrix(train_df, id_to_clause)
    val_texts,   val_labels   = _build_contract_matrix(val_df,   id_to_clause)

    # Combined word + character n-gram features via FeatureUnion
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=ngram_word,
            max_features=max_features_word, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_char,
            max_features=max_features_char, sublinear_tf=True,
        )),
    ])
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)

    if verbose:
        print(f"Features: {X_train.shape[1]:,} (word+char)")

    C_grid = [0.1, 1.0, 10.0]

    estimators = []
    chosen_Cs: list[float | None] = []
    for i in range(train_labels.shape[1]):
        col = train_labels[:, i]
        if len(np.unique(col)) < 2:
            est = DummyClassifier(strategy="most_frequent")
            est.fit(X_train, col)
            estimators.append(est)
            chosen_Cs.append(None)
            continue

        # Stratified CV needs ≥ cv samples for BOTH classes (positive AND negative).
        # Common clauses (e.g. "Parties") appear in nearly every contract → n_neg is tiny.
        n_pos = int(col.sum())
        n_neg = int(len(col) - n_pos)
        min_class = min(n_pos, n_neg)
        cv_safe = min(3, min_class)   # 3 if both classes have ≥3, else 2, else 0

        # Only per-clause tune when CV is reliable: require ≥10 of the minority class
        # so each of 3 folds has ≥3 samples. Otherwise fall back to C=1.0 (tuned C grid
        # on tiny positive counts is pure noise — that was the V3 regression).
        if tune_C and min_class >= 10:
            base = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
            gs = GridSearchCV(base, {"C": C_grid}, cv=cv_safe, scoring="f1", n_jobs=1)
            gs.fit(X_train, col)
            best_C = float(gs.best_params_["C"])
        else:
            best_C = 1.0

        est = LogisticRegression(
            class_weight="balanced", max_iter=1000, C=best_C, solver="lbfgs",
        )

        if calibrate and cv_safe >= 2:
            est = CalibratedClassifierCV(est, cv=cv_safe, method="sigmoid")

        est.fit(X_train, col)
        estimators.append(est)
        chosen_Cs.append(best_C)

    # Collect class-1 probabilities; convert to log-odds for sigmoid-based evaluation
    eps = 1e-7
    val_probs = np.column_stack([_tfidf_proba_col(est, X_val) for est in estimators])
    p = np.clip(val_probs, eps, 1 - eps)
    val_logits = np.log(p / (1 - p))

    best_t, val_metrics = _tune_global_threshold(val_logits, val_labels)

    pipeline = _HybridPipeline(vectorizer, estimators)

    if verbose:
        chosen_summary = pd.Series([c for c in chosen_Cs if c is not None]).value_counts().to_dict() if tune_C else {1.0: sum(1 for c in chosen_Cs if c is not None)}
        print(f"{artifact_name} → val micro_F1={val_metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
        if tune_C:
            print(f"  per-clause C distribution: {chosen_summary}")

    return ModelArtifacts(
        model_name=artifact_name,
        model=pipeline,
        tokenizer=None,
        best_threshold=best_t,
        val_metrics=val_metrics,
        history=pd.DataFrame(),
        id_to_clause=id_to_clause,
        val_logits=val_logits,
        val_labels=val_labels,
    )


# ─── MiniLM embedding backbone ──────────────────────────────────────────────


class _EmbeddingPipeline:
    """Wraps a sentence-transformer encoder + per-label LR estimators.

    predict_proba(texts) returns class-1 probabilities (n_texts, n_labels).
    Embedding generation: split on "\\n\\n", encode each passage, mean-pool per contract.
    """

    def __init__(self, encoder_name: str, estimators: list, embeddings: np.ndarray | None = None):
        self.encoder_name = encoder_name
        self.estimators_ = estimators
        # Encoder is loaded lazily (avoids pickling torch modules into checkpoints)
        self._encoder = None
        self._train_embeddings = embeddings

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder

    def encode_contracts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        encoder = self._get_encoder()
        vectors = []
        for text in texts:
            passages = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not passages:
                passages = [text[:2000]]
            # Clip each passage to stay within MiniLM's 256-token window
            passages = [p[:1500] for p in passages]
            emb = encoder.encode(passages, batch_size=batch_size, show_progress_bar=False)
            vectors.append(emb.mean(axis=0))
        return np.vstack(vectors)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.encode_contracts(texts)
        return np.column_stack([_tfidf_proba_col(est, X) for est in self.estimators_])


def train_minilm_lr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    id_to_clause: dict[int, str],
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_path: str | None = "checkpoints/minilm_embeddings.npz",
    tune_C: bool = True,
    artifact_name: str = "MiniLM + LR",
    verbose: bool = True,
) -> ModelArtifacts:
    """Encode contracts with MiniLM (CPU-friendly) + per-label LR on 384-dim embeddings."""
    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import GridSearchCV

    train_texts, train_labels = _build_contract_matrix(train_df, id_to_clause)
    val_texts,   val_labels   = _build_contract_matrix(val_df,   id_to_clause)

    pipeline = _EmbeddingPipeline(encoder_name, estimators=[])

    cache_ok = False
    if cache_path and Path(cache_path).exists():
        try:
            cached = np.load(cache_path)
            X_train = cached["train"]
            X_val   = cached["val"]
            if X_train.shape[0] == len(train_texts) and X_val.shape[0] == len(val_texts):
                cache_ok = True
                if verbose:
                    print(f"Loaded cached MiniLM embeddings from {cache_path}")
        except Exception:
            cache_ok = False

    if not cache_ok:
        if verbose:
            print(f"Encoding {len(train_texts) + len(val_texts)} contracts with {encoder_name}…")
        X_train = pipeline.encode_contracts(train_texts)
        X_val   = pipeline.encode_contracts(val_texts)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, train=X_train, val=X_val)
            if verbose:
                print(f"Cached embeddings → {cache_path}")

    C_grid = [0.1, 1.0, 10.0]
    estimators = []
    for i in range(train_labels.shape[1]):
        col = train_labels[:, i]
        if len(np.unique(col)) < 2:
            est = DummyClassifier(strategy="most_frequent")
            est.fit(X_train, col)
        else:
            n_pos = int(col.sum())
            n_neg = int(len(col) - n_pos)
            min_class = min(n_pos, n_neg)
            cv_safe = min(3, min_class)
            if tune_C and min_class >= 10:
                base = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
                gs = GridSearchCV(base, {"C": C_grid}, cv=cv_safe, scoring="f1", n_jobs=1)
                gs.fit(X_train, col)
                best_C = float(gs.best_params_["C"])
            else:
                best_C = 1.0
            est = LogisticRegression(
                class_weight="balanced", max_iter=1000, C=best_C, solver="lbfgs",
            )
            est.fit(X_train, col)
        estimators.append(est)

    pipeline.estimators_ = estimators

    eps = 1e-7
    val_probs = np.column_stack([_tfidf_proba_col(est, X_val) for est in estimators])
    p = np.clip(val_probs, eps, 1 - eps)
    val_logits = np.log(p / (1 - p))

    best_t, val_metrics = _tune_global_threshold(val_logits, val_labels)

    if verbose:
        print(f"{artifact_name} → val micro_F1={val_metrics['micro_f1']:.4f}, threshold={best_t:.2f}")

    return ModelArtifacts(
        model_name=artifact_name,
        model=pipeline,
        tokenizer=None,
        best_threshold=best_t,
        val_metrics=val_metrics,
        history=pd.DataFrame(),
        id_to_clause=id_to_clause,
        val_logits=val_logits,
        val_labels=val_labels,
    )


# ─── Ensemble helper ────────────────────────────────────────────────────────


class _EnsemblePipeline:
    """Averages probabilities from two pipelines that each expose predict_proba."""

    def __init__(self, pipeline_a, pipeline_b, weight_a: float = 0.5):
        self.pipeline_a = pipeline_a
        self.pipeline_b = pipeline_b
        self.weight_a = weight_a

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        pa = self.pipeline_a.predict_proba(texts)
        pb = self.pipeline_b.predict_proba(texts)
        return self.weight_a * pa + (1.0 - self.weight_a) * pb


def ensemble_artifacts(
    artifacts_a: ModelArtifacts,
    artifacts_b: ModelArtifacts,
    weight_a: float | None = 0.5,
    artifact_name: str = "Ensemble (TF-IDF + MiniLM)",
    verbose: bool = True,
) -> ModelArtifacts:
    """Average two models' validation probabilities and re-tune threshold.

    If ``weight_a`` is None, sweep weights in {0.1, 0.3, 0.5, 0.7, 0.9} and pick the
    one that maximises micro-F1 on the validation set (one hyperparameter tuned on
    val — fair for a held-out test set later).

    Assumes both artifacts were evaluated on the same validation set in the same order.
    """
    from scipy.special import expit as sigmoid

    pa = sigmoid(artifacts_a.val_logits)
    pb = sigmoid(artifacts_b.val_logits)
    val_labels = artifacts_a.val_labels  # same val set
    eps = 1e-7

    def _logits_for(weight: float) -> np.ndarray:
        p_avg = weight * pa + (1.0 - weight) * pb
        p_clip = np.clip(p_avg, eps, 1 - eps)
        return np.log(p_clip / (1 - p_clip))

    if weight_a is None:
        candidates = [0.1, 0.3, 0.5, 0.7, 0.9]
        best_w, best_f1, best_logits, best_t, best_metrics = 0.5, -1.0, None, 0.5, {}
        for w in candidates:
            logits_w = _logits_for(w)
            t_w, metrics_w = _tune_global_threshold(logits_w, val_labels)
            if metrics_w["micro_f1"] > best_f1:
                best_w, best_f1, best_logits = w, metrics_w["micro_f1"], logits_w
                best_t, best_metrics = t_w, metrics_w
        weight_a = best_w
        val_logits = best_logits
        val_metrics = best_metrics
        best_t_out = best_t
        if verbose:
            print(f"  ensemble weight sweep → best weight_a={weight_a} (micro_F1={best_f1:.4f})")
    else:
        val_logits = _logits_for(weight_a)
        best_t_out, val_metrics = _tune_global_threshold(val_logits, val_labels)

    pipeline = _EnsemblePipeline(artifacts_a.model, artifacts_b.model, weight_a=weight_a)

    if verbose:
        print(f"{artifact_name} → val micro_F1={val_metrics['micro_f1']:.4f}, threshold={best_t_out:.2f}, weight_a={weight_a}")

    return ModelArtifacts(
        model_name=artifact_name,
        model=pipeline,
        tokenizer=None,
        best_threshold=best_t_out,
        val_metrics=val_metrics,
        history=pd.DataFrame(),
        id_to_clause=artifacts_a.id_to_clause,
        val_logits=val_logits,
        val_labels=val_labels,
    )


# ─── Early-fusion hybrid (TF-IDF ⊕ MiniLM embeddings) ──────────────────────


class _HybridFeaturePipeline:
    """Concatenates sparse TF-IDF features with dense MiniLM embeddings at inference.

    predict_proba(texts) -> (n_texts, n_labels) class-1 probabilities.
    Reuses an already-fit vectorizer and embedding encoder.
    """

    def __init__(self, vectorizer, embedding_pipeline: "_EmbeddingPipeline",
                 estimators: list, minilm_scale: float = 1.0):
        self.vectorizer = vectorizer
        self.embedding_pipeline = embedding_pipeline
        self.estimators_ = estimators
        self.minilm_scale = float(minilm_scale)

    def _transform(self, texts: list[str]):
        from scipy.sparse import hstack, csr_matrix
        X_tfidf = self.vectorizer.transform(texts)               # sparse
        X_emb   = self.embedding_pipeline.encode_contracts(texts)  # dense
        X_emb   = X_emb * self.minilm_scale
        return hstack([X_tfidf, csr_matrix(X_emb)]).tocsr()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self._transform(texts)
        return np.column_stack([_tfidf_proba_col(e, X) for e in self.estimators_])


def train_hybrid_features_lr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    id_to_clause: dict[int, str],
    ngram_word: tuple[int, int] = (1, 3),
    ngram_char: tuple[int, int] = (3, 5),
    max_features_word: int = 100_000,
    max_features_char: int = 50_000,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_path: str | None = "checkpoints/minilm_embeddings.npz",
    tune_C: bool = False,
    minilm_scale: float = 1.0,
    artifact_name: str = "V7 — Hybrid features (TF-IDF ⊕ MiniLM)",
    verbose: bool = True,
) -> ModelArtifacts:
    """Early-fusion: concatenate TF-IDF (word+char) sparse features with dense MiniLM
    embeddings and train per-label LR on the joint matrix.

    Usually beats late-fusion (probability averaging) when one model (MiniLM here) is
    much weaker — the LR learns to weight each feature source per-label rather than
    forcing a single global weight_a.
    """
    from pathlib import Path
    from scipy.sparse import hstack, csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import FeatureUnion
    from sklearn.model_selection import GridSearchCV

    train_texts, train_labels = _build_contract_matrix(train_df, id_to_clause)
    val_texts,   val_labels   = _build_contract_matrix(val_df,   id_to_clause)

    # 1) TF-IDF (word + char)
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=ngram_word,
            max_features=max_features_word, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_char,
            max_features=max_features_char, sublinear_tf=True,
        )),
    ])
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_val_tfidf   = vectorizer.transform(val_texts)

    # 2) MiniLM embeddings (cached npz: {"train": ..., "val": ...})
    emb_pipeline = _EmbeddingPipeline(encoder_name, estimators=[])
    cache_ok = False
    if cache_path and Path(cache_path).exists():
        try:
            cached = np.load(cache_path)
            X_train_emb, X_val_emb = cached["train"], cached["val"]
            if (X_train_emb.shape[0] == len(train_texts)
                and X_val_emb.shape[0] == len(val_texts)):
                cache_ok = True
                if verbose:
                    print(f"Loaded cached MiniLM embeddings from {cache_path}")
        except Exception:
            cache_ok = False
    if not cache_ok:
        if verbose:
            print(f"Encoding {len(train_texts) + len(val_texts)} contracts with {encoder_name}…")
        X_train_emb = emb_pipeline.encode_contracts(train_texts)
        X_val_emb   = emb_pipeline.encode_contracts(val_texts)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, train=X_train_emb, val=X_val_emb)

    # 3) Concatenate. Scale MiniLM features so L2-regularized LR doesn't drown them
    # under 150k sparse TF-IDF dims (each MiniLM coefficient pays the same regularization
    # cost as each TF-IDF coefficient; scaling features up lets the LR choose to use them).
    X_train_emb_scaled = X_train_emb * float(minilm_scale)
    X_val_emb_scaled   = X_val_emb   * float(minilm_scale)
    X_train = hstack([X_train_tfidf, csr_matrix(X_train_emb_scaled)]).tocsr()
    X_val   = hstack([X_val_tfidf,   csr_matrix(X_val_emb_scaled)]).tocsr()

    if verbose:
        print(f"Hybrid features: {X_train.shape[1]:,} (TF-IDF {X_train_tfidf.shape[1]:,} + MiniLM {X_train_emb.shape[1]} × {minilm_scale})")

    # 4) Per-label LR (same safe-CV logic as v2)
    C_grid = [0.1, 1.0, 10.0]
    estimators = []
    for i in range(train_labels.shape[1]):
        col = train_labels[:, i]
        if len(np.unique(col)) < 2:
            est = DummyClassifier(strategy="most_frequent")
            est.fit(X_train, col)
            estimators.append(est)
            continue

        n_pos = int(col.sum())
        n_neg = int(len(col) - n_pos)
        min_class = min(n_pos, n_neg)
        cv_safe = min(3, min_class)

        if tune_C and min_class >= 10:
            base = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
            gs = GridSearchCV(base, {"C": C_grid}, cv=cv_safe, scoring="f1", n_jobs=1)
            gs.fit(X_train, col)
            best_C = float(gs.best_params_["C"])
        else:
            best_C = 1.0

        est = LogisticRegression(
            class_weight="balanced", max_iter=1000, C=best_C, solver="lbfgs",
        )
        est.fit(X_train, col)
        estimators.append(est)

    # 5) Probabilities → log-odds for sigmoid-based evaluators
    eps = 1e-7
    val_probs = np.column_stack([_tfidf_proba_col(est, X_val) for est in estimators])
    p = np.clip(val_probs, eps, 1 - eps)
    val_logits = np.log(p / (1 - p))

    best_t, val_metrics = _tune_global_threshold(val_logits, val_labels)

    # Wire up inference pipeline; emb_pipeline needs estimators=[] (unused) but a live encoder
    pipeline = _HybridFeaturePipeline(vectorizer, emb_pipeline, estimators, minilm_scale=minilm_scale)

    if verbose:
        print(f"{artifact_name} → val micro_F1={val_metrics['micro_f1']:.4f}, threshold={best_t:.2f}")

    return ModelArtifacts(
        model_name=artifact_name,
        model=pipeline,
        tokenizer=None,
        best_threshold=best_t,
        val_metrics=val_metrics,
        history=pd.DataFrame(),
        id_to_clause=id_to_clause,
        val_logits=val_logits,
        val_labels=val_labels,
    )
