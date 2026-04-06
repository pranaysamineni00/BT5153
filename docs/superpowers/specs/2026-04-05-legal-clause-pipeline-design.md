# Fineprint: Legal Clause Classification Pipeline — Design Spec
Date: 2026-04-05

## Scope
Steps 1–5 of the BT5153 project spec. Step 6 (UI) deferred to a later session.

## File Structure

```
legal_clause_classification/
├── pipeline.ipynb          # unified orchestration notebook (Colab-compatible)
├── data_loading.py         # Step 1: CUAD + LEDGAR loading
├── preprocessing.py        # Step 2: EDA, filtering, chunking, splits
├── training.py             # Step 3: all 5 model training loops
├── evaluation.py           # Step 4: metrics, confusion matrix, per-clause analysis
└── data/
    ├── cuad/               # existing
    └── ledgar/             # new
```

`cuad_chunk_multilabel.py` is superseded — its logic migrates into `preprocessing.py` and `training.py`.

---

## Module Designs

### data_loading.py
- `load_cuad(data_dir)` → `pd.DataFrame` (existing logic from notebook cells 2–4, extracted to function)
- `load_ledgar()` → HuggingFace `datasets.DatasetDict` via `coastalcph/lex_glue`, config `ledgar`
- Both functions print confirmation stats (num contracts, num examples, num categories)

### preprocessing.py
Replaces `cuad_chunk_multilabel.py`. All existing functions migrate here with these additions/changes:

**EDA:**
- `plot_clause_frequency(cuad_df)` → bar chart of positive rate per clause type (41 bars)

**Clause filtering:**
- `filter_clauses(cuad_df, min_positives=20)` → drops clause types below threshold, returns filtered df + exclusion log. Target set per spec: non-compete, indemnification, limitation of liability, auto-renewal, termination for convenience, governing law, IP assignment, confidentiality, dispute resolution, force majeure.

**Chunking:**
- `build_chunk_examples(...)` — updated: BERT-family uses `max_length=512, stride=128`; Longformer uses `max_length=4096, stride=512`
- `compute_sample_weights(chunk_examples)` → returns per-sample float weight array; all-negative chunks get weight=0.1 (feature-level downweighting per CUAD paper), positive chunks get weight=1.0
- `compute_pos_weight(chunk_examples)` → existing, unchanged

**Aggregation:**
- `aggregate_contract_predictions(chunk_predictions_df)` → max-probability rollup per clause across all chunks of a contract
- `build_chunk_to_contract_map(chunk_examples)` → dict mapping chunk_index → contract_title

**Splits:** unchanged 80/10/10 contract-level split.

### training.py
All functions return a common `ModelArtifacts` dataclass:
```python
@dataclass
class ModelArtifacts:
    model_name: str          # human label, e.g. "Legal-BERT"
    model: Any               # trained model or sklearn pipeline
    tokenizer: Any           # None for TF-IDF
    best_threshold: float    # global threshold (per-clause tuning happens in evaluation.py)
    val_metrics: dict        # micro/macro F1 on val set
    history: pd.DataFrame    # epoch-level training log (empty for TF-IDF)
    id_to_clause: dict
```

**Five training functions:**

1. `train_tfidf_lr(train_df, val_df, id_to_clause)`:
   - TF-IDF vectorizer (max_features=50000, ngram_range=(1,2))
   - `MultiOutputClassifier(LogisticRegression(class_weight='balanced', max_iter=1000))`
   - Input: contract-level (not chunk-level), full contract text

2. `train_bert_ledgar_cuad(ledgar_dataset, train_dataset, val_dataset, ...)`:
   - Phase 1: fine-tune `bert-base-uncased` on LEDGAR (multi-class, 100 categories, 3 epochs)
   - Phase 2: strip classification head, attach new multi-label head (N_target_clauses output)
   - Phase 3: fine-tune on CUAD chunks with BCEWithLogitsLoss + pos_weight + sample_weights

3. `train_bert_cuad(train_dataset, val_dataset, ...)`:
   - `bert-base-uncased` directly on CUAD chunks, same loss setup

4. `train_legal_bert_cuad(train_dataset, val_dataset, ...)`:
   - `nlpaueb/legal-bert-base-uncased` on CUAD chunks

5. `train_longformer_cuad(train_dataset, val_dataset, ...)`:
   - `allenai/longformer-base-4096` on CUAD (4096 token chunks, stride 512)
   - Uses global attention on [CLS] token

6. `train_legalbert_longformer_cuad(train_dataset, val_dataset, ...)`:
   - Initialize Longformer architecture from Legal-BERT weights (weight mapping per Mamakas et al. 2022)
   - Fine-tune on CUAD with same setup as Longformer

**Shared training config:** AdamW, lr=2e-5, weight_decay=0.01, warmup_ratio=0.1, 3 epochs, batch_size=8, BCEWithLogitsLoss with pos_weight, sample_weight applied via manual loss masking.

### evaluation.py

**Per-clause threshold tuning:**
- `tune_per_clause_thresholds(logits, labels, thresholds=np.arange(0.1, 0.91, 0.05))` → dict[clause_name → best_threshold], optimising per-clause F1 on val logits

**Metrics per model:**
- `compute_per_clause_metrics(logits, labels, thresholds, id_to_clause)` → DataFrame with columns: clause_type, precision, recall, F1, precision_at_80_recall, AUPR, n_positive_train
- `compute_aggregate_metrics(logits, labels, thresholds)` → macro F1, micro F1, overall precision, recall
- `precision_at_recall_threshold(y_true, y_score, recall_target=0.80)` → scalar

**Visualisations:**
- `plot_confusion_matrix(logits, labels, thresholds, id_to_clause)` → heatmap of predicted vs true clause co-occurrences (multi-label: rows = true positives, cols = predicted)
- `plot_model_comparison(results_dict)` → grouped bar chart, all 5 models × clause types

---

## pipeline.ipynb — Cell Structure

```
[Section 0]  Install & imports
[Section 1]  Load CUAD + LEDGAR  (data_loading.py)
[Section 2]  EDA + clause filtering + chunking  (preprocessing.py)
[Section 3a] Train TF-IDF + LR
[Section 3b] Train BERT (LEDGAR → CUAD)
[Section 3c] Train BERT (CUAD only)
[Section 3d] Train Legal-BERT (CUAD)
[Section 3e] Train Longformer (CUAD)
[Section 3f] Train Legal-BERT → Longformer (CUAD)
[Section 4]  Evaluate all 5 models, per-clause metrics, confusion matrices
[Section 5]  LLM risk summarisation (Step 5 — Claude API via Colab secrets)
```

---

## Key Decisions

- **Token lengths:** 512/stride 128 for BERT-family; 4096/stride 512 for Longformer
- **Sample downweighting:** all-negative chunks get loss weight=0.1; implemented by multiplying per-sample BCEWithLogitsLoss by weight tensor
- **Per-clause thresholds:** tuned on validation logits independently per clause, reported alongside results
- **LEDGAR adaptation:** full fine-tune on 60k examples, head stripped before CUAD phase
- **Legal-BERT → Longformer weight init:** shared encoder layers copied by name; position embeddings extended via interpolation
- **Colab compatibility:** all secrets (Claude API key) via `google.colab.userdata`; `.py` files uploaded alongside notebook
- **Existing notebook cells 0–14 (EDA):** kept as-is, new cells added from Section 1 onward
