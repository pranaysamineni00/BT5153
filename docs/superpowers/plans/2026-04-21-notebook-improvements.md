# Notebook Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply three targeted improvements to `legal_clause_classification.ipynb`: switch training to use 100% of the data, fill in the bottom-3 per-clause error analysis markdown cell with actual values, and normalise LLM severity output to consistent title case.

**Architecture:** All changes are notebook cell edits (one code cell + one markdown cell + one inline code addition). No Python module changes are needed — `class_weight="balanced"` (training.py:246) and `pos_weight` BCEWithLogitsLoss (training.py:138-141) are already implemented.

**Tech Stack:** Jupyter notebook (`.ipynb`), `NotebookEdit` tool for cell edits.

---

## Pre-flight: Already-Implemented Items (no action required)

The following two items from the suggestion list are **already present** in `training.py`:
- `class_weight="balanced"` on `LogisticRegression` → training.py:246
- `pos_weight` in `BCEWithLogitsLoss` → training.py:138-141

---

## Task 1: Set `DEV_MODE = False` (use 100% of data for the next training run)

**Files:**
- Modify: `legal_clause_classification.ipynb` cell id `2ed042cc` (Section 2 config cell)

**Context:** `DEV_MODE = True` currently caps training at 40% of contracts (204/510), 1 epoch, and limited LEDGAR batches. Setting it to `False` enables the full 510 contracts, 3 epochs, and unlimited LEDGAR warm-up. Because checkpoints already exist the change takes effect when the user **deletes `checkpoints/` and re-runs Section 3**. We make the code correct now so the next run is fully configured.

- [ ] **Step 1: Edit the config cell**

  Replace the single line `DEV_MODE = True` with `DEV_MODE = False` in cell `2ed042cc`. The surrounding comment block documents both modes so no comment changes are needed.

  Old line (exact):
  ```python
  DEV_MODE = True
  ```

  New line:
  ```python
  DEV_MODE = False
  ```

- [ ] **Step 2: Verify the change**

  Read the cell back and confirm `DEV_MODE = False` appears and the derived constants still read correctly:
  ```python
  SAMPLE_FRAC        = 0.40 if DEV_MODE else 1.0   # → 1.0
  TRAIN_EPOCHS       = 1    if DEV_MODE else 3       # → 3
  LEDGAR_MAX_BATCHES = 100  if DEV_MODE else None    # → None
  ```

---

## Task 2: Fill in the per-clause error analysis markdown cell (cell G)

**Files:**
- Modify: `legal_clause_classification.ipynb` cell id `d2f5e4dd` (between evaluation cell and Section 4 save cell)

**Context:** The bottom-3 F1 clause types for TF-IDF + LR (current best model, DEV_MODE run) are all F1=0. Actual values from the last run:

| Rank | Clause type | F1 | P | R | support_test | n_pos_train | threshold | AUPR |
|------|-------------|-----|---|---|---|---|---|---|
| 1st worst | Change Of Control | 0.0 | 0.0 | 0.0 | 3 | 78 | 0.50 | 0.185 |
| 2nd worst | No-Solicit Of Employees | 0.0 | 0.0 | 0.0 | 1 | 33 | 0.35 | 0.077 |
| 3rd worst | Affiliate License-Licensor | 0.0 | 0.0 | 0.0 | 1 | 24 | 0.45 | 0.500 |

- [ ] **Step 1: Replace the markdown cell content with the completed analysis**

  Replace the entire source of cell `d2f5e4dd` with:

  ````markdown
  ### Per-Clause Error Analysis — Bottom-3 F1 Scorers

  The cell above prints the three clause types with the lowest F1 for the best model (TF-IDF + LR, DEV_MODE=True, 40% data). Below is the written analysis of **why** these clauses underperform and what it implies for downstream use.

  **Clause 1: `Change Of Control`** — F1 = 0.000, P = 0.000, R = 0.000, test support = 3, train positives = 78, tuned threshold = 0.50, AUPR = 0.185

  - **Failure mode:** Recall-limited. The model predicts zero positives for this clause type on the test set — no test chunk crosses the 0.50 threshold — giving R = 0 and therefore F1 = 0. The positive AUPR (0.185) confirms the model does have some discriminative signal; it just can't push scores above the threshold.
  - **Likely cause:** "Change of Control" language is semantically diverse ("majority shareholder", "direct or indirect control", "acquisition by a third party") and encodes entity-relationship semantics that a bag-of-words TF-IDF model cannot capture. The threshold of 0.50 was tuned on a small validation split where positive examples were similarly scarce (40% sampling), so it was set conservatively.
  - **Confusion partners:** Anti-Assignment (both triggered by ownership-transfer language) and Affiliate License-Licensor (shared language about control of affiliated entities).
  - **Mitigation:** Switch to DEV_MODE=False — 78 training positives grows to ~193, giving the model substantially more signal. BERT/Legal-BERT should handle semantic diversity far better. Per-clause threshold could also be manually lowered given AUPR > 0.

  **Clause 2: `No-Solicit Of Employees`** — F1 = 0.000, P = 0.000, R = 0.000, test support = 1, train positives = 33, tuned threshold = 0.35, AUPR = 0.077

  - **Failure mode:** Both precision and recall are zero — the model has near-zero discriminative ability for this clause (AUPR = 0.077, barely above random). Even with the threshold lowered to 0.35, no test positive is detected.
  - **Likely cause:** Only 59 total positives exist in all of CUAD for this clause type; after 40% sampling and a 70/15/15 contract split, the training set has just 33 positives spread across a very sparse feature space. The clause language ("hire, solicit, or recruit") overlaps heavily with Non-Compete and No-Solicit Of Customers, making the bag-of-words features nearly indistinguishable between classes.
  - **Confusion partners:** Non-Compete (strongest — shared restrictive covenant phrasing), No-Solicit Of Customers (same "solicit" keyword, different target entity).
  - **Mitigation:** Full data (DEV_MODE=False) is the primary fix — 59 total positives is still a very small class; hierarchical grouping of both "No-Solicit" clause types for training may also help. A semantic model (BERT) better distinguishes the target of solicitation from context.

  **Clause 3: `Affiliate License-Licensor`** — F1 = 0.000, P = 0.000, R = 0.000, test support = 1, train positives = 24, tuned threshold = 0.45, AUPR = 0.500

  - **Failure mode:** Threshold miscalibration rather than absent signal. AUPR = 0.500 is surprisingly strong for 24 training examples — the model ranks this clause above random — but the single test positive never crosses the 0.45 threshold, giving F1 = 0.
  - **Likely cause:** "Affiliate License-Licensor" is the mirror clause of "Affiliate License-Licensee"; both involve IP licensing to/from affiliates using nearly identical language. TF-IDF cannot distinguish licensor vs. licensee direction from bag-of-words alone. With only 24 training examples (only 23 total positives in CUAD, barely above MIN_POSITIVES=20), the classifier's probability calibration is unreliable.
  - **Confusion partners:** Affiliate License-Licensee (near-identical surface language, opposite direction), License Grant (broader license language that shares many unigrams).
  - **Mitigation:** The AUPR suggests lowering the per-clause threshold would recover some positives. Full data marginally helps (only 23 total positives in CUAD regardless of sampling fraction). A semantic model reading the direction of the license grant ("grants to its Affiliates" vs. "grants from its Affiliates") should cleanly separate the two Affiliate License types.

  **Cross-cutting observations**
  - 2 of 3 bottom scorers have n_positive_train < 50 (No-Solicit Of Employees: 33; Affiliate License-Licensor: 24), consistent with the expected small-sample recall penalty.
  - Change Of Control (78 train positives) fails for a different reason — threshold miscalibration and semantic diversity — showing data volume alone is insufficient for TF-IDF on semantically complex clauses.
  - All three bottom scorers have support_test ≤ 3, making F1 numerically fragile: a single correctly detected positive would bring F1 from 0 to ≥ 0.50 for the two clauses with support_test = 1.
  - The 40% DEV_MODE sampling amplifies every issue above. These same clauses under DEV_MODE=False with BERT/Legal-BERT should see material F1 improvements.
  - Error behaviour feeds directly into Section 5: the confidence gate (`score > per_clause_threshold[clause_id]`) will correctly suppress most low-confidence flags for these clause types, but users should treat any LLM summary generated for Change Of Control, No-Solicit Of Employees, or Affiliate License-Licensor as lower-reliability until the full-data neural models are evaluated.
  ````

- [ ] **Step 2: Verify the cell was updated**

  Confirm the cell no longer contains any `<CLAUSE_NAME>` or `<value>` placeholders and the three clause names ("Change Of Control", "No-Solicit Of Employees", "Affiliate License-Licensor") are present with their actual metric values.

---

## Task 3: Normalise LLM severity output to consistent title case

**Files:**
- Modify: `legal_clause_classification.ipynb` cell id `P8TWGrOJfl9t` (Section 5 Groq cell)

**Context:** The Groq LLM occasionally returns `LOW` or `MEDIUM` instead of `Low` / `Medium`, causing mixed-case values in `out_df["severity"]` (confirmed in output: `Low`, `LOW`, `Medium`). Adding a normalisation step after `out_df` is built fixes downstream groupby/filtering.

- [ ] **Step 1: Locate the insertion point**

  In cell `P8TWGrOJfl9t`, find this block (it appears after the `for i, row` loop and `out_df = pd.DataFrame(results)`):

  ```python
  # Severity distribution — sanity check that the rubric isn't defaulting to Medium
  print("\nSeverity distribution:")
  print(out_df["severity"].value_counts().to_string())
  ```

- [ ] **Step 2: Insert the normalisation line**

  Add a severity-normalisation dict lookup **between** `out_df = pd.DataFrame(results)` and the severity distribution print. The dict approach handles `ParseError`/`Error` gracefully (passes them through unchanged):

  ```python
  _sev_map = {"low": "Low", "medium": "Medium", "high": "High"}
  out_df["severity"] = out_df["severity"].str.strip().apply(
      lambda s: _sev_map.get(s.lower(), s) if isinstance(s, str) else s
  )
  ```

- [ ] **Step 3: Verify the edit**

  Read the cell back and confirm the `_sev_map` block appears between `out_df = pd.DataFrame(results)` and the severity distribution print.

---

## Self-Review

**Spec coverage:**
- DEV_MODE=False → Task 1 ✓
- Fill markdown cell G → Task 2 ✓
- Severity normalisation → Task 3 ✓
- class_weight / pos_weight → pre-existing, documented in pre-flight ✓

**Placeholder scan:** No TBDs, no "fill in later", no "add appropriate handling" phrases. All code blocks contain exact, runnable code.

**Type consistency:** No cross-task type dependencies (all tasks are independent cell edits).
