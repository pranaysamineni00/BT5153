# LexScan

LexScan is a legal contract review dashboard that helps a user upload a contract, detect important clauses, review the risk level of each clause, inspect supporting evidence, generate a plain-English summary, and ask follow-up questions through a guardrailed RAG chatbot.

The app is built to be useful even when every AI layer is not available. If a trained classifier checkpoint is missing, it falls back to a lighter baseline or keyword heuristics. If OpenAI is not configured, the core clause detection still works, the summary / clause explanation / chatbot features stay off, and the second-stage review falls back to a non-LLM `LLM_SKIPPED` path instead of crashing the workflow.

## What the app does

At a high level, the app supports these workflows:

1. Upload a contract in `PDF`, `DOCX`, `DOC`, or `TXT`.
2. Extract the contract text and classify legal clauses from the document.
3. Assign clause categories and risk levels such as `HIGH`, `MEDIUM`, or `LOW`.
4. Run a second-stage review that uses retrieval and LLM checks to validate or correct the original clause label.
5. Show a plain-English contract summary for non-lawyers.
6. Explain a selected clause in simple terms.
7. Let the user ask contract-specific questions through a floating chatbot grounded in the uploaded contract.

## Main features

### 1. Contract upload and text extraction

The dashboard accepts the most common contract formats:

- `PDF`
- `DOCX`
- `DOC`
- `TXT`

The backend saves the uploaded file to a temporary location, extracts text, and removes the temporary file immediately after processing.

Techniques used:

- `pypdf` or `PyPDF2` for PDF extraction
- `python-docx` for Word documents
- multi-encoding fallback for text files using `utf-8`, `latin-1`, and `cp1252`
- Flask file upload handling with a strict extension allowlist

Why this matters:

- contracts often arrive in inconsistent formats
- extraction failures are common in document tools, so the app uses multiple fallbacks before giving up

### 2. Clause detection with multiple reliability modes

The core task of the app is to detect important legal clauses from a contract.

The classifier supports three operating modes:

1. `model`: a fine-tuned Legal-BERT style model if the checkpoint is available
2. `baseline`: a TF-IDF + logistic regression baseline if the BERT checkpoint is missing
3. `heuristic`: regex and keyword rules if no trained artifacts are present

By default, the app now prefers the TF-IDF baseline over regex-only fallback, and baseline thresholds are slightly relaxed so borderline real-world detections can still surface for the second-stage review to inspect.

Techniques used:

- multi-label clause classification over the CUAD clause set
- sliding-window inference for long contracts in model mode
- probability calibration and per-clause thresholds
- keyword and regex heuristics as a safe fallback
- pre-defined clause metadata such as risk level and business category

Why this matters:

- legal contracts are long, so the model cannot always read the full document in one pass
- multi-label classification is needed because a single contract can contain many clauses
- fallback modes keep the app usable in lightweight local setups

### 3. Risk tagging and clause grouping

After clauses are detected, the app adds:

- a clause category such as `Financial & Liability` or `IP & Licensing`
- a risk level such as `HIGH`, `MEDIUM`, or `LOW`
- a short `why this risk` explanation and a `watch out for` note for the clause detail view

When multiple clause labels point to the same supporting passage, the UI groups them together so the user does not see a stack of duplicate evidence cards.

Techniques used:

- rule-based mapping from clause name to category
- a hybrid risk tagger: snippet-level rules first, with an LLM fallback only for ambiguous or higher-priority clauses
- evidence-key grouping so clauses with the same supporting snippet are merged in the UI

Why this matters:

- business users usually care first about risk and relevance, not raw model outputs
- grouping repeated evidence makes the results easier to scan

### 4. Excerpt extraction for evidence view

Each clause card can show the part of the contract that supports the prediction.

The system tries several strategies so the evidence panel is rarely blank.

Techniques used:

- optional LLM-based excerpt extraction
- keyword-based snippet finding
- term-density fallback when exact keyword matching is weak
- first-substantive-paragraph fallback when no stronger excerpt can be localized

Why this matters:

- users need to see the actual contract text, not just a label
- evidence-first UX improves trust and makes manual review faster

### 5. Second-stage review pipeline

This is one of the most important features in the app.

The first-stage classifier proposes clause labels. Then a second-stage review checks whether the predicted clause is truly present and whether the chosen label is the best one.

The second-stage review can:

- accept the original label
- reject the label
- rerank the label to a better clause
- escalate the case to human review

If the OpenAI review layer is unavailable, the pipeline still returns structured review output, but marks the item as `LLM_SKIPPED` instead of running the full presence and support checks.

Techniques used:

- retrieval-augmented review over the `CUAD` training split only
- retrieval of positive examples for the predicted clause
- retrieval of hard negative examples for confusing neighboring clauses
- clause-definition lookup from the RAG index
- two strict JSON LLM agents:
  - `ClausePresenceAgent`
  - `EvidenceSupportAgent`
- deterministic final routing in [`decision_router.py`](/Users/saiashwin/BT5153/decision_router.py)

How the review pipeline works:

1. The classifier predicts a clause label.
2. The pipeline retrieves candidate labels and supporting examples from the CUAD-based RAG index.
3. The presence agent checks whether the target clause is actually present in the uploaded snippet.
4. The support agent checks whether the snippet supports the predicted label or whether another label fits better.
5. The deterministic router converts those signals into one final action: `ACCEPT`, `REJECT`, `RERANK_LABEL`, or `HUMAN_REVIEW`.

Why this matters:

- first-stage classifiers can overpredict or confuse nearby clause types
- retrieved positive and negative examples make the review step more grounded
- deterministic routing keeps the final decision easy to inspect and debug

### 6. Retrieval layer for second-stage review

The second-stage review depends on a dedicated retrieval system built from the CUAD training split.

Techniques used:

- a cached RAG index stored in `checkpoints/rag_index.joblib`
- either TF-IDF embeddings or sentence-transformer embeddings
- optional FAISS for dense vector retrieval
- sparse TF-IDF storage to reduce memory pressure
- semantic neighbor maps between clause labels
- confusion maps derived from validation-time model errors
- manual hard-negative mappings for commonly confused clauses

Why this matters:

- legal clauses are semantically close to each other
- confusion-aware negatives help the second-stage review reason about edge cases
- caching avoids rebuilding retrieval artifacts every time the app starts

### 7. Plain-English contract summary

The app can generate a plain-English summary meant for non-lawyers.

The summary covers:

- document type
- key obligations
- payment or duration if present
- termination conditions
- unusual or one-sided terms

Techniques used:

- OpenAI-powered two-step summary flow:
  1. identify the contract type
  2. summarize the contract in 4-6 bullets
- fallback direct prompt if the two-step flow fails
- bullet normalization logic so the UI gets clean structured summary points

Why this matters:

- many users want a fast overview before reading clause-by-clause details
- separating document-type detection from summarization usually produces cleaner summaries

### 8. Clause explainer

When a user selects a clause excerpt, the app can explain it in simpler business language.

Techniques used:

- OpenAI chat completion with a prompt tuned for non-lawyers
- excerpt trimming so the prompt stays short and focused
- short-answer generation capped for readability

Why this matters:

- a clause label alone is not enough for many users
- this feature translates legal wording into direct business meaning

### 9. Guardrailed RAG chatbot

The app includes a floating chatbot in the bottom-right corner of the dashboard.

The chatbot is designed to answer questions about the uploaded contract only. It is not a general chatbot.

What the chatbot supports:

- obligations and responsibilities
- payment terms
- termination rights
- clause meaning
- contract risk questions
- follow-up questions on the same uploaded document

What the chatbot refuses:

- jokes
- weather
- general world knowledge
- unrelated personal or professional advice

Its refusal style is intentionally explicit:

`I am just a legal contract support agent, I can't answer this.`

Techniques used:

- per-document RAG index built from the uploaded contract text
- paragraph-aware chunking with overlap
- TF-IDF by default, with optional sentence-transformer embeddings
- cosine-similarity style retrieval over contract chunks
- short per-document conversation memory
- clause alias expansion so questions like "limitation of liability" can map to internal clause labels
- contract-only evidence citations in each answer
- optional CUAD examples as background support only, never as direct contract evidence
- guardrails for out-of-scope questions and low-evidence answers
- suggested starter questions generated from detected clauses and summary metadata

Special chatbot behavior:

- the chat window starts as a round launcher icon
- the panel can be minimized and reopened
- the assistant shows a typing `...` indicator while generating a reply
- some clause-guidance questions can still be answered even when document evidence is thin, as long as the question is clearly about a detected contract clause and the answer stays framed as general clause guidance

Why this matters:

- users naturally want to ask follow-up questions instead of reading a static report
- contract-grounded retrieval reduces hallucination risk
- strict guardrails stop the chatbot from drifting into unrelated topics

### 10. Suggested contract questions

After a document is analyzed, the chatbot shows 3-4 suggested questions.

These include:

- generic prompts such as obligations or termination risk
- contract-specific prompts derived from the detected clauses

Techniques used:

- deterministic prompt generation from classification results
- summary-aware prompt hints, especially for licensing contracts

Why this matters:

- suggested queries help users discover what the chatbot is good at
- deterministic generation avoids spending extra tokens on UI suggestions

### 11. Frontend dashboard and review workflow

The UI is a single-page dashboard served by Flask from [`static/index.html`](/Users/saiashwin/BT5153/static/index.html).

What the UI includes:

- file upload panel
- clause counts by risk
- clause review list
- detailed evidence pane
- second-stage review statistics
- contract summary panel
- floating chatbot widget

Techniques used:

- server-rendered static frontend with plain HTML, CSS, and JavaScript
- asynchronous fetch calls to Flask API endpoints
- optimistic chat UX with a temporary typing indicator
- card-based drill-down for clause evidence and review details
- filterable second-stage review results

Why this matters:

- the frontend is lightweight and easy to run locally
- the workflow is built around "scan first, inspect deeper when needed"

### 12. Reliability and performance safeguards

This app includes several practical safeguards for local development.

Techniques used:

- optional `.env` loading
- in-memory LRU cache for uploaded documents and chat state
- lazy construction of the chatbot document index
- memory logging to `dashboard_server.log`
- thread-count environment limits for BLAS and tokenizers
- Waitress in local production-style mode, with fallback to Flask's built-in server
- restart helper in [`run_dashboard.py`](/Users/saiashwin/BT5153/run_dashboard.py)

Why this matters:

- ML + retrieval + LLM apps can be memory-heavy
- lazy loading helps the app stay responsive on laptops

## Architecture overview

The current app is organized roughly like this:

- [`app.py`](/Users/saiashwin/BT5153/app.py): Flask server, routes, caching, file handling, and app startup
- [`classifier.py`](/Users/saiashwin/BT5153/classifier.py): clause detection logic, fallback modes, metadata, and excerpt extraction
- [`review_pipeline.py`](/Users/saiashwin/BT5153/review_pipeline.py): second-stage review orchestration
- [`agents.py`](/Users/saiashwin/BT5153/agents.py): retrieval and strict JSON review agents
- [`decision_router.py`](/Users/saiashwin/BT5153/decision_router.py): deterministic final review routing
- [`retriever.py`](/Users/saiashwin/BT5153/retriever.py): retrieval helpers for positives, negatives, and clause definitions
- [`rag_index.py`](/Users/saiashwin/BT5153/rag_index.py): CUAD-based RAG index build and cache logic
- [`document_rag.py`](/Users/saiashwin/BT5153/document_rag.py): per-document chunking and retrieval for chatbot answers
- [`contract_chat.py`](/Users/saiashwin/BT5153/contract_chat.py): chatbot logic, guardrails, suggested questions, and answer generation
- [`risk_tagger.py`](/Users/saiashwin/BT5153/risk_tagger.py): contextual clause risk tagging with rule-first logic and optional LLM fallback
- [`llm_summary.py`](/Users/saiashwin/BT5153/llm_summary.py): contract summary generation
- [`openai_utils.py`](/Users/saiashwin/BT5153/openai_utils.py): shared OpenAI configuration helpers
- [`static/index.html`](/Users/saiashwin/BT5153/static/index.html): dashboard UI

## API endpoints used by the dashboard

The frontend talks to these Flask routes:

- `GET /api/status`
- `POST /api/classify`
- `POST /api/excerpt`
- `POST /api/explain`
- `POST /api/chat/session`
- `POST /api/chat`

## Running the app from the terminal

### 1. Go to the project directory

```bash
cd /Users/saiashwin/BT5153
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dashboard dependencies

For the dashboard app, the simplest route is:

```bash
pip install --upgrade pip
pip install -r requirements_web.txt
```

If you also want the notebook and model-development stack:

```bash
pip install -r requirements.txt
```

If you want to run the tests and `pytest` is not already installed:

```bash
pip install pytest
```

### 4. Add your environment variables

Create a `.env` file in the project root:

```bash
touch .env
```

Recommended `.env` contents:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CONTRACT_SUMMARY_MODEL=gpt-4o
OPENAI_REVIEW_MODEL=gpt-4o-mini
LEXSCAN_CHAT_MODEL=gpt-4o-mini
LEXSCAN_ENABLE_SECOND_STAGE_REVIEW=true
LEXSCAN_ENABLE_CHATBOT=true
LEXSCAN_RAG_EMBEDDING_BACKEND=tfidf
LEXSCAN_CLASSIFIER_MODE=auto
```

Notes:

- `OPENAI_API_KEY` is required for the AI summary, second-stage LLM review, clause explanation, and chatbot.
- If `OPENAI_API_KEY` is missing, the basic clause classifier can still run.
- `LEXSCAN_CLASSIFIER_MODE=auto` lets the app choose the best available classifier mode.

### 5. Start the dashboard

You can run the app directly:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5001
```

### 6. Optional: use the restart helper

If you want the helper script that kills any old process on port `5001`, starts the app, and tries to reopen the browser automatically:

```bash
python run_dashboard.py
```

### 7. View logs if something goes wrong

The server writes logs to:

```text
dashboard_server.log
```

You can watch the log live:

```bash
tail -f dashboard_server.log
```

## Optional configuration

Some useful runtime knobs:

- `LEXSCAN_ENABLE_SECOND_STAGE_REVIEW`
- `LEXSCAN_ENABLE_CHATBOT`
- `LEXSCAN_BASELINE_THRESHOLD_FLOOR`
- `LEXSCAN_BASELINE_THRESHOLD_CEILING`
- `LEXSCAN_CHAT_MODEL`
- `LEXSCAN_CHAT_TOP_K`
- `LEXSCAN_CHAT_CHUNK_CHARS`
- `LEXSCAN_CHAT_CHUNK_OVERLAP`
- `LEXSCAN_CHAT_HISTORY_TURNS`
- `LEXSCAN_CHAT_MIN_SCORE`
- `LEXSCAN_RISK_TAGGER_LLM_MODEL`
- `LEXSCAN_ENABLE_RISK_LLM_FALLBACK`
- `LEXSCAN_RISK_TAGGER_MAX_LLM_CALLS`
- `LEXSCAN_RAG_EMBEDDING_BACKEND`
- `LEXSCAN_RAG_EMBEDDING_MODEL`
- `LEXSCAN_DOC_CACHE_MAX`
- `LEXSCAN_MAX_RAG_CACHE_MB`
- `LEXSCAN_CLASSIFIER_MODE`
- `LEXSCAN_TORCH_DEVICE`

## Demo files

The repo includes a few sample text contracts in `static/`:

- [`static/demo_contract.txt`](/Users/saiashwin/BT5153/static/demo_contract.txt)
- [`static/demo_research_collab.txt`](/Users/saiashwin/BT5153/static/demo_research_collab.txt)
- [`static/demo_saas_license.txt`](/Users/saiashwin/BT5153/static/demo_saas_license.txt)

These are useful for quick local testing of the dashboard and chatbot.

## Testing

To run the test suite:

```bash
pytest
```

Or run only a focused set:

```bash
pytest tests/test_contract_chat.py tests/test_review_pipeline.py tests/test_app_chat.py
```

## Current limitations

- The chatbot keeps short memory only for the current uploaded document.
- Uploaded documents and chat state are stored in memory, not in a database.
- The chatbot is a contract-support assistant, not a general assistant and not legal advice.
- Some advanced features depend on OpenAI access.
- PDF extraction quality still depends on the quality of the source PDF.

## In simple terms

If you want the shortest description of the project:

LexScan is a contract review assistant that combines supervised clause detection, retrieval-based review, plain-English summarization, and a guardrailed contract chatbot in one local dashboard.
