from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make project root importable when running from webapp/ or repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.contract_analyzer import analyze_risks, classify_document_type, summarize_contract
from webapp.document_parser import extract_text
from webapp.model_loader import load_classifier, predict_clauses

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Contract Review Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Contract Review Assistant")
st.caption(
    "Upload a contract to get an instant document type classification, "
    "plain-English summary, and flagged risk clauses."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
1. **Upload** a PDF, DOCX, or TXT contract
2. A trained TF-IDF classifier detects which clause types are present
3. GPT-4o identifies the document type and summarises it in plain English
4. GPT-4o explains the risk level of each flagged clause
        """
    )
    st.divider()
    st.caption("Clause classifier: TF-IDF + Logistic Regression (trained on CUAD dataset)")

# ── File upload ────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload your contract",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, TXT",
)

if not uploaded:
    st.info("Upload a contract above to begin.")
    st.stop()

# ── Extract text ───────────────────────────────────────────────────────────────

with st.spinner("Reading document…"):
    contract_text = extract_text(uploaded.getvalue(), uploaded.name)

if not contract_text.strip():
    st.error("Could not extract any text from the uploaded file. Please try a different file.")
    st.stop()

with st.expander("Raw extracted text (first 2 000 chars)", expanded=False):
    st.text(contract_text[:2_000])

# ── Load classifier ────────────────────────────────────────────────────────────

artifacts = load_classifier()

# ── Run analysis ───────────────────────────────────────────────────────────────

progress = st.progress(0, text="Starting analysis…")

progress.progress(10, text="Classifying clauses…")
flagged_clauses = predict_clauses(contract_text, artifacts)

progress.progress(35, text="Identifying document type…")
doc_type = classify_document_type(contract_text)

progress.progress(60, text="Summarising contract…")
summary = summarize_contract(contract_text, doc_type)

progress.progress(80, text="Analysing risks…")
risks = analyze_risks(contract_text, flagged_clauses)

progress.progress(100, text="Done!")
progress.empty()

# ── Results layout ─────────────────────────────────────────────────────────────

col_type, col_clauses = st.columns([2, 1])

with col_type:
    st.metric("Document Type", doc_type)

with col_clauses:
    st.metric("Flagged Clause Types", len(flagged_clauses))

st.divider()

# Plain-language summary
st.subheader("Plain-English Summary")
st.markdown(summary)

st.divider()

# Risk flags table
st.subheader("Risk Flags")

if not risks:
    st.success("No risky clauses detected.")
else:
    _LEVEL_ICON = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

    rows = []
    for r in risks:
        level = r.get("risk_level", "Unknown")
        rows.append(
            {
                "Risk": f"{_LEVEL_ICON.get(level, '⚪')} {level}",
                "Clause Type": r.get("clause", ""),
                "What it means": r.get("plain_explanation", ""),
                "Watch out for": r.get("watch_out_for", ""),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download report as CSV",
        data=csv,
        file_name=f"contract_risk_report_{uploaded.name.rsplit('.', 1)[0]}.csv",
        mime="text/csv",
    )

# ── All detected clause types (collapsed) ─────────────────────────────────────

if flagged_clauses:
    with st.expander(f"All {len(flagged_clauses)} detected clause types"):
        for c in sorted(flagged_clauses):
            st.write(f"• {c}")
