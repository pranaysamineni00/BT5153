from __future__ import annotations

import json
import re

import streamlit as st
from openai import OpenAI

_MAX_CHARS_TYPE = 3_000
_MAX_CHARS_SUMMARY = 6_000
_MAX_CHARS_RISK = 8_000


@st.cache_resource(show_spinner=False)
def _client() -> OpenAI:
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def classify_document_type(text: str) -> str:
    """Return document type as a short label, e.g. 'Software License Agreement'."""
    resp = _client().chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal assistant. Identify the contract document type "
                    "in 3–6 words (e.g. 'Non-Disclosure Agreement', 'Software License Agreement', "
                    "'Employment Contract'). Reply with only the document type, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Contract excerpt:\n\n{text[:_MAX_CHARS_TYPE]}",
            },
        ],
    )
    return (resp.choices[0].message.content or "Unknown Document Type").strip()


def summarize_contract(text: str, doc_type: str) -> str:
    """Return a plain-English bullet-point summary aimed at non-lawyers."""
    resp = _client().chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal assistant that explains contracts in plain English "
                    "for people without a legal background. Be concise and avoid jargon."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Summarize this {doc_type} in plain English using 4–6 bullet points. "
                    "Cover: (1) parties involved, (2) key obligations of each party, "
                    "(3) payment terms or duration if present, (4) termination conditions, "
                    "(5) any unusual or one-sided terms worth noting.\n\n"
                    f"Contract:\n\n{text[:_MAX_CHARS_SUMMARY]}"
                ),
            },
        ],
    )
    return (resp.choices[0].message.content or "_Summary unavailable._").strip()


def analyze_risks(text: str, flagged_clauses: list[str]) -> list[dict]:
    """Return a list of risk dicts for each flagged clause.

    Each dict has keys: clause, risk_level, plain_explanation, watch_out_for.
    risk_level is one of 'Low', 'Medium', 'High'.
    """
    if not flagged_clauses:
        return []

    clause_list = "\n".join(f"- {c}" for c in flagged_clauses)
    prompt = (
        "You are a legal risk analyst. The following clause types were detected in a contract.\n"
        "Return a JSON object of the form:\n"
        '  {"clauses": [ {"clause": "...", "risk_level": "Low|Medium|High", '
        '"plain_explanation": "...", "watch_out_for": "..."}, ... ]}\n'
        "- clause: the clause type name\n"
        "- risk_level: one of Low, Medium, or High\n"
        "- plain_explanation: 1–2 sentences in plain English describing what the clause means\n"
        "- watch_out_for: one concrete thing the signer should be cautious about\n\n"
        f"Detected clauses:\n{clause_list}\n\n"
        f"Relevant contract text:\n\n{text[:_MAX_CHARS_RISK]}"
    )

    resp = _client().chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a legal risk analyst. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    if not raw:
        # Model returned empty content (content filter or refusal) — degrade gracefully
        return [
            {
                "clause": c,
                "risk_level": "Unknown",
                "plain_explanation": "The model could not generate an explanation for this clause.",
                "watch_out_for": "Review this clause manually.",
            }
            for c in flagged_clauses
        ]

    parsed = json.loads(raw)

    # Expecting {"clauses": [...]}; fall back to first list-valued key
    if isinstance(parsed, dict):
        if "clauses" in parsed and isinstance(parsed["clauses"], list):
            return parsed["clauses"]
        for value in parsed.values():
            if isinstance(value, list):
                return value
        return []

    return parsed if isinstance(parsed, list) else []
