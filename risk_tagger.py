"""Context-aware risk tagging for detected CUAD clauses."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from classifier import RISK_LEVELS
from openai_utils import get_openai_client, has_openai_package, openai_api_key

logger = logging.getLogger(__name__)

_RISK_VALUES = {"HIGH", "MEDIUM", "LOW"}

WATCH_OUT_FOR: dict[str, str] = {
    "Cap On Liability": "Confirm the cap amount is proportionate to deal size and excludes IP/indemnity carve-outs.",
    "Uncapped Liability": "Identify which categories of damages are uncapped and their exposure.",
    "Indemnification": "Check scope, mutuality, defense control, and any caps or carve-outs.",
    "IP Ownership Assignment": "Verify which IP transfers, when, and whether background IP is preserved.",
    "Joint IP Ownership": "Clarify each party's exploitation, licensing, and accounting rights.",
    "License Grant": "Check exclusivity, scope, sublicensing, and territorial limits.",
    "Non-Compete": "Look at duration, geography, and scope of restricted activities.",
    "Exclusivity": "Check duration, geography, product scope, and any minimum-volume conditions.",
    "Non-Solicit Of Customers": "Verify duration and the definition of customer.",
    "No-Solicit Of Customers": "Verify duration and the definition of customer.",
    "No-Solicit Of Employees": "Check duration and whether general advertising is exempted.",
    "Termination For Convenience": "Confirm notice period, transition obligations, and any termination fees.",
    "Change Of Control": "Identify what triggers the clause and the counterparty's rights on a sale.",
    "Anti-Assignment": "Check whether assignment to affiliates or in M&A is permitted.",
    "Source Code Escrow": "Check release conditions, beneficiaries, and whether updates are escrowed.",
    "Liquidated Damages": "Verify the formula and that it is a reasonable pre-estimate of harm.",
    "Most Favored Nation": "Identify which terms are MFN-protected and the audit mechanism.",
    "Auto-Renewal": "Check the cancellation notice window and how price increases are handled.",
    "Renewal Term": "Note the renewal length and whether either party may opt out.",
    "Notice Period To Terminate Renewal": "Confirm the notice window is workable and tracked internally.",
    "Governing Law": "Confirm the chosen law is acceptable and a familiar forum to your counsel.",
    "Dispute Resolution": "Check forum, venue, and whether mandatory mediation/arbitration applies.",
    "Arbitration": "Verify the seat, rules, language, and arbitrator selection.",
    "Force Majeure": "Check covered events and whether epidemics or supply-chain disruption are listed.",
    "Insurance": "Confirm coverage types, minimum limits, and additional-insured language.",
    "Warranty Duration": "Note the warranty length and any exclusions or remedies.",
    "Audit Rights": "Check audit frequency, notice, and who pays for the audit.",
    "Covenant Not To Sue": "Identify the covered claims and any survival period.",
    "Non-Disparagement": "Confirm scope, duration, and exceptions for honest opinions or required disclosures.",
    "Post-Termination Services": "Check duration, fee structure, and minimum service levels.",
    "Revenue/Profit Sharing": "Verify the formula, audit rights, and reporting cadence.",
    "Affiliate License-Licensor": "Check which affiliates qualify and the duration of their rights.",
    "Affiliate License-Licensee": "Check which affiliates qualify and the duration of their rights.",
    "Third Party Beneficiary": "Identify any third parties that can enforce the agreement.",
    "Non-Transferable License": "Confirm the limits on transfer, sublicensing, and use by affiliates.",
    "Irrevocable Or Perpetual License": "Verify the perpetual or irrevocable scope cannot be terminated for breach.",
    "Rofr/Rofo/Rofn": "Check trigger events, response window, and price determination.",
    "Document Name": "Sanity-check the document title matches the actual agreement type.",
    "Parties": "Confirm legal entity names, signing authority, and affiliate definitions.",
    "Agreement Date": "Note the agreement date for term calculations.",
    "Effective Date": "Confirm when obligations begin if different from the agreement date.",
    "Expiration Date": "Track the expiration or renewal trigger.",
}

_DEFAULT_WATCH = "Review the clause language carefully and consult counsel if anything is unclear."

_DOLLAR_RE = re.compile(
    r"\$\s?([\d,]+(?:\.\d+)?)(?:\s*(million|mm|m|billion|bn|thousand|k))?",
    re.IGNORECASE,
)

_NOTICE_DAYS_RE = re.compile(
    r"(?:upon|with|by|on|provide|provides|providing|giving|give)\s+(?:not\s+less\s+than\s+|at\s+least\s+|a\s+minimum\s+of\s+)?(\d{1,3})\s*(?:\(\d+\)\s*)?(?:calendar\s+|business\s+|working\s+)?days?(?:'?\s*(?:prior|advance)?\s*)?(?:written\s+)?(?:notice|notification)?",
    re.IGNORECASE,
)


def _parse_dollars(text: str) -> list[float]:
    out: list[float] = []
    for amount_str, unit in _DOLLAR_RE.findall(text):
        try:
            value = float(amount_str.replace(",", ""))
        except ValueError:
            continue
        unit = (unit or "").lower()
        if unit in {"million", "mm", "m"}:
            value *= 1_000_000
        elif unit in {"billion", "bn"}:
            value *= 1_000_000_000
        elif unit in {"thousand", "k"}:
            value *= 1_000
        out.append(value)
    return out


def _parse_notice_days(text: str) -> int | None:
    matches = [int(m) for m in _NOTICE_DAYS_RE.findall(text)]
    return min(matches) if matches else None


def _rule_cap_on_liability(snippet: str, fallback: str) -> dict | None:
    s = snippet.lower()
    if any(p in s for p in ("no cap", "uncapped", "without limitation", "shall not be limited")):
        return {
            "risk": "HIGH",
            "risk_reason": "Liability appears to be uncapped or unlimited.",
            "watch_out_for": "Push for a cap; uncapped liability is a major exposure.",
        }
    amounts = _parse_dollars(snippet)
    if amounts:
        cap = max(amounts)
        if cap >= 1_000_000:
            return {
                "risk": "HIGH",
                "risk_reason": f"Liability cap is ${cap:,.0f} and creates high absolute exposure.",
                "watch_out_for": "Confirm carve-outs such as IP, confidentiality, indemnity, and insurance coverage at this level.",
            }
        if cap < 50_000:
            return {
                "risk": "LOW",
                "risk_reason": f"Liability is tightly capped at ${cap:,.0f}.",
                "watch_out_for": "Verify the cap is still sufficient given deal size and realistic harm scenarios.",
            }
        return {
            "risk": "MEDIUM",
            "risk_reason": f"Liability cap is ${cap:,.0f}.",
            "watch_out_for": "Compare the cap to deal value and check carve-outs.",
        }
    if "fees paid" in s or "amounts paid" in s or "12 months" in s or "twelve (12) months" in s:
        return {
            "risk": "MEDIUM",
            "risk_reason": "Liability is capped at fees paid, often over the last 12 months.",
            "watch_out_for": "If contract value is small the cap may be inadequate; check carve-outs for IP and indemnity.",
        }
    return None


def _rule_termination_for_convenience(snippet: str, fallback: str) -> dict | None:
    notice = _parse_notice_days(snippet)
    if notice is None:
        return None
    if notice >= 60:
        return {
            "risk": "LOW",
            "risk_reason": f"Termination requires {notice} days' notice, which gives a longer wind-down period.",
            "watch_out_for": "Plan transition obligations and any final-payment terms within the notice window.",
        }
    if notice < 30:
        return {
            "risk": "HIGH",
            "risk_reason": f"Only {notice} days' notice is required to terminate, which creates a short runway.",
            "watch_out_for": "Negotiate a longer notice window or transition assistance.",
        }
    return {
        "risk": "MEDIUM",
        "risk_reason": f"Termination notice is {notice} days.",
        "watch_out_for": "Confirm transition obligations and any termination fees.",
    }


def _rule_unbounded(snippet: str, fallback: str, label: str) -> dict | None:
    s = snippet.lower()
    has_world = any(p in s for p in ("worldwide", "world-wide", "throughout the world", "globally"))
    has_perpetual = any(p in s for p in ("perpetual", "in perpetuity", "indefinite", "no expiration"))
    bounded = bool(re.search(r"\b(within|in)\s+(the\s+)?(united states|u\.s\.|usa|europe|asia|[a-z\s]{3,30})", s)) or bool(re.search(r"\b\d+\s*(year|month)s?\b", s))
    if has_world or has_perpetual:
        markers = [m for m, p in [("worldwide", has_world), ("perpetual", has_perpetual)] if p]
        return {
            "risk": "HIGH",
            "risk_reason": f"{label} appears unbounded ({', '.join(markers)}).",
            "watch_out_for": "Negotiate explicit geographic and temporal limits.",
        }
    if bounded:
        return {
            "risk": "MEDIUM",
            "risk_reason": f"{label} has explicit geographic or temporal limits.",
            "watch_out_for": "Confirm the limits match your business plan.",
        }
    return None


def _rule_liquidated_damages(snippet: str, fallback: str) -> dict | None:
    amounts = _parse_dollars(snippet)
    if amounts or re.search(r"per\s+(day|week|month|incident|breach)", snippet, re.IGNORECASE):
        return {
            "risk": "HIGH",
            "risk_reason": "Liquidated damages include a specific dollar amount or formula.",
            "watch_out_for": "Confirm the amount is a reasonable pre-estimate of harm, not a penalty.",
        }
    return None


def _rule_ip_assignment(snippet: str, fallback: str) -> dict | None:
    s = snippet.lower()
    has_full = "all right, title" in s or "all rights, title" in s
    has_perpetual = "perpetual" in s or "in perpetuity" in s
    has_term_only = "during the term" in s or "term of this agreement" in s
    if has_full and has_perpetual:
        return {
            "risk": "HIGH",
            "risk_reason": "The clause assigns all right, title, and interest on a full and perpetual basis.",
            "watch_out_for": "Carve out background IP, residuals, or pre-existing materials.",
        }
    if has_term_only:
        return {
            "risk": "MEDIUM",
            "risk_reason": "The IP assignment appears limited to the term of the agreement.",
            "watch_out_for": "Confirm what reverts on termination and whether licenses survive.",
        }
    return None


def _rule_auto_renewal(snippet: str, fallback: str) -> dict | None:
    notice = _parse_notice_days(snippet)
    if notice is None:
        if re.search(r"automatic(ally)?\s+renew", snippet, re.IGNORECASE):
            return {
                "risk": "HIGH",
                "risk_reason": "The contract auto-renews without a clear cancellation notice window.",
                "watch_out_for": "Add an opt-out mechanism and a calendar reminder before renewal.",
            }
        return None
    if notice >= 30:
        return {
            "risk": "MEDIUM",
            "risk_reason": f"Auto-renewal cancellation notice is {notice} days.",
            "watch_out_for": "Track the notice deadline; missing it locks in another term.",
        }
    return {
        "risk": "HIGH",
        "risk_reason": f"Auto-renewal cancellation requires only {notice} days' notice and is easy to miss.",
        "watch_out_for": "Negotiate a longer window or remove auto-renewal.",
    }


def _rule_governing_law(snippet: str, fallback: str) -> dict | None:
    home = {"singapore", "united states", "u.s.", "usa", "delaware", "new york", "california"}
    matches = re.findall(r"laws?\s+of\s+(?:the\s+)?(?:state\s+of\s+)?([A-Za-z][A-Za-z\s]{1,30})", snippet)
    if not matches:
        return None
    jurisdiction = matches[0].strip()
    if jurisdiction.lower() in home:
        return {
            "risk": "LOW",
            "risk_reason": f"Governing law is {jurisdiction.title()}, which is typically a familiar jurisdiction.",
            "watch_out_for": "Confirm the venue clause is consistent with the chosen law.",
        }
    return {
        "risk": "MEDIUM",
        "risk_reason": f"Governing law is {jurisdiction.title()}, which may be a foreign jurisdiction for home counsel.",
        "watch_out_for": "Engage local counsel for enforcement and check tax or withholding implications.",
    }


_RULES: dict[str, Any] = {
    "Cap On Liability": _rule_cap_on_liability,
    "Uncapped Liability": _rule_cap_on_liability,
    "Termination For Convenience": _rule_termination_for_convenience,
    "Liquidated Damages": _rule_liquidated_damages,
    "IP Ownership Assignment": _rule_ip_assignment,
    "Governing Law": _rule_governing_law,
}


def _apply_snippet_rule(clause: str, snippet: str, fallback: str) -> dict | None:
    rule = _RULES.get(clause)
    if rule is None:
        if clause in {"Non-Compete", "Exclusivity"}:
            return _rule_unbounded(snippet, fallback, clause)
        if clause in {"Auto-Renewal", "Renewal Term"}:
            return _rule_auto_renewal(snippet, fallback)
        return None
    try:
        return rule(snippet, fallback)
    except Exception:
        logger.exception("Snippet rule failed for %s", clause)
        return None


_LLM_SYSTEM = (
    "You are a senior commercial lawyer reviewing one contract clause. "
    "Return strict JSON with three fields: risk (HIGH|MEDIUM|LOW), risk_reason "
    "(one short sentence citing what in the snippet drives the risk), and "
    "watch_out_for (one short sentence of practical advice). "
    "If the snippet is too short or unrelated to the labeled clause, set risk to "
    "the provided fallback and explain why."
)

_LLM_USER_TEMPLATE = (
    "Clause label: {clause}\n"
    "Static fallback risk: {fallback}\n"
    "Snippet (truncated):\n\"\"\"\n{snippet}\n\"\"\"\n\n"
    "Return JSON only, no commentary."
)


def _llm_evaluate(clause: str, snippet: str, fallback: str, model: str) -> dict | None:
    if not openai_api_key() or not has_openai_package():
        return None
    try:
        client = get_openai_client(timeout=8.0)
        snippet_clean = snippet.replace('"""', "''")[:1200]
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": _LLM_USER_TEMPLATE.format(clause=clause, fallback=fallback, snippet=snippet_clean)},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception:
        logger.exception("LLM risk fallback failed for %s", clause)
        return None

    risk = str(data.get("risk", "")).upper().strip()
    if risk not in _RISK_VALUES:
        return None
    return {
        "risk": risk,
        "risk_reason": str(data.get("risk_reason", "")).strip()[:280] or f"AI-classified as {risk}.",
        "watch_out_for": str(data.get("watch_out_for", "")).strip()[:280] or WATCH_OUT_FOR.get(clause, _DEFAULT_WATCH),
    }


def evaluate_clause_risk(
    clause: str,
    snippet: str,
    fallback_risk: str,
    *,
    use_llm: bool = True,
    low_confidence: bool = False,
    llm_model: str = "gpt-4o-mini",
) -> dict:
    """Returns {risk, risk_reason, watch_out_for, source}."""
    fallback = fallback_risk if fallback_risk in _RISK_VALUES else "LOW"
    snippet = (snippet or "").strip()
    static_watch = WATCH_OUT_FOR.get(clause, _DEFAULT_WATCH)

    if snippet:
        rule_out = _apply_snippet_rule(clause, snippet, fallback)
        if rule_out:
            return {
                "risk": rule_out["risk"],
                "risk_reason": rule_out["risk_reason"],
                "watch_out_for": rule_out.get("watch_out_for") or static_watch,
                "source": "rule",
            }

    needs_llm = use_llm and snippet and (low_confidence or fallback in {"HIGH", "MEDIUM"})
    if needs_llm:
        llm_out = _llm_evaluate(clause, snippet, fallback, llm_model)
        if llm_out:
            return {**llm_out, "source": "llm"}

    return {
        "risk": fallback,
        "risk_reason": f"No contextual trigger fired, so the default {fallback.lower()} risk was kept for this clause.",
        "watch_out_for": static_watch,
        "source": "static",
    }


def _is_rejected(review_items: Iterable[dict] | None, clause_label: str) -> bool:
    if not review_items:
        return False
    label_lower = clause_label.lower()
    for item in review_items:
        if str(item.get("final_decision", "")).upper() != "REJECT":
            continue
        if str(item.get("original_model_label", "")).lower() == label_lower:
            return True
        if str(item.get("final_label", "")).lower() == label_lower:
            return True
    return False


def enrich_clause_risks(
    clauses: list[dict],
    *,
    review_items: Iterable[dict] | None = None,
    config=None,
) -> list[dict]:
    """Mutates each clause dict in-place adding risk_reason / watch_out_for / risk_source."""
    if not clauses:
        return clauses

    use_llm_global = bool(getattr(config, "enable_risk_llm_fallback", True))
    llm_model = str(getattr(config, "risk_tagger_llm_model", "gpt-4o-mini"))
    budget = int(getattr(config, "risk_tagger_llm_budget", 10))
    review_list = list(review_items or [])

    def _llm_priority(c: dict) -> tuple[int, float]:
        bucket = 0 if c.get("low_confidence") else (1 if c.get("risk") in {"HIGH", "MEDIUM"} else 2)
        return (bucket, -float(c.get("confidence") or 0.0))

    ordered_indexes = sorted(range(len(clauses)), key=lambda i: _llm_priority(clauses[i]))
    llm_calls_remaining = budget

    for idx in ordered_indexes:
        clause = clauses[idx]
        label = str(clause.get("clause") or "").strip()
        snippet = str(clause.get("snippet") or "").strip()
        fallback = str(clause.get("risk") or "").upper().strip() or RISK_LEVELS.get(label, "LOW")

        if _is_rejected(review_list, label):
            clause["risk_reason"] = "The second-stage review rejected this clause, so no contextual risk escalation was applied."
            clause["watch_out_for"] = WATCH_OUT_FOR.get(label, _DEFAULT_WATCH)
            clause["risk_source"] = "static"
            continue

        allow_llm = use_llm_global and llm_calls_remaining > 0
        verdict = evaluate_clause_risk(
            label,
            snippet,
            fallback,
            use_llm=allow_llm,
            low_confidence=bool(clause.get("low_confidence")),
            llm_model=llm_model,
        )
        if verdict["source"] == "llm":
            llm_calls_remaining -= 1
        clause["risk"] = verdict["risk"]
        clause["risk_reason"] = verdict["risk_reason"]
        clause["watch_out_for"] = verdict["watch_out_for"]
        clause["risk_source"] = verdict["source"]

    logger.info(
        "risk_tagger: enriched %d clauses (llm_calls_used=%d/%d, sources=%s)",
        len(clauses),
        budget - llm_calls_remaining,
        budget,
        {c.get("risk_source") for c in clauses},
    )
    return clauses


def recompute_risk_summary(clauses: list[dict]) -> dict[str, int]:
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for clause in clauses:
        risk = str(clause.get("risk", "")).upper()
        if risk in counts:
            counts[risk] += 1
    return counts
