from config import ReviewConfig
from risk_tagger import enrich_clause_risks, evaluate_clause_risk


def test_evaluate_clause_risk_uses_rules_for_uncapped_liability():
    verdict = evaluate_clause_risk(
        "Uncapped Liability",
        "Supplier liability shall not be limited and applies without limitation.",
        "HIGH",
        use_llm=False,
    )

    assert verdict["risk"] == "HIGH"
    assert verdict["source"] == "rule"
    assert "uncapped" in verdict["watch_out_for"].lower() or "cap" in verdict["watch_out_for"].lower()


def test_enrich_clause_risks_skips_rejected_review_items():
    clauses = [
        {
            "clause": "Governing Law",
            "confidence": 0.88,
            "risk": "LOW",
            "snippet": "This agreement is governed by the laws of Delaware.",
        }
    ]
    review_items = [
        {
            "original_model_label": "Governing Law",
            "final_decision": "REJECT",
            "final_label": None,
        }
    ]

    enriched = enrich_clause_risks(
        clauses,
        review_items=review_items,
        config=ReviewConfig(enable_risk_llm_fallback=False),
    )

    assert enriched[0]["risk_source"] == "static"
    assert "rejected" in enriched[0]["risk_reason"].lower()
    assert enriched[0]["watch_out_for"]


def test_enrich_clause_risks_applies_notice_rule_without_llm():
    clauses = [
        {
            "clause": "Termination For Convenience",
            "confidence": 0.67,
            "risk": "MEDIUM",
            "snippet": "Either party may terminate this Agreement upon 15 days written notice.",
            "low_confidence": True,
        }
    ]

    enriched = enrich_clause_risks(
        clauses,
        review_items=[],
        config=ReviewConfig(enable_risk_llm_fallback=False),
    )

    assert enriched[0]["risk"] == "HIGH"
    assert enriched[0]["risk_source"] == "rule"
    assert "15 days" in enriched[0]["risk_reason"]
