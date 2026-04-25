from decision_router import route_review_decision


def test_router_accepts_high_confidence_supported_labels():
    result = route_review_decision(
        original_model_label="Governing Law",
        model_confidence=0.92,
        presence="PRESENT",
        supports_label="YES",
        better_label_if_any=None,
        reason="Matches the definition and examples.",
    )

    assert result["final_decision"] == "ACCEPT"
    assert result["final_label"] == "Governing Law"


def test_router_reranks_when_support_says_no_and_better_label_exists():
    result = route_review_decision(
        original_model_label="Assignment",
        model_confidence=0.61,
        presence="PRESENT",
        supports_label="NO",
        better_label_if_any="Change Of Control",
        reason="The snippet is about change of control, not assignment.",
    )

    assert result["final_decision"] == "RERANK_LABEL"
    assert result["final_label"] == "Change Of Control"


def test_router_rejects_absent_or_unsupported_labels():
    result = route_review_decision(
        original_model_label="Non-Compete",
        model_confidence=0.77,
        presence="ABSENT",
        supports_label="UNCERTAIN",
        better_label_if_any=None,
        reason="The snippet does not contain a non-compete obligation.",
    )

    assert result["final_decision"] == "REJECT"
    assert result["final_label"] is None


def test_router_sends_uncertain_cases_to_human_review():
    result = route_review_decision(
        original_model_label="Indemnification",
        model_confidence=0.64,
        presence="UNCERTAIN",
        supports_label="YES",
        better_label_if_any=None,
        reason="The language is close but incomplete.",
    )

    assert result["final_decision"] == "HUMAN_REVIEW"
    assert result["final_label"] == "Indemnification"
