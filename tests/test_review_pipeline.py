from agents import AgentResult
from config import ReviewConfig
from review_pipeline import SecondStageReviewPipeline, compare_baseline_vs_review


class _StubClassifier:
    thresholds = {}
    id_to_clause = {0: "Governing Law", 1: "Dispute Resolution", 2: "Assignment"}
    validation_logits = None
    validation_labels = None

    def get_top_candidate_labels(self, snippet, top_k=3, ensure_labels=None):
        labels = [
            {"label": "Governing Law", "confidence": 0.88},
            {"label": "Dispute Resolution", "confidence": 0.54},
            {"label": "Assignment", "confidence": 0.31},
        ]
        existing = {item["label"] for item in labels}
        for label in ensure_labels or []:
            if label not in existing:
                labels.append({"label": label, "confidence": 0.2})
        return labels[:top_k]


class _StubRetriever:
    def get_definition(self, label):
        return f"Definition for {label}"

    def retrieve_positive_examples(self, label, query_snippet, top_k=3):
        return [
            {
                "source_contract_id": "C1",
                "snippet_text": f"Positive support for {label}",
                "clause_label": label,
            }
        ][:top_k]

    def retrieve_hard_negatives(self, label, query_snippet, top_k=2, model_top_labels=None):
        alt = next((item for item in (model_top_labels or []) if item != label), "Dispute Resolution")
        return [
            {
                "source_contract_id": "C2",
                "snippet_text": f"Potential confusion with {alt}",
                "clause_label": alt,
            }
        ][:top_k]


def _review_config():
    return ReviewConfig(
        rag_index_cache_path="",
        review_model="gpt-4o-mini",
        enable_second_stage_review=True,
    )


def test_review_prediction_skips_llm_when_unavailable(monkeypatch):
    pipeline = SecondStageReviewPipeline(_StubClassifier(), config=_review_config())
    prediction = {
        "clause": "Governing Law",
        "confidence": 0.88,
        "snippet": "This agreement is governed by Delaware law.",
    }

    monkeypatch.setattr("review_pipeline.review_llm_available", lambda: False)

    reviewed = pipeline.review_prediction(prediction)

    assert reviewed["review_status"] == "LLM_SKIPPED"
    assert reviewed["final_decision"] == "ACCEPT"
    assert reviewed["candidate_labels"][0] == "Governing Law"


def test_review_prediction_initializes_retriever_lazily(monkeypatch):
    pipeline = SecondStageReviewPipeline(_StubClassifier(), config=_review_config())
    prediction = {
        "clause": "Governing Law",
        "confidence": 0.88,
        "snippet": "This agreement is governed by Delaware law.",
    }

    monkeypatch.setattr("review_pipeline.review_llm_available", lambda: True)
    monkeypatch.setattr("review_pipeline.get_default_retriever", lambda **kwargs: _StubRetriever())
    monkeypatch.setattr(
        pipeline.presence_agent,
        "run",
        lambda packet, label: AgentResult(
            payload={"presence": "PRESENT", "reason": "Clause is present.", "confidence": 0.91},
            valid_json=True,
            raw_response='{"presence":"PRESENT"}',
        ),
    )
    monkeypatch.setattr(
        pipeline.support_agent,
        "run",
        lambda packet, label: AgentResult(
            payload={"supports_label": "YES", "better_label_if_any": None, "reason": "Evidence matches.", "confidence": 0.89},
            valid_json=True,
            raw_response='{"supports_label":"YES"}',
        ),
    )

    reviewed = pipeline.review_prediction(prediction)

    assert reviewed["review_status"] == "READY"
    assert reviewed["final_decision"] == "ACCEPT"
    assert reviewed["retrieved_evidence"]["positive_examples"]


def test_review_prediction_falls_back_to_human_review_after_invalid_json(monkeypatch):
    pipeline = SecondStageReviewPipeline(_StubClassifier(), config=_review_config(), retriever=_StubRetriever())
    prediction = {
        "clause": "Governing Law",
        "confidence": 0.72,
        "snippet": "This agreement is governed by Delaware law.",
    }

    monkeypatch.setattr("review_pipeline.review_llm_available", lambda: True)
    monkeypatch.setattr(
        pipeline.presence_agent,
        "run",
        lambda packet, label: AgentResult(payload=None, valid_json=False, raw_response="not json"),
    )
    monkeypatch.setattr(
        pipeline.support_agent,
        "run",
        lambda packet, label: AgentResult(payload=None, valid_json=False, raw_response="still not json"),
    )

    reviewed = pipeline.review_prediction(prediction)

    assert reviewed["review_status"] == "HUMAN_REVIEW"
    assert reviewed["final_decision"] == "HUMAN_REVIEW"


def test_compare_baseline_vs_review_reports_metrics_and_decisions():
    review_outputs = [
        {
            "original_model_label": "A",
            "final_decision": "ACCEPT",
            "final_label": "A",
        },
        {
            "original_model_label": "B",
            "final_decision": "REJECT",
            "final_label": None,
        },
        {
            "original_model_label": "C",
            "final_decision": "RERANK_LABEL",
            "final_label": "D",
        },
    ]
    true_labels = [["A"], [], ["D"]]

    metrics = compare_baseline_vs_review(review_outputs, true_labels)

    assert set(metrics) == {"baseline", "reviewed", "decision_counts", "false_positive_reduction"}
    assert metrics["decision_counts"]["ACCEPT"] == 1
    assert metrics["decision_counts"]["REJECT"] == 1
    assert metrics["decision_counts"]["RERANK_LABEL"] == 1
    assert metrics["false_positive_reduction"] >= 0
