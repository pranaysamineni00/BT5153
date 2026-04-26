from config import ReviewConfig
from contract_chat import ContractChatbot, GUARDRAIL_MESSAGE, suggested_queries_for_contract
from document_rag import build_document_index


class _StubRetriever:
    known_labels = {"Termination For Convenience", "License Grant", "Uncapped Liability"}

    def get_definition(self, label):
        return f"{label} definition"

    def retrieve_positive_examples(self, label, query_snippet, top_k=2):
        return [
            {
                "clause_label": label,
                "snippet_text": f"CUAD example for {label}",
            }
        ][:top_k]


def _config():
    return ReviewConfig(
        rag_embedding_backend="tfidf",
        rag_index_cache_path="",
        enable_chatbot=True,
        chat_chunk_chars=220,
        chat_chunk_overlap=40,
        chat_min_score=0.05,
    )


def _classification_result():
    return {
        "clauses": [
            {"clause": "Termination For Convenience"},
            {"clause": "License Grant"},
        ]
    }


def test_suggested_queries_include_generic_and_clause_specific_prompts():
    suggestions = suggested_queries_for_contract(
        _classification_result(),
        {"doc_type": "Software License Agreement"},
    )

    assert len(suggestions) == 4
    assert "What are the main obligations of each party?" in suggestions
    assert any("termination" in item.lower() or "license" in item.lower() for item in suggestions)


def test_contract_chat_refuses_unrelated_questions(monkeypatch):
    monkeypatch.setattr("contract_chat.chatbot_available", lambda config=None: True)
    index = build_document_index(
        "This agreement grants a software license and allows termination for breach.",
        config=_config(),
    )
    bot = ContractChatbot(index, config=_config(), retriever=_StubRetriever())

    payload = bot.answer(
        message="What is the weather in Singapore?",
        history=[],
        classification_result=_classification_result(),
        summary={"doc_type": "Software License Agreement"},
    )

    assert payload["status"] == "refused"
    assert payload["answer"] == GUARDRAIL_MESSAGE


def test_contract_chat_returns_insufficient_evidence_when_scores_are_too_low(monkeypatch):
    monkeypatch.setattr("contract_chat.chatbot_available", lambda config=None: True)
    config = _config()
    config = ReviewConfig(
        rag_embedding_backend="tfidf",
        rag_index_cache_path="",
        enable_chatbot=True,
        chat_chunk_chars=220,
        chat_chunk_overlap=40,
        chat_min_score=0.95,
    )
    index = build_document_index(
        "This agreement grants a software license and allows termination for breach.",
        config=config,
    )
    bot = ContractChatbot(index, config=config, retriever=_StubRetriever())

    payload = bot.answer(
        message="What are the payment terms?",
        history=[],
        classification_result=_classification_result(),
        summary={"doc_type": "Software License Agreement"},
    )

    assert payload["status"] == "insufficient_evidence"
    assert payload["citations"]


def test_contract_chat_returns_answer_with_citations(monkeypatch):
    monkeypatch.setattr("contract_chat.chatbot_available", lambda config=None: True)
    index = build_document_index(
        "Alpha grants Beta a non-exclusive license to use the software.\n\n"
        "Either party may terminate for material breach with thirty days notice.",
        config=_config(),
    )
    bot = ContractChatbot(index, config=_config(), retriever=_StubRetriever())
    monkeypatch.setattr(
        bot,
        "_generate_answer",
        lambda **kwargs: "The contract grants a non-exclusive software license [1].",
    )

    payload = bot.answer(
        message="What rights are being granted under the license?",
        history=[],
        classification_result=_classification_result(),
        summary={"doc_type": "Software License Agreement"},
    )

    assert payload["status"] == "ready"
    assert payload["citations"]
    assert payload["retrieved_examples"]
    assert "[1]" in payload["answer"]


def test_contract_chat_answers_clause_guidance_question_with_background_support(monkeypatch):
    monkeypatch.setattr("contract_chat.chatbot_available", lambda config=None: True)
    config = ReviewConfig(
        rag_embedding_backend="tfidf",
        rag_index_cache_path="",
        enable_chatbot=True,
        chat_chunk_chars=220,
        chat_chunk_overlap=40,
        chat_min_score=0.99,
    )
    index = build_document_index(
        "General services agreement with standard payment and notice terms.",
        config=config,
    )
    bot = ContractChatbot(index, config=config, retriever=_StubRetriever())
    seen_kwargs = {}

    def _fake_generate_answer(**kwargs):
        seen_kwargs.update(kwargs)
        return "Direct contract support is limited, but uncapped liability can expose a party to unlimited losses."

    monkeypatch.setattr(bot, "_generate_answer", _fake_generate_answer)

    payload = bot.answer(
        message="Why should I be concerned about uncapped liability clause?",
        history=[],
        classification_result={
            "clauses": [
                {
                    "clause": "Uncapped Liability",
                    "confidence": 0.91,
                    "snippet": "The supplier remains liable for all losses with no monetary cap.",
                }
            ]
        },
        summary={"doc_type": "Services Agreement"},
    )

    assert payload["status"] == "ready"
    assert payload["citations"]
    assert payload["retrieved_examples"]
    assert seen_kwargs["allow_background_guidance"] is True
    assert seen_kwargs["clause_focus"] == "Uncapped Liability"
