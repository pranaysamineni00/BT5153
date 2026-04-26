import app as app_module


def test_chat_session_requires_document_id():
    client = app_module.app.test_client()

    response = client.post("/api/chat/session", json={})

    assert response.status_code == 400


def test_chat_session_returns_cached_state():
    client = app_module.app.test_client()
    doc_id = app_module._cache_document(
        "Sample contract text",
        chat_state={
            "history": [],
            "classification_result": {"clauses": []},
            "summary": {"doc_type": ""},
            "suggested_queries": ["What are the main obligations of each party?"],
            "history_limit": 6,
            "status": "ready",
        },
    )

    response = client.post("/api/chat/session", json={"document_id": doc_id})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ready"
    assert payload["history_limit"] == 6


def test_chat_endpoint_returns_guardrail_payload(monkeypatch):
    client = app_module.app.test_client()
    doc_id = app_module._cache_document(
        "Sample contract text",
        chat_state={
            "history": [],
            "classification_result": {"clauses": []},
            "summary": {"doc_type": ""},
            "suggested_queries": ["What are the main obligations of each party?"],
            "history_limit": 6,
            "status": "ready",
            "document_index": object(),
        },
    )

    class _StubBot:
        def __init__(self, document_index, config=None):
            pass

        def answer(self, **kwargs):
            return {
                "status": "refused",
                "answer": "I am just a legal contract support agent, I can't answer this.",
                "citations": [],
                "retrieved_examples": [],
                "suggested_queries": ["What are the main obligations of each party?"],
                "model": "",
                "history_used": 0,
            }

    monkeypatch.setattr(app_module, "ContractChatbot", _StubBot)

    response = client.post("/api/chat", json={"document_id": doc_id, "message": "Tell me a joke"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "refused"
    assert "legal contract support agent" in payload["answer"]
