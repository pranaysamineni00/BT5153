import llm_summary


def test_parse_summary_bullets_handles_markdown_and_continuations():
    summary = (
        "- Parties: Alpha licenses software to Beta.\n"
        "  This continues on the next line.\n"
        "- Term: The agreement lasts one year.\n"
        "3. Termination: Either side can end for breach."
    )

    bullets = llm_summary.parse_summary_bullets(summary)

    assert bullets == [
        "Parties: Alpha licenses software to Beta. This continues on the next line.",
        "Term: The agreement lasts one year.",
        "Termination: Either side can end for breach.",
    ]


def test_build_contract_summary_returns_disabled_payload_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    payload = llm_summary.build_contract_summary("Sample contract text")

    assert payload["status"] == "disabled"
    assert payload["available"] is False
    assert payload["bullets"] == []
    assert "OPENAI_API_KEY" in payload["message"]


def test_build_contract_summary_formats_successful_result(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_summary, "_has_openai_package", lambda: True)
    monkeypatch.setattr(
        llm_summary,
        "classify_document_type",
        lambda text: "Software License Agreement",
    )
    monkeypatch.setattr(
        llm_summary,
        "summarize_contract",
        lambda text, doc_type: (
            "- Parties: Alpha licenses software to Beta.\n"
            "- Payment: Beta pays annually.\n"
            "- Termination: Either side can terminate for breach."
        ),
    )

    payload = llm_summary.build_contract_summary("Sample contract text")

    assert payload["status"] == "ready"
    assert payload["available"] is True
    assert payload["doc_type"] == "Software License Agreement"
    assert payload["bullets"] == [
        "Parties: Alpha licenses software to Beta.",
        "Payment: Beta pays annually.",
        "Termination: Either side can terminate for breach.",
    ]
