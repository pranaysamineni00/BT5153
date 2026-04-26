from document_rag import build_document_index, chunk_contract_text
from config import ReviewConfig


def test_chunk_contract_text_creates_multiple_paragraph_aware_chunks():
    text = (
        "Parties paragraph with names and duties.\n\n"
        "Payment paragraph describing invoicing and fees.\n\n"
        "Termination paragraph covering notice and breach."
    )

    chunks = chunk_contract_text(text, chunk_chars=70, overlap_chars=15)

    assert len(chunks) >= 2
    assert chunks[0].text
    assert chunks[0].start_char < chunks[0].end_char


def test_build_document_index_ranks_relevant_chunk():
    text = (
        "Alpha grants Beta a non-exclusive license to use the software.\n\n"
        "Either party may terminate for material breach with thirty days notice."
    )
    config = ReviewConfig(
        rag_embedding_backend="tfidf",
        rag_index_cache_path="",
        chat_chunk_chars=120,
        chat_chunk_overlap=20,
    )

    index = build_document_index(text, config=config)
    results = index.search("What does the license allow?", top_k=1)

    assert len(results) == 1
    assert "license" in results[0].text.lower()
