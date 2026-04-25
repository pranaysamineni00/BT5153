import numpy as np

from rag_index import RagExample, RagIndex
from retriever import RagRetriever


def _make_index() -> RagIndex:
    entries = [
        RagExample(
            example_id="a1",
            clause_label="Governing Law",
            clause_definition="Contract language describing governing law.",
            source_contract_id="C1",
            snippet_text="This agreement is governed by Delaware law.",
            answer_text="governed by Delaware law",
            question="What is the governing law?",
        ),
        RagExample(
            example_id="a2",
            clause_label="Dispute Resolution",
            clause_definition="Contract language describing dispute resolution.",
            source_contract_id="C2",
            snippet_text="The parties submit to exclusive jurisdiction in New York.",
            answer_text="exclusive jurisdiction in New York",
            question="What is the dispute resolution clause?",
        ),
        RagExample(
            example_id="a3",
            clause_label="License Grant",
            clause_definition="Contract language describing a license grant.",
            source_contract_id="C3",
            snippet_text="Licensor grants a non-exclusive license to the software.",
            answer_text="grants a non-exclusive license",
            question="What is the license grant clause?",
        ),
        RagExample(
            example_id="a4",
            clause_label="IP Ownership Assignment",
            clause_definition="Contract language describing IP ownership assignment.",
            source_contract_id="C4",
            snippet_text="All work product and inventions are owned by the company.",
            answer_text="owned by the company",
            question="What is the IP ownership clause?",
        ),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    index = RagIndex(
        entries=entries,
        embeddings=embeddings,
        label_definitions={
            "Governing Law": "Contract language describing governing law.",
            "Dispute Resolution": "Contract language describing dispute resolution.",
            "License Grant": "Contract language describing a license grant.",
            "IP Ownership Assignment": "Contract language describing IP ownership assignment.",
        },
        entry_ids_by_label={
            "Governing Law": [0],
            "Dispute Resolution": [1],
            "License Grant": [2],
            "IP Ownership Assignment": [3],
        },
        hard_negative_entry_ids_by_label={
            "Governing Law": [1],
            "Dispute Resolution": [],
            "License Grant": [],
            "IP Ownership Assignment": [],
        },
        label_neighbor_map={
            "Governing Law": ["Dispute Resolution", "License Grant"],
            "Dispute Resolution": ["Governing Law"],
            "License Grant": ["IP Ownership Assignment"],
            "IP Ownership Assignment": ["License Grant"],
        },
        train_contract_ids=["C1", "C2", "C3", "C4"],
        split_seed=42,
        vector_store_type="numpy",
        embedding_backend="tfidf",
        embedding_model_name="tfidf",
        confusion_map={"License Grant": ["IP Ownership Assignment"]},
        tfidf_vectorizer=None,
    )

    def _encode_query(text: str) -> np.ndarray:
        lower = text.lower()
        if "ownership" in lower or "owned" in lower or "inventions" in lower:
            query = np.asarray([0.0, 0.9, 0.1], dtype=np.float32)
        elif "license" in lower or "licensor" in lower or "grant" in lower:
            query = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        return query / np.linalg.norm(query)

    index.encode_query = _encode_query  # type: ignore[method-assign]
    return index


def test_get_hard_negative_labels_prefers_manual_mapping():
    retriever = RagRetriever(_make_index())

    negatives = retriever.get_hard_negative_labels("Governing Law")

    assert negatives == ["Dispute Resolution"]


def test_get_hard_negative_labels_uses_model_top_labels_before_other_fallbacks():
    retriever = RagRetriever(_make_index())

    negatives = retriever.get_hard_negative_labels(
        "License Grant",
        model_top_labels=["License Grant", "IP Ownership Assignment", "Governing Law"],
    )

    assert negatives == ["IP Ownership Assignment", "Governing Law"]


def test_retrieve_positive_examples_returns_label_matched_examples():
    retriever = RagRetriever(_make_index())

    matches = retriever.retrieve_positive_examples(
        "License Grant",
        "The agreement grants a software license to the customer.",
        top_k=1,
    )

    assert len(matches) == 1
    assert matches[0]["clause_label"] == "License Grant"


def test_retrieve_hard_negatives_returns_confusing_examples():
    retriever = RagRetriever(_make_index())

    matches = retriever.retrieve_hard_negatives(
        "License Grant",
        "Ownership of inventions and software remains with the company.",
        top_k=1,
    )

    assert len(matches) == 1
    assert matches[0]["clause_label"] == "IP Ownership Assignment"
