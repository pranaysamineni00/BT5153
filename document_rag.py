"""Document-level retrieval for contract chat."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import ReviewConfig, get_review_config


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    paragraph_start: int
    paragraph_end: int


@dataclass
class DocumentSearchResult:
    chunk_id: str
    text: str
    score: float
    start_char: int
    end_char: int


@dataclass
class DocumentIndex:
    chunks: list[DocumentChunk]
    embeddings: np.ndarray
    embedding_backend: str
    embedding_model_name: str
    tfidf_vectorizer: Any = None
    encoder: Any = None

    def encode_query(self, text: str) -> np.ndarray:
        if self.embedding_backend == "sentence-transformers" and self.encoder is not None:
            encoded = self.encoder.encode([text], show_progress_bar=False)
            return _normalize_rows(np.asarray(encoded, dtype=np.float32))[0]

        encoded = self.tfidf_vectorizer.transform([text]).toarray().astype(np.float32)
        return _normalize_rows(encoded)[0]

    def search(self, query_text: str, top_k: int = 4) -> list[DocumentSearchResult]:
        if not self.chunks or top_k <= 0:
            return []

        query = self.encode_query(query_text)
        scores = self.embeddings @ query
        order = np.argsort(-scores)[: min(top_k, len(self.chunks))]
        return [
            DocumentSearchResult(
                chunk_id=self.chunks[idx].chunk_id,
                text=self.chunks[idx].text,
                score=float(scores[idx]),
                start_char=self.chunks[idx].start_char,
                end_char=self.chunks[idx].end_char,
            )
            for idx in order
        ]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-7, None)
    return matrix / norms


def chunk_contract_text(text: str, chunk_chars: int, overlap_chars: int) -> list[DocumentChunk]:
    paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
    if not paragraphs:
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            return []
        return [
            DocumentChunk(
                chunk_id="chunk-1",
                text=cleaned[:chunk_chars],
                start_char=0,
                end_char=min(len(cleaned), chunk_chars),
                paragraph_start=0,
                paragraph_end=0,
            )
        ]

    chunks: list[DocumentChunk] = []
    para_offsets: list[tuple[int, int]] = []
    cursor = 0
    for para in paragraphs:
        start = text.find(para, cursor)
        if start == -1:
            start = cursor
        end = start + len(para)
        para_offsets.append((start, end))
        cursor = end

    buffer: list[str] = []
    chunk_start_para = 0
    chunk_text = ""

    def flush(end_para: int) -> None:
        if not buffer:
            return
        combined = " ".join(buffer).strip()
        if not combined:
            return
        start_char = para_offsets[chunk_start_para][0]
        end_char = para_offsets[end_para][1]
        chunks.append(
            DocumentChunk(
                chunk_id=f"chunk-{len(chunks) + 1}",
                text=combined,
                start_char=start_char,
                end_char=end_char,
                paragraph_start=chunk_start_para,
                paragraph_end=end_para,
            )
        )

    for idx, para in enumerate(paragraphs):
        candidate = f"{chunk_text}\n\n{para}".strip() if chunk_text else para
        if chunk_text and len(candidate) > chunk_chars:
            flush(idx - 1)
            if overlap_chars > 0 and buffer:
                overlap_text = chunk_text[-overlap_chars:].strip()
                buffer = [overlap_text] if overlap_text else []
            else:
                buffer = []
            chunk_start_para = idx
            chunk_text = " ".join(buffer).strip()
            candidate = f"{chunk_text}\n\n{para}".strip() if chunk_text else para

        if not buffer:
            chunk_start_para = idx
        buffer.append(para)
        chunk_text = candidate

    flush(len(paragraphs) - 1)
    return chunks


def _encode_texts(texts: list[str], config: ReviewConfig) -> tuple[np.ndarray, str, Any]:
    preferred_backend = (config.rag_embedding_backend or "tfidf").strip().lower()

    if preferred_backend == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=40_000, ngram_range=(1, 2), sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
        return _normalize_rows(matrix), "tfidf", vectorizer

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=40_000, ngram_range=(1, 2), sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
        return _normalize_rows(matrix), "tfidf", vectorizer

    encoder = SentenceTransformer(config.rag_embedding_model)
    matrix = encoder.encode(texts, show_progress_bar=False)
    return _normalize_rows(np.asarray(matrix, dtype=np.float32)), "sentence-transformers", encoder


def build_document_index(text: str, config: ReviewConfig | None = None) -> DocumentIndex:
    config = config or get_review_config()
    chunks = chunk_contract_text(text, config.chat_chunk_chars, config.chat_chunk_overlap)
    if not chunks:
        raise RuntimeError("Document chunking produced zero chunks.")

    embeddings, backend_name, helper = _encode_texts([item.text for item in chunks], config)
    return DocumentIndex(
        chunks=chunks,
        embeddings=embeddings,
        embedding_backend=backend_name,
        embedding_model_name=config.rag_embedding_model if backend_name == "sentence-transformers" else backend_name,
        tfidf_vectorizer=helper if backend_name == "tfidf" else None,
        encoder=helper if backend_name == "sentence-transformers" else None,
    )
