"""Build and cache a RAG knowledge base from the CUAD training split only."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    MANUAL_HARD_NEGATIVE_LABELS,
    ReviewConfig,
    expand_label_alias,
    get_review_config,
)
from data_loading import load_cuad
from preprocessing import build_contract_records, split_contract_records

try:
    import joblib
except ImportError:  # pragma: no cover - joblib is in requirements, this keeps import-time resilient
    joblib = None


_RAG_INDEX_CACHE: "RagIndex | None" = None


@dataclass
class RagExample:
    """Single train-split evidence item used for retrieval."""

    example_id: str
    clause_label: str
    clause_definition: str
    source_contract_id: str
    snippet_text: str
    answer_text: str
    question: str


@dataclass
class RagIndex:
    """In-memory RAG index plus metadata needed for retrieval."""

    entries: list[RagExample]
    embeddings: np.ndarray
    label_definitions: dict[str, str]
    entry_ids_by_label: dict[str, list[int]]
    hard_negative_entry_ids_by_label: dict[str, list[int]]
    label_neighbor_map: dict[str, list[str]]
    train_contract_ids: list[str]
    split_seed: int
    vector_store_type: str
    embedding_backend: str
    embedding_model_name: str
    confusion_map: dict[str, list[str]] = field(default_factory=dict)
    tfidf_vectorizer: Any = None
    _encoder: Any = field(default=None, repr=False, compare=False)
    _faiss_index: Any = field(default=None, repr=False, compare=False)

    def attach_runtime(self) -> None:
        """Restore non-serializable runtime helpers after loading from disk."""
        if self.embedding_backend == "sentence-transformers" and self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model_name)

        if self.vector_store_type != "faiss" or self._faiss_index is not None:
            return

        try:
            import faiss
        except ImportError:
            self.vector_store_type = "numpy"
            return

        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self.embeddings.astype(np.float32))
        self._faiss_index = index

    def encode_query(self, text: str) -> np.ndarray:
        self.attach_runtime()
        if self.embedding_backend == "sentence-transformers":
            encoded = self._encoder.encode([text], show_progress_bar=False)
            return _normalize_rows(np.asarray(encoded, dtype=np.float32))[0]

        if self.tfidf_vectorizer is None:
            raise RuntimeError("TF-IDF fallback vectorizer is missing from the RAG index.")

        encoded = self.tfidf_vectorizer.transform([text]).toarray().astype(np.float32)
        return _normalize_rows(encoded)[0]

    def search_entries(
        self,
        query_text: str,
        candidate_indices: list[int],
        top_k: int,
    ) -> list[RagExample]:
        if not candidate_indices or top_k <= 0:
            return []

        query = self.encode_query(query_text)
        if self.vector_store_type == "faiss" and self._faiss_index is not None:
            query_batch = query.reshape(1, -1).astype(np.float32)
            if len(candidate_indices) == len(self.entries):
                _, indices = self._faiss_index.search(query_batch, min(top_k, len(self.entries)))
                return [self.entries[idx] for idx in indices[0] if idx != -1]

            candidate_set = set(candidate_indices)
            search_k = min(max(top_k * 8, top_k + 8), len(self.entries))
            _, indices = self._faiss_index.search(query_batch, search_k)
            filtered = [idx for idx in indices[0] if idx in candidate_set and idx != -1]
            if len(filtered) >= top_k:
                return [self.entries[idx] for idx in filtered[:top_k]]

        subset = self.embeddings[candidate_indices]
        scores = subset @ query
        order = np.argsort(-scores)[:top_k]
        return [self.entries[candidate_indices[i]] for i in order]

    def save(self, path: str | None = None) -> None:
        if joblib is None:
            return

        if not path:
            return

        payload = {
            "entries": [asdict(item) for item in self.entries],
            "embeddings": self.embeddings,
            "label_definitions": self.label_definitions,
            "entry_ids_by_label": self.entry_ids_by_label,
            "hard_negative_entry_ids_by_label": self.hard_negative_entry_ids_by_label,
            "label_neighbor_map": self.label_neighbor_map,
            "train_contract_ids": self.train_contract_ids,
            "split_seed": self.split_seed,
            "vector_store_type": self.vector_store_type,
            "embedding_backend": self.embedding_backend,
            "embedding_model_name": self.embedding_model_name,
            "confusion_map": self.confusion_map,
            "tfidf_vectorizer": self.tfidf_vectorizer,
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, out_path)

    @classmethod
    def load(cls, path: str) -> "RagIndex | None":
        if joblib is None:
            return None

        in_path = Path(path)
        if not in_path.exists():
            return None

        payload = joblib.load(in_path)
        index = cls(
            entries=[RagExample(**item) for item in payload["entries"]],
            embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
            label_definitions=dict(payload["label_definitions"]),
            entry_ids_by_label={k: list(v) for k, v in payload["entry_ids_by_label"].items()},
            hard_negative_entry_ids_by_label={
                k: list(v) for k, v in payload["hard_negative_entry_ids_by_label"].items()
            },
            label_neighbor_map={k: list(v) for k, v in payload["label_neighbor_map"].items()},
            train_contract_ids=list(payload["train_contract_ids"]),
            split_seed=int(payload["split_seed"]),
            vector_store_type=str(payload["vector_store_type"]),
            embedding_backend=str(payload["embedding_backend"]),
            embedding_model_name=str(payload["embedding_model_name"]),
            confusion_map={k: list(v) for k, v in payload.get("confusion_map", {}).items()},
            tfidf_vectorizer=payload.get("tfidf_vectorizer"),
        )
        index.attach_runtime()
        return index


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-7, None)
    return matrix / norms


def _default_clause_definition(label: str) -> str:
    human = label.replace("/", " or ").replace("-", " ").replace(" Of ", " of ")
    return f"Contract language describing {human.lower()}."


def _question_to_definition(question: str, label: str) -> str:
    cleaned = " ".join((question or "").split()).strip()
    if not cleaned:
        return _default_clause_definition(label)

    cleaned = re.sub(r"^Highlight the parts \(if any\)\s+of (?:this )?contract\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^What\s+", "", cleaned, flags=re.I)
    cleaned = cleaned.rstrip(" ?.")
    if not cleaned:
        return _default_clause_definition(label)
    cleaned = cleaned[0].upper() + cleaned[1:]
    if not cleaned.endswith("."):
        cleaned += "."
    return cleaned


def _extract_answer_snippet(
    contract_text: str,
    answer_start: int,
    answer_text: str,
    window: int,
) -> str:
    start = max(0, int(answer_start) - window)
    end = min(len(contract_text), int(answer_start) + len(answer_text) + window)

    para_start = contract_text.rfind("\n\n", 0, int(answer_start))
    para_end = contract_text.find("\n\n", int(answer_start) + len(answer_text))
    if para_start != -1:
        start = max(start, para_start + 2)
    if para_end != -1:
        end = min(end, para_end)

    snippet = " ".join(contract_text[start:end].split())
    return snippet[: (window * 2 + max(len(answer_text), 80))].strip()


def _load_training_split(cuad_df, split_seed: int) -> tuple[Any, set[str]]:
    records = build_contract_records(cuad_df)
    train_records, _, _ = split_contract_records(records, seed=split_seed)
    train_titles = {record["contract_title"] for record in train_records}
    train_df = cuad_df[cuad_df["contract_title"].isin(train_titles)].copy()
    return train_df, train_titles


def _build_label_definitions(train_df) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for clause_label, group in train_df.groupby("clause_type", sort=True):
        questions = [item for item in group["question"].dropna().astype(str).unique().tolist() if item.strip()]
        if questions:
            shortest = min(questions, key=len)
            definitions[clause_label] = _question_to_definition(shortest, clause_label)
        else:
            definitions[clause_label] = _default_clause_definition(clause_label)
    return definitions


def _encode_texts(
    texts: list[str],
    config: ReviewConfig,
) -> tuple[np.ndarray, str, Any]:
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


def _build_label_neighbor_map(
    definitions: dict[str, str],
    config: ReviewConfig,
) -> dict[str, list[str]]:
    labels = sorted(definitions)
    if not labels:
        return {}

    texts = [f"Label: {label}\nDefinition: {definitions[label]}" for label in labels]
    embeddings, _, _ = _encode_texts(texts, config)
    similarity = embeddings @ embeddings.T

    neighbors: dict[str, list[str]] = {}
    for idx, label in enumerate(labels):
        order = np.argsort(-similarity[idx])
        neighbors[label] = [labels[j] for j in order if labels[j] != label][:5]
    return neighbors


def _build_hard_negative_entry_ids(
    entry_ids_by_label: dict[str, list[int]],
) -> dict[str, list[int]]:
    known_labels = set(entry_ids_by_label)
    output: dict[str, list[int]] = {}
    for label in known_labels:
        raw_negatives = MANUAL_HARD_NEGATIVE_LABELS.get(label, ())
        candidate_labels: list[str] = []
        for raw in raw_negatives:
            candidate_labels.extend(expand_label_alias(raw, known_labels=known_labels))
        candidate_labels = [item for item in candidate_labels if item != label]
        entry_ids: list[int] = []
        for negative_label in candidate_labels:
            entry_ids.extend(entry_ids_by_label.get(negative_label, []))
        output[label] = entry_ids
    return output


def build_rag_index(
    cuad_df: Any | None = None,
    *,
    config: ReviewConfig | None = None,
    confusion_map: dict[str, list[str]] | None = None,
    force_rebuild: bool = False,
) -> RagIndex:
    """Build a CUAD train-split-only RAG index for the review layer."""
    global _RAG_INDEX_CACHE

    config = config or get_review_config()

    if not force_rebuild and cuad_df is None and _RAG_INDEX_CACHE is not None:
        if (
            _RAG_INDEX_CACHE.embedding_backend == config.rag_embedding_backend
            and (
                _RAG_INDEX_CACHE.embedding_backend != "sentence-transformers"
                or _RAG_INDEX_CACHE.embedding_model_name == config.rag_embedding_model
            )
        ):
            if confusion_map:
                _RAG_INDEX_CACHE.confusion_map = {k: list(v) for k, v in confusion_map.items()}
            return _RAG_INDEX_CACHE

    if not force_rebuild and cuad_df is None:
        cached = RagIndex.load(config.rag_index_cache_path)
        if cached is not None:
            backend_matches = cached.embedding_backend == config.rag_embedding_backend
            model_matches = (
                cached.embedding_backend != "sentence-transformers"
                or cached.embedding_model_name == config.rag_embedding_model
            )
            if backend_matches and model_matches:
                if confusion_map:
                    cached.confusion_map = {k: list(v) for k, v in confusion_map.items()}
                _RAG_INDEX_CACHE = cached
                return cached

    if cuad_df is None:
        cuad_df = load_cuad(config.data_dir)

    train_df, train_titles = _load_training_split(cuad_df, split_seed=config.split_seed)
    definitions = _build_label_definitions(train_df)
    examples: list[RagExample] = []

    positive_rows = train_df[train_df["has_answer"]].copy()
    for row in positive_rows.itertuples(index=False):
        definition = definitions.get(row.clause_type, _default_clause_definition(row.clause_type))
        for idx, (answer_text, answer_start) in enumerate(zip(row.answer_texts, row.answer_starts)):
            snippet = _extract_answer_snippet(
                row.contract_text,
                int(answer_start),
                str(answer_text),
                config.example_snippet_chars,
            )
            if not snippet:
                continue

            examples.append(
                RagExample(
                    example_id=f"{row.contract_title}::{row.clause_type}::{idx}",
                    clause_label=row.clause_type,
                    clause_definition=definition,
                    source_contract_id=row.contract_title,
                    snippet_text=snippet,
                    answer_text=str(answer_text),
                    question=str(row.question),
                )
            )

    if not examples:
        raise RuntimeError("RAG index build produced zero train-split examples.")

    texts = [
        (
            f"Clause label: {item.clause_label}\n"
            f"Definition: {item.clause_definition}\n"
            f"Snippet: {item.snippet_text}"
        )
        for item in examples
    ]
    embeddings, backend_name, backend_helper = _encode_texts(texts, config)
    entry_ids_by_label: dict[str, list[int]] = {}
    for idx, item in enumerate(examples):
        entry_ids_by_label.setdefault(item.clause_label, []).append(idx)

    vector_store_type = "numpy"
    faiss_index = None
    try:
        import faiss

        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss_index.add(embeddings.astype(np.float32))
        vector_store_type = "faiss"
    except ImportError:
        faiss_index = None

    rag_index = RagIndex(
        entries=examples,
        embeddings=embeddings,
        label_definitions=definitions,
        entry_ids_by_label=entry_ids_by_label,
        hard_negative_entry_ids_by_label=_build_hard_negative_entry_ids(entry_ids_by_label),
        label_neighbor_map=_build_label_neighbor_map(definitions, config),
        train_contract_ids=sorted(train_titles),
        split_seed=config.split_seed,
        vector_store_type=vector_store_type,
        embedding_backend=backend_name,
        embedding_model_name=config.rag_embedding_model if backend_name == "sentence-transformers" else backend_name,
        confusion_map={k: list(v) for k, v in (confusion_map or {}).items()},
        tfidf_vectorizer=backend_helper if backend_name == "tfidf" else None,
        _encoder=backend_helper if backend_name == "sentence-transformers" else None,
        _faiss_index=faiss_index,
    )

    rag_index.save(config.rag_index_cache_path)
    _RAG_INDEX_CACHE = rag_index
    return rag_index
