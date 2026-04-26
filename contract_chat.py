"""Guardrailed contract chat built on document retrieval."""
from __future__ import annotations

import logging
import re
from typing import Any

from config import ReviewConfig, expand_label_alias, get_review_config
from document_rag import DocumentIndex, DocumentSearchResult
from openai_utils import get_openai_client, has_openai_package, openai_api_key
from retriever import RagRetriever, get_default_retriever

logger = logging.getLogger(__name__)

GUARDRAIL_MESSAGE = "I am just a legal contract support agent, I can't answer this."
GENERIC_SUGGESTED_QUERIES = [
    "What are the main obligations of each party?",
    "What termination rights or risks should I pay attention to?",
]
_ALLOWED_KEYWORDS = {
    "agreement",
    "clause",
    "contract",
    "customer",
    "duration",
    "effective",
    "end",
    "fee",
    "governing law",
    "indemn",
    "insurance",
    "liability",
    "license",
    "notice",
    "obligation",
    "party",
    "payment",
    "renew",
    "risk",
    "term",
    "terminate",
    "termination",
    "vendor",
    "warranty",
}
_REFUSAL_KEYWORDS = {
    "capital of",
    "currency",
    "doctor",
    "joke",
    "movie",
    "poem",
    "recipe",
    "restaurant",
    "stock price",
    "weather",
    "who won",
}


def chatbot_available(config: ReviewConfig | None = None) -> bool:
    cfg = config or get_review_config()
    return bool(cfg.enable_chatbot and openai_api_key() and has_openai_package())


def suggested_queries_for_contract(
    classification_result: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> list[str]:
    suggestions: list[str] = list(GENERIC_SUGGESTED_QUERIES)
    clauses = list((classification_result or {}).get("clauses", []))
    doc_type = (summary or {}).get("doc_type", "")

    for clause in clauses[:4]:
        label = (clause.get("clause") or "").strip()
        if not label:
            continue
        lowered = label.lower()
        if "termination" in lowered or "renewal" in lowered or "expiration" in lowered:
            suggestions.append("Does this contract allow early termination for convenience?")
        elif "liability" in lowered or "indemn" in lowered:
            suggestions.append(f"What does the {label.lower()} clause mean here?")
        elif "license" in lowered or "ip" in lowered:
            suggestions.append(f"What rights are being granted under the {label.lower()} clause?")
        else:
            suggestions.append(f"What should I know about the {label.lower()} clause?")

    if doc_type and "license" in doc_type.lower():
        suggestions.append("What limits are placed on using the licensed material?")

    unique: list[str] = []
    seen: set[str] = set()
    for item in suggestions:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
        if len(unique) == 4:
            break
    return unique


def _contract_question_allowed(question: str, classification_result: dict[str, Any] | None, history: list[dict[str, str]]) -> bool:
    lowered = " ".join((question or "").lower().split())
    if not lowered:
        return False
    if any(item in lowered for item in _REFUSAL_KEYWORDS):
        return False
    if any(item in lowered for item in _ALLOWED_KEYWORDS):
        return True
    if any(token in lowered for token in {"this", "that", "it", "here", "above"}) and history:
        return True

    clauses = list((classification_result or {}).get("clauses", []))
    for clause in clauses:
        label = (clause.get("clause") or "").strip().lower()
        if label and label in lowered:
            return True
    return False


def _format_citations(results: list[DocumentSearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item.chunk_id,
            "text": item.text,
            "score": round(item.score, 3),
            "start_char": item.start_char,
            "end_char": item.end_char,
        }
        for item in results
    ]


def _question_seeks_clause_guidance(question: str) -> bool:
    lowered = " ".join((question or "").lower().split())
    if not lowered:
        return False
    guidance_phrases = (
        "why should i be concerned",
        "why is this concerning",
        "why is this a risk",
        "what does this mean",
        "what does that mean",
        "what should i know",
        "what is the risk",
        "what risk",
        "what should i be worried",
        "what should i worry",
        "why does this matter",
        "is this risky",
        "is this a problem",
    )
    if any(phrase in lowered for phrase in guidance_phrases):
        return True
    return any(token in lowered for token in {"concern", "concerning", "risk", "risky", "matter", "mean", "worried", "worry"})


def _classification_clauses(classification_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    return list((classification_result or {}).get("clauses", []))


def _clause_names(clause: dict[str, Any]) -> list[str]:
    names: list[str] = []
    primary = (clause.get("clause") or "").strip()
    if primary:
        names.append(primary)
    for item in clause.get("co_clauses") or []:
        name = (item.get("clause") or "").strip()
        if name:
            names.append(name)
    return names


def _clause_detection_result(
    clause_focus: str | None,
    classification_result: dict[str, Any] | None,
) -> DocumentSearchResult | None:
    if not clause_focus:
        return None
    target = clause_focus.strip().lower()
    if not target:
        return None

    for clause in _classification_clauses(classification_result):
        clause_names = [item.lower() for item in _clause_names(clause)]
        if target not in clause_names:
            continue
        snippet = (clause.get("snippet") or "").strip()
        if not snippet:
            continue
        confidence = float(clause.get("confidence") or 0.0)
        safe_id = re.sub(r"[^a-z0-9]+", "-", target).strip("-") or "clause"
        return DocumentSearchResult(
            chunk_id=f"classifier-{safe_id}",
            text=snippet,
            score=confidence,
            start_char=-1,
            end_char=-1,
        )
    return None


def _pick_clause_focus(question: str, classification_result: dict[str, Any] | None, retriever: RagRetriever | None) -> str | None:
    lowered = (question or "").lower()
    clauses = _classification_clauses(classification_result)
    for clause in clauses:
        label = (clause.get("clause") or "").strip()
        if label and label.lower() in lowered:
            return label

    tokens = re.findall(r"[a-z0-9]+", lowered)
    max_window = min(5, len(tokens))
    known_labels = {name for clause in clauses for name in _clause_names(clause)}
    if retriever is not None:
        known_labels = known_labels | set(retriever.known_labels)
    for window_size in range(max_window, 0, -1):
        for start in range(0, len(tokens) - window_size + 1):
            phrase = " ".join(tokens[start : start + window_size])
            for candidate in expand_label_alias(phrase, known_labels=known_labels or None):
                if candidate in known_labels:
                    return candidate
    return clauses[0]["clause"] if clauses else None


class ContractChatbot:
    """RAG-backed contract support assistant with retrieval guardrails."""

    def __init__(
        self,
        document_index: DocumentIndex,
        *,
        config: ReviewConfig | None = None,
        retriever: RagRetriever | None = None,
    ) -> None:
        self.document_index = document_index
        self.config = config or get_review_config()
        self.retriever = retriever

    def _retriever(self) -> RagRetriever | None:
        if self.retriever is not None:
            return self.retriever
        try:
            self.retriever = get_default_retriever(config=self.config)
        except Exception:
            logger.exception("Failed to initialize CUAD retriever for chat background examples")
            self.retriever = None
        return self.retriever

    def _search_contract_evidence(
        self,
        message: str,
        *,
        clause_focus: str | None,
        classification_result: dict[str, Any] | None,
    ) -> list[DocumentSearchResult]:
        queries = [message]
        if clause_focus:
            queries.append(clause_focus)

        merged: dict[str, DocumentSearchResult] = {}
        search_width = max(self.config.chat_top_k * 2, self.config.chat_top_k)
        for query in queries:
            if not query.strip():
                continue
            for item in self.document_index.search(query, top_k=search_width):
                existing = merged.get(item.chunk_id)
                if existing is None or item.score > existing.score:
                    merged[item.chunk_id] = item

        detected = _clause_detection_result(clause_focus, classification_result)
        if detected is not None:
            existing = merged.get(detected.chunk_id)
            if existing is None or detected.score > existing.score:
                merged[detected.chunk_id] = detected

        return sorted(merged.values(), key=lambda item: item.score, reverse=True)[: self.config.chat_top_k]

    def _background_clause_support(
        self,
        *,
        clause_focus: str | None,
        message: str,
        retriever: RagRetriever | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        if not clause_focus or retriever is None:
            return "", []

        clause_definition = ""
        try:
            clause_definition = retriever.get_definition(clause_focus).strip()
        except Exception:
            logger.exception("Failed to fetch clause definition for chat")

        retrieved_examples: list[dict[str, Any]] = []
        try:
            retrieved_examples = retriever.retrieve_positive_examples(
                clause_focus,
                message,
                top_k=min(2, self.config.retriever_top_positives),
            )
        except Exception:
            logger.exception("Failed to fetch CUAD examples for chat")
        return clause_definition, retrieved_examples

    def answer(
        self,
        *,
        message: str,
        history: list[dict[str, str]] | None = None,
        classification_result: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        history = list(history or [])[-self.config.chat_history_turns :]
        suggestions = suggested_queries_for_contract(classification_result, summary)

        if not self.config.enable_chatbot:
            return self._base_payload("disabled", "Chatbot is disabled.", [], [], suggestions, len(history))
        if not chatbot_available(self.config):
            return self._base_payload("disabled", "OpenAI is not configured for chat.", [], [], suggestions, len(history))
        if not _contract_question_allowed(message, classification_result, history):
            return self._base_payload("refused", GUARDRAIL_MESSAGE, [], [], suggestions, len(history))

        clause_focus = _pick_clause_focus(message, classification_result, None)
        results = self._search_contract_evidence(
            message,
            clause_focus=clause_focus,
            classification_result=classification_result,
        )
        citations = _format_citations(results)
        strongest = max((item.score for item in results), default=0.0)
        retriever: RagRetriever | None = None
        clause_definition = ""
        retrieved_examples: list[dict[str, Any]] = []
        if strongest < self.config.chat_min_score:
            if clause_focus and _question_seeks_clause_guidance(message):
                retriever = self._retriever()
                clause_definition, retrieved_examples = self._background_clause_support(
                    clause_focus=clause_focus,
                    message=message,
                    retriever=retriever,
                )
                if clause_definition or retrieved_examples:
                    try:
                        answer_text = self._generate_answer(
                            message=message,
                            history=history,
                            citations=citations,
                            retrieved_examples=retrieved_examples,
                            clause_focus=clause_focus,
                            clause_definition=clause_definition,
                            allow_background_guidance=True,
                            evidence_limited=True,
                        )
                    except Exception:
                        logger.exception("Contract chat generation failed for clause guidance fallback")
                    else:
                        return self._base_payload(
                            "ready",
                            answer_text,
                            citations,
                            retrieved_examples,
                            suggestions,
                            len(history),
                        )
            return self._base_payload(
                "insufficient_evidence",
                f"{GUARDRAIL_MESSAGE} I could not find enough support for that in the uploaded contract.",
                citations,
                retrieved_examples,
                suggestions,
                len(history),
            )

        retriever = self._retriever()
        if clause_focus and retriever is not None:
            clause_definition, retrieved_examples = self._background_clause_support(
                clause_focus=clause_focus,
                message=message,
                retriever=retriever,
            )

        try:
            answer_text = self._generate_answer(
                message=message,
                history=history,
                citations=citations,
                retrieved_examples=retrieved_examples,
                clause_focus=clause_focus,
                clause_definition=clause_definition,
            )
        except Exception:
            logger.exception("Contract chat generation failed")
            return self._base_payload(
                "error",
                "The contract was analyzed, but the chat answer could not be generated.",
                citations,
                retrieved_examples,
                suggestions,
                len(history),
            )

        return self._base_payload(
            "ready",
            answer_text,
            citations,
            retrieved_examples,
            suggestions,
            len(history),
        )

    def _base_payload(
        self,
        status: str,
        answer: str,
        citations: list[dict[str, Any]],
        retrieved_examples: list[dict[str, Any]],
        suggestions: list[str],
        history_used: int,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "answer": answer,
            "citations": citations,
            "retrieved_examples": retrieved_examples,
            "suggested_queries": suggestions,
            "model": self.config.chat_model if status == "ready" else "",
            "history_used": history_used,
        }

    def _generate_answer(
        self,
        *,
        message: str,
        history: list[dict[str, str]],
        citations: list[dict[str, Any]],
        retrieved_examples: list[dict[str, Any]],
        clause_focus: str | None = None,
        clause_definition: str = "",
        allow_background_guidance: bool = False,
        evidence_limited: bool = False,
    ) -> str:
        evidence_lines = [
            f"[{idx + 1}] {item['text']}"
            for idx, item in enumerate(citations[: self.config.chat_top_k])
        ]
        example_lines = [
            f"- {item.get('clause_label', 'Example')}: {item.get('snippet_text', '')}"
            for item in retrieved_examples[:2]
            if item.get("snippet_text")
        ]
        focus_lines = []
        if clause_focus:
            focus_lines.append(f"Detected clause focus: {clause_focus}")
        if clause_definition:
            focus_lines.append(f"Clause definition: {clause_definition}")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a legal contract support agent. Answer only using the uploaded contract evidence "
                    "and optional CUAD background examples. Do not answer unrelated questions. Do not claim to be a lawyer. "
                    "If the evidence is not enough, say so plainly. Keep answers concise and practical. "
                    "For clause-explanation questions, you may explain the general concern or practical effect of the clause "
                    "using the clause definition and CUAD examples, but you must clearly separate that background from what the uploaded contract shows."
                ),
            }
        ]
        for turn in history[-self.config.chat_history_turns :]:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            content = (turn.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": content})

        user_prompt = (
            f"User question: {message}\n\n"
            f"{chr(10).join(focus_lines) if focus_lines else ''}\n\n"
            "Uploaded contract evidence:\n"
            f"{chr(10).join(evidence_lines) if evidence_lines else 'None'}\n\n"
            "Optional CUAD background examples:\n"
            f"{chr(10).join(example_lines) if example_lines else 'None'}\n\n"
            "Instructions:\n"
            f"{'- Start by saying the direct support in the uploaded contract is limited for this question.' + chr(10) if evidence_limited else ''}"
            f"{'- You may explain the general concern of the clause using the definition and CUAD examples, but do not present that background as a contract-specific fact.' + chr(10) if allow_background_guidance else '- Answer only from the uploaded contract evidence.' + chr(10)}"
            "- Treat CUAD examples as background only, not direct evidence.\n"
            "- If the answer is uncertain, say that clearly.\n"
            "- When helpful, mention citation numbers like [1] or [2]."
        )
        messages.append({"role": "user", "content": user_prompt})

        response = get_openai_client(timeout=45.0).chat.completions.create(
            model=self.config.chat_model,
            temperature=0.1,
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()
