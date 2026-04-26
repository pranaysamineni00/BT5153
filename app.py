"""Flask server for the LexScan legal clause classification interface."""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import gc
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from classifier import LegalClauseClassifier
from contract_chat import ContractChatbot, chatbot_available, suggested_queries_for_contract
from config import get_review_config
from document_rag import build_document_index
from llm_summary import build_contract_summary, is_summary_enabled
from openai_utils import get_openai_client, has_openai_package, openai_api_key
from risk_tagger import enrich_clause_risks, recompute_risk_summary
from review_pipeline import review_contract_predictions

_LOG_PATH = Path(__file__).with_name("dashboard_server.log")

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_PATH, encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

_classifier: LegalClauseClassifier | None = None

# In-memory cache of extracted document text keyed by a generated id. Used by
# the lazy /api/excerpt endpoint so the client doesn't have to re-POST the
# entire document on every card expansion. Bounded LRU — oldest evicted first.
_DOC_CACHE_MAX = int(os.getenv("LEXSCAN_DOC_CACHE_MAX", "6"))
_document_cache: "OrderedDict[str, str]" = OrderedDict()
_chat_state_cache: "OrderedDict[str, dict]" = OrderedDict()


def _cache_document(text: str, chat_state: dict | None = None) -> str:
    doc_id = uuid4().hex
    _document_cache[doc_id] = text
    if chat_state is not None:
        _chat_state_cache[doc_id] = chat_state
    while len(_document_cache) > _DOC_CACHE_MAX:
        stale_doc_id, _ = _document_cache.popitem(last=False)
        _chat_state_cache.pop(stale_doc_id, None)
    return doc_id


def _lookup_document(doc_id: str) -> str | None:
    if doc_id in _document_cache:
        _document_cache.move_to_end(doc_id)
        return _document_cache[doc_id]
    return None


def _lookup_chat_state(doc_id: str) -> dict | None:
    if doc_id in _chat_state_cache:
        _chat_state_cache.move_to_end(doc_id)
        return _chat_state_cache[doc_id]
    return None


def _rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(usage / (1024 * 1024), 1)
    return round(usage / 1024, 1)


def _log_memory_snapshot(context: str) -> None:
    rss = _rss_mb()
    if rss is None:
        return
    logger.info("%s | rss=%.1f MB", context, rss)


def _status_mode() -> str:
    if _classifier is not None:
        return _classifier.mode

    preference = os.getenv("LEXSCAN_CLASSIFIER_MODE", "baseline").strip().lower()
    base_dir = Path(__file__).parent
    checkpoint_path = base_dir / "models" / "Legal-BERT_(CUAD).pt"
    baseline_path = base_dir / "checkpoints" / "tfidf_lr_artifacts.joblib"
    if preference == "heuristic":
        return "heuristic"
    if preference == "baseline":
        return "baseline" if baseline_path.exists() else "heuristic"
    if preference == "model":
        return "model" if checkpoint_path.exists() else "heuristic"
    if checkpoint_path.exists():
        return "model"
    if baseline_path.exists():
        return "baseline"
    return "heuristic"


def get_classifier() -> LegalClauseClassifier:
    global _classifier
    if _classifier is None:
        _classifier = LegalClauseClassifier()
    return _classifier


def extract_text(file_path: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore[no-redef]
            except ImportError:
                raise RuntimeError("PDF support requires pypdf: pip install pypdf")
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if ext in (".docx", ".doc"):
        try:
            import docx
        except ImportError:
            raise RuntimeError("DOCX support requires python-docx: pip install python-docx")
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    # Plain text — try common encodings
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=enc) as fh:
                return fh.read()
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Could not decode file as text.")


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def status():
    review_config = get_review_config()
    return jsonify({
        "mode": _status_mode(),
        "status": "ready",
        "llm_summary_enabled": is_summary_enabled(),
        "second_stage_review_enabled": review_config.enable_second_stage_review,
        "chatbot_enabled": chatbot_available(review_config),
    })


@app.route("/api/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file attached."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".doc", ".txt"):
        return jsonify({"error": f"Unsupported format '{ext}'. Upload PDF, DOCX, or TXT."}), 400

    logger.info(
        "Classification request received | file=%s | request_bytes=%s",
        file.filename,
        request.content_length or "unknown",
    )
    _log_memory_snapshot("Before extraction")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        text = extract_text(tmp_path, file.filename)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not text.strip():
        return jsonify({"error": "No readable text found in this file."}), 400

    logger.info(
        "Text extracted | file=%s | chars=%d | words=%d",
        file.filename,
        len(text),
        len(text.split()),
    )
    _log_memory_snapshot("After extraction")

    clf = get_classifier()
    try:
        result = clf.classify(text)
    except Exception as exc:
        logger.exception("Classification error")
        return jsonify({"error": f"Classification failed: {exc}"}), 500

    _log_memory_snapshot("After clause classification")
    review_config = get_review_config()
    try:
        result["second_stage_review"] = review_contract_predictions(
            text,
            clf,
            classification_result=result,
            config=review_config,
        )
    except Exception:
        logger.exception("Second-stage review error")
        result["second_stage_review"] = {
            "review_status": "ERROR",
            "items": [],
            "decision_counts": {
                "ACCEPT": 0,
                "REJECT": 0,
                "RERANK_LABEL": 0,
                "HUMAN_REVIEW": 0,
            },
            "message": "Second-stage review failed during this request.",
        }
    _log_memory_snapshot("After second-stage review")
    try:
        result["clauses"] = enrich_clause_risks(
            list(result.get("clauses", [])),
            review_items=result["second_stage_review"].get("items", []),
            config=review_config,
        )
        result["risk_summary"] = recompute_risk_summary(result["clauses"])
    except Exception:
        logger.exception("Risk enrichment error")
    result["llm_summary"] = build_contract_summary(text)
    logger.info(
        "AI summary status | available=%s | status=%s",
        result["llm_summary"].get("available"),
        result["llm_summary"].get("status"),
    )
    _log_memory_snapshot("After AI summary")
    result["word_count"] = len(text.split())
    result["char_count"] = len(text)
    chat_state = {
        "history": [],
        "classification_result": {"clauses": list(result.get("clauses", []))},
        "summary": result["llm_summary"],
        "suggested_queries": suggested_queries_for_contract(result, result["llm_summary"]),
        "history_limit": review_config.chat_history_turns,
        "status": "ready" if chatbot_available(review_config) else "disabled",
        "chat_error": "",
        "document_index": None,
    }

    result["document_id"] = _cache_document(text, chat_state=chat_state)
    result["chatbot"] = {
        "status": chat_state["status"],
        "suggested_queries": chat_state["suggested_queries"],
        "history_limit": chat_state["history_limit"],
    }
    gc.collect()
    return jsonify(result)

@app.route("/api/excerpt", methods=["POST"])
def excerpt():
    payload = request.get_json(silent=True) or {}
    doc_id = (payload.get("document_id") or "").strip()
    clause = (payload.get("clause") or "").strip()

    if not doc_id or not clause:
        return jsonify({"error": "Missing 'document_id' or 'clause'."}), 400

    text = _lookup_document(doc_id)
    if text is None:
        return jsonify({"error": "Document not in cache — please re-upload."}), 404

    try:
        snippet = get_classifier().extract_excerpt(text, clause)
    except Exception as exc:
        logger.exception("Excerpt extraction error")
        return jsonify({"error": f"Excerpt extraction failed: {exc}"}), 500

    return jsonify({"excerpt": snippet})


@app.route("/api/explain", methods=["POST"])
def explain():
    payload = request.get_json(silent=True) or {}
    excerpt = (payload.get("excerpt") or "").strip()
    clause = (payload.get("clause") or "").strip()

    if not excerpt:
        return jsonify({"error": "Missing 'excerpt' in request body."}), 400

    if not openai_api_key():
        return jsonify({
            "error": "OpenAI API key not configured. Add OPENAI_API_KEY to the .env file."
        }), 500

    if not has_openai_package():
        return jsonify({"error": "openai package not installed: pip install openai"}), 500

    excerpt_for_prompt = excerpt[:4000]
    user_prompt = (
        f"The following is an excerpt from a legal contract"
        f"{f' identified as a \"{clause}\" clause' if clause else ''}.\n"
        f"Explain in plain, simple terms what this clause means and why it matters "
        f"to a non-lawyer (e.g., a business owner or procurement manager). "
        f"Keep it under 120 words.\n\n"
        f"Excerpt:\n\"\"\"\n{excerpt_for_prompt}\n\"\"\""
    )

    try:
        response = get_openai_client(timeout=30.0).chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a legal-tech assistant who explains contract clauses in plain English for non-lawyers."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        explanation = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.exception("OpenAI explain error")
        return jsonify({"error": f"LLM call failed: {exc}"}), 500

    return jsonify({"explanation": explanation})


@app.route("/api/chat/session", methods=["POST"])
def chat_session():
    payload = request.get_json(silent=True) or {}
    doc_id = (payload.get("document_id") or "").strip()
    if not doc_id:
        return jsonify({"error": "Missing 'document_id'."}), 400

    state = _lookup_chat_state(doc_id)
    if state is None:
        return jsonify({"error": "Document not in cache — please re-upload."}), 404

    return jsonify({
        "status": state.get("status", "error"),
        "suggested_queries": state.get("suggested_queries", []),
        "history_limit": state.get("history_limit", 0),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    doc_id = (payload.get("document_id") or "").strip()
    message = (payload.get("message") or "").strip()
    if not doc_id or not message:
        return jsonify({"error": "Missing 'document_id' or 'message'."}), 400

    state = _lookup_chat_state(doc_id)
    if state is None:
        return jsonify({"error": "Document not in cache — please re-upload."}), 404

    document_index = state.get("document_index")
    if document_index is None:
        text = _lookup_document(doc_id)
        if text is None:
            return jsonify({"error": "Document not in cache — please re-upload."}), 404
        try:
            document_index = build_document_index(text, config=get_review_config())
            state["document_index"] = document_index
            _log_memory_snapshot("After chat document index build")
        except Exception:
            logger.exception("Document chat index build failed")
            state["status"] = "error"
            return jsonify({
                "status": "error",
                "answer": "The uploaded contract could not be prepared for chat.",
                "citations": [],
                "retrieved_examples": [],
                "suggested_queries": state.get("suggested_queries", []),
                "model": "",
                "history_used": 0,
            }), 200

    bot = ContractChatbot(document_index, config=get_review_config())
    history = list(state.get("history", []))
    response_payload = bot.answer(
        message=message,
        history=history,
        classification_result=state.get("classification_result"),
        summary=state.get("summary"),
    )

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response_payload.get("answer", "")})
    state["history"] = history[-(get_review_config().chat_history_turns * 2) :]
    gc.collect()
    return jsonify(response_payload)


def serve_app() -> None:
    """Run the dashboard with a production-style local server when available."""
    logger.info("Server log file: %s", _LOG_PATH)
    logger.info("Initializing Legal-BERT classifier...")
    get_classifier()
    clf = get_classifier()
    logger.info("Mode: %s | Server: http://127.0.0.1:5001", clf.mode)

    try:
        from waitress import serve
    except ImportError:
        logger.warning(
            "waitress is not installed; falling back to Flask's development server."
        )
        app.run(
            host="127.0.0.1",
            port=5001,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
        return

    logger.info("Serving dashboard with Waitress.")
    serve(
        app,
        host="127.0.0.1",
        port=5001,
        threads=8,
        connection_limit=100,
        channel_timeout=120,
        cleanup_interval=30,
    )


if __name__ == "__main__":
    serve_app()
