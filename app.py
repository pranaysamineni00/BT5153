"""Flask server for the LexScan legal clause classification interface."""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from classifier import LegalClauseClassifier
from llm_summary import build_contract_summary, is_summary_enabled

_LOG_PATH = Path(__file__).with_name("dashboard_server.log")

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
    clf = get_classifier()
    return jsonify({
        "mode": clf.mode,
        "status": "ready",
        "llm_summary_enabled": is_summary_enabled(),
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

    try:
        result = get_classifier().classify(text)
    except Exception as exc:
        logger.exception("Classification error")
        return jsonify({"error": f"Classification failed: {exc}"}), 500

    _log_memory_snapshot("After clause classification")
    result["llm_summary"] = build_contract_summary(text)
    logger.info(
        "AI summary status | available=%s | status=%s",
        result["llm_summary"].get("available"),
        result["llm_summary"].get("status"),
    )
    _log_memory_snapshot("After AI summary")
    result["word_count"] = len(text.split())
    result["char_count"] = len(text)
    return jsonify(result)


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
