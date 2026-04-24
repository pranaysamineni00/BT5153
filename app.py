"""Flask server for the LexScan legal clause classification interface."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from classifier import LegalClauseClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

_classifier: LegalClauseClassifier | None = None


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
    return jsonify({"mode": clf.mode, "status": "ready"})


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

    try:
        result = get_classifier().classify(text)
    except Exception as exc:
        logger.exception("Classification error")
        return jsonify({"error": f"Classification failed: {exc}"}), 500

    result["word_count"] = len(text.split())
    result["char_count"] = len(text)
    return jsonify(result)


if __name__ == "__main__":
    logger.info("Initializing Legal-BERT classifier...")
    get_classifier()
    clf = get_classifier()
    logger.info("Mode: %s | Server: http://127.0.0.1:5001", clf.mode)
    app.run(host="127.0.0.1", port=5001, debug=False)
