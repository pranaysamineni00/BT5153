from __future__ import annotations

import io


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from an uploaded PDF, DOCX, or TXT file."""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        return _from_pdf(file_bytes)
    elif ext == "docx":
        return _from_docx(file_bytes)
    else:
        return file_bytes.decode("utf-8", errors="replace")


def _from_pdf(data: bytes) -> str:
    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _from_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
