"""Utility helpers for text extraction from uploaded files."""

from __future__ import annotations

import io
import logging

from PyPDF2 import PdfReader
import docx


def extract_text_from_file(file_data: bytes, file_type: str) -> str:
    """Return raw text from a PDF or DOCX file.

    Args:
        file_data: File content as byte string.
        file_type: MIME type of the uploaded file.

    Returns:
        Concatenated text extracted from all pages or paragraphs.
    """
    text = ""
    try:
        if file_type == "application/pdf":
            reader = PdfReader(io.BytesIO(file_data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        elif file_type in (
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ):
            doc = docx.Document(io.BytesIO(file_data))
            text = "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as exc:  # pragma: no cover - log only
        logging.error("Error reading file: %s", exc)
    return text
