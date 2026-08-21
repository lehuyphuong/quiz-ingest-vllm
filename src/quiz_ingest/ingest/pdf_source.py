"""
ingest/pdf_source.py

Reads a PDF, stops extracting as soon as MAX_EXTRACTED_CHARS is hit so
large files don't burn CPU/RAM on text that will never be used.
"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

MAX_RAW_BYTES = 60 * 1024 * 1024  # 60MB, rejected before opening the file
MAX_EXTRACTED_CHARS = 200_000


class PdfTooLargeError(ValueError):
    pass


def extract_pdf_text(path: str | Path) -> str:
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_RAW_BYTES:
        raise PdfTooLargeError(
            f"{path.name} is {size / 1e6:.1f}MB, exceeds the {MAX_RAW_BYTES / 1e6:.0f}MB cap"
        )

    reader = PdfReader(str(path))
    chunks: list[str] = []
    total_chars = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
        total_chars += len(text)
        if total_chars >= MAX_EXTRACTED_CHARS:
            break

    full_text = "\n".join(chunks)
    return full_text[:MAX_EXTRACTED_CHARS]
