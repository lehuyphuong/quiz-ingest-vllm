"""
ingest/chunking.py

Deliberately simple char-window chunking with overlap. `chunk_size` and
`overlap` are the two knobs worth sweeping alongside model/batch-size when
optimizing Faithfulness -- see this repo's README, "Tuning chunking" --
because how much grounding context makes it into each chunk affects
Faithfulness independently of which model does the generating.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    id: int
    text: str
    source: str


def chunk_text(
    text: str, *, source: str, chunk_size: int = 1000, overlap: int = 150
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(id=idx, text=piece, source=source))
            idx += 1
        if end == n:
            break
        start = end - overlap
    return chunks
