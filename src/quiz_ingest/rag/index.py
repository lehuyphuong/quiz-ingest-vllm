"""
rag/index.py

Builds an index over document chunks and retrieves the top-k most relevant
for the quiz topic. Two stages:

  1. BM25 (local, free, no GPU call) narrows down to MAX_CHUNKS_TO_EMBED
     candidates -- kept even though embedding is self-hosted now (no GCP
     request-count quota to dodge anymore), because it's still a real
     latency win: fewer chunks means a shorter embed call AND a shorter
     rerank prompt, which matters for stage-level token/s telemetry.
  2. Embedding cosine similarity against the query, over ONE batched
     embed() call for all candidate chunks (not one call per chunk).

Rerank uses an INDEX-based response schema ({"index": int, "score": float}),
not an open string ID field -- this is a defensive choice carried over from
a real failure mode with small/weak models (see generation/schemas.py's
docstring), kept even though a capable model on GPU is less likely to need
it.
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from quiz_ingest.ingest.chunking import Chunk
from quiz_ingest.llm.base import LLMBackend

MAX_CHUNKS_TO_EMBED = 120


def _bm25_prefilter(chunks: list[Chunk], query: str, top_n: int) -> list[Chunk]:
    if len(chunks) <= top_n:
        return chunks
    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked_idx = np.argsort(scores)[::-1][:top_n]
    return [chunks[i] for i in ranked_idx]


def _cosine_topk(
    query_vec: list[float], chunk_vecs: list[list[float]], top_k: int
) -> list[int]:
    q = np.array(query_vec)
    q = q / (np.linalg.norm(q) + 1e-9)
    m = np.array(chunk_vecs)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    sims = m @ q
    return list(np.argsort(sims)[::-1][:top_k])


class RagIndex:
    def __init__(self, backend: LLMBackend):
        self._backend = backend
        self._chunks: list[Chunk] = []

    async def build_and_retrieve(
        self, chunks: list[Chunk], query: str, top_k: int = 12
    ) -> list[Chunk]:
        """
        Single entry point: prefilter -> embed candidates + query in ONE
        call -> cosine rerank -> return top_k chunks. This is retrieval
        for the WHOLE request, done once -- callers should not call this
        per-question (see the earlier quizrag-scale project's rag/index.py
        docstring for the reasoning; unchanged here).
        """
        candidates = _bm25_prefilter(chunks, query, MAX_CHUNKS_TO_EMBED)
        if not candidates:
            return []

        texts_to_embed = [query] + [c.text for c in candidates]
        vectors, _telemetry = await self._backend.embed(texts_to_embed)
        query_vec, chunk_vecs = vectors[0], vectors[1:]

        top_idx = _cosine_topk(query_vec, chunk_vecs, min(top_k, len(candidates)))
        return [candidates[i] for i in top_idx]
