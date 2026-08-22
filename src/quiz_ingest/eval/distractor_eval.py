"""
eval/distractor_eval.py

Distractor Diversity (Alhazmi et al., EMNLP 2024 survey,
aclanthology.org/2024.emnlp-main.799) -- pure cosine-similarity math on
embeddings, no LLM judge involved. Lower mean pairwise similarity among a
question's 3 distractors = higher diversity score (1 - mean_similarity).

One embed() call covers ALL distractor text for the whole batch of items
at once, not one call per question.
"""
from __future__ import annotations

import numpy as np

from quiz_ingest.generation.batch_quiz_gen import QuizItem
from quiz_ingest.llm.base import LLMBackend


async def score_distractor_diversity(
    backend: LLMBackend, items: list[QuizItem]
) -> tuple[dict[int, float], object | None]:
    all_texts: list[str] = []
    spans: list[tuple[int, int, int]] = []  # (item_index, start_pos, count)
    for it in items:
        texts = [d.get("text", "") for d in it.distractors if d.get("text")]
        if len(texts) < 2:
            continue
        start = len(all_texts)
        all_texts.extend(texts)
        spans.append((it.index, start, len(texts)))

    if not all_texts:
        return {}, None

    vectors, telemetry = await backend.embed(all_texts)
    vecs = np.array(vectors)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    unit_vecs = vecs / norms

    scores: dict[int, float] = {}
    for item_index, start, count in spans:
        group = unit_vecs[start : start + count]
        sim_matrix = group @ group.T
        # mean of off-diagonal entries only
        n = count
        off_diag_sum = sim_matrix.sum() - np.trace(sim_matrix)
        mean_sim = off_diag_sum / (n * (n - 1))
        scores[item_index] = float(1.0 - mean_sim)
    return scores, telemetry
