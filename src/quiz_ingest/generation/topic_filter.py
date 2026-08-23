"""
generation/topic_filter.py

Catches wholesale off-topic hallucination: a generated item whose
question+correct_answer has nothing to do with ANY retrieved context
chunk, despite being valid, well-formed JSON. Observed directly in a
real load test run: a request grounded in the "Attention Is All You
Need" paper, topic "training efficiency of the transformer", produced a
fully-formed question about mitochondria and cellular respiration --
the model's own prose even said "the provided reference context does
not contain any information about [the topic]" and then generated the
biology content anyway (a known LLM failure mode: falling back to a
memorized template example when genuinely uncertain, rather than
declining or asking a weaker-but-grounded question).

None of the 3 eval metrics catch this on their own: a hallucinated
question can still be perfectly FAITHFUL (its correct_answer is
internally consistent with its own self-reported grounding) and have
high ANSWER RELEVANCE (the reverse-questions match the given question
just fine) -- both metrics operate on the item in isolation, neither
one checks "does this have anything to do with the actual source
document at all". That's what this file adds.
"""
from __future__ import annotations

import numpy as np

from quiz_ingest.generation.batch_quiz_gen import QuizItem
from quiz_ingest.ingest.chunking import Chunk
from quiz_ingest.llm.base import LLMBackend

# Deliberately loose: this is a coarse "has anything to do with the
# source at all" sanity check, not a precision grounding metric (that's
# Faithfulness's job, checked against real context, see eval/ragas_eval.py).
# Not calibrated against a labeled dataset -- treat as a starting point,
# tune against your own false-positive/false-negative rate once you have
# a batch of real runs to look at.
DEFAULT_OFF_TOPIC_SIMILARITY_THRESHOLD = 0.3


async def detect_off_topic_indices(
    embed_backend: LLMBackend,
    *,
    items: list[QuizItem],
    context_chunks: list[Chunk],
    threshold: float = DEFAULT_OFF_TOPIC_SIMILARITY_THRESHOLD,
) -> set[int]:
    """
    ONE batched embed() call for all items + all context chunks. For each
    item, takes the MAX cosine similarity across all context chunks (not
    the average) -- an item only needs to be grounded in ONE chunk to be
    legitimate, since retrieval already narrowed context_chunks down to
    the top_k most relevant for the whole job, not all of which are
    relevant to any single question.
    """
    if not items or not context_chunks:
        return set()

    item_texts = [f"{it.question} {it.correct_answer}" for it in items]
    context_texts = [c.text for c in context_chunks]
    vectors, _telemetry = await embed_backend.embed(item_texts + context_texts)

    item_vecs = np.array(vectors[: len(item_texts)])
    context_vecs = np.array(vectors[len(item_texts) :])
    item_unit = item_vecs / (np.linalg.norm(item_vecs, axis=1, keepdims=True) + 1e-9)
    context_unit = context_vecs / (np.linalg.norm(context_vecs, axis=1, keepdims=True) + 1e-9)

    sims = item_unit @ context_unit.T  # (n_items, n_chunks)
    max_sim_per_item = sims.max(axis=1)

    return {it.index for it, max_sim in zip(items, max_sim_per_item) if max_sim < threshold}
