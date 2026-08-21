"""
generation/batch_quiz_gen.py

Generates B questions, then B sets of distractors, each in ONE call.
Both calls share the exact same `shared_prefix` (system instruction +
retrieved context) -- built once by build_shared_prefix() and passed
unchanged to both calls, so vLLM's prefix cache hits on the second call.

QUESTION_BATCH_SIZE default of 8 is carried over from the earlier project
as a starting point, NOT re-derived for this repo's model/hardware --
re-tune it with scripts/bench_stage_timing.py before trusting it (see this
repo's README, "Tuning batch size").
"""
from __future__ import annotations

from dataclasses import dataclass

from quiz_ingest.generation.schemas import (
    DISTRACTOR_SCHEMA_HINT,
    NON_CONVERSATIONAL_SYSTEM_INSTRUCTION,
    QUESTION_SCHEMA_HINT,
)
from quiz_ingest.ingest.chunking import Chunk
from quiz_ingest.llm.base import CallTelemetry, LLMBackend

QUESTION_BATCH_SIZE = 8


@dataclass
class QuizItem:
    index: int
    question: str
    correct_answer: str
    supporting_fact: str
    distractors: list[dict]  # [{"text": ..., "type": ...}, ...]
    source_chunk_ids: list[int]


def build_shared_prefix(topic: str, context_chunks: list[Chunk]) -> str:
    """
    Built ONCE per job/group and reused byte-for-byte across every call in
    that group (question gen, distractor gen, both eval steps) -- this is
    the string that needs to stay identical for prefix caching to pay off.
    Do not string-interpolate anything item-specific in here.
    """
    context_block = "\n\n".join(
        f"[source #{c.id}] {c.text}" for c in context_chunks
    )
    return (
        f"{NON_CONVERSATIONAL_SYSTEM_INSTRUCTION}\n\n"
        f"Topic: {topic}\n\n"
        f"Reference context (use ONLY facts stated here):\n{context_block}"
    )


async def generate_question_batch(
    backend: LLMBackend, *, shared_prefix: str, batch_size: int
) -> tuple[list[dict], CallTelemetry, list[int]]:
    item_prompts = [
        f"Write one multiple-choice question grounded in the reference context above."
        for _ in range(batch_size)
    ]
    result = await backend.generate_json_batch(
        shared_prefix=shared_prefix,
        item_prompts=item_prompts,
        schema_hint=QUESTION_SCHEMA_HINT,
        max_tokens=200 * batch_size,
    )
    return result.items, result.telemetry, result.parse_failures


async def generate_distractor_batch(
    backend: LLMBackend,
    *,
    shared_prefix: str,
    questions: list[dict],
) -> tuple[list[dict], CallTelemetry, list[int]]:
    item_prompts = [
        f"For the question \"{q.get('question', '')}\" with correct answer "
        f"\"{q.get('correct_answer', '')}\", write 3 wrong options: one "
        f"near-miss, one common misconception, one plausible-but-unrelated."
        for q in questions
    ]
    result = await backend.generate_json_batch(
        shared_prefix=shared_prefix,
        item_prompts=item_prompts,
        schema_hint=DISTRACTOR_SCHEMA_HINT,
        max_tokens=150 * len(questions),
    )
    return result.items, result.telemetry, result.parse_failures


def assemble_quiz_items(
    questions: list[dict],
    distractors: list[dict],
    dropped_indices: set[int],
    context_chunks: list[Chunk],
) -> list[QuizItem]:
    """
    Drops any index present in `dropped_indices` (parse failures from
    EITHER call) instead of delivering a partially-filled item -- see
    this repo's README, "Eval score integrity", for why silently keeping
    a half-populated item is worse than dropping it. Also drops any item
    whose question field is blank even if it otherwise parsed, carried
    over from a real bug in the earlier project.
    """
    by_index_distractors = {d.get("index"): d for d in distractors}
    items: list[QuizItem] = []
    for q in questions:
        idx = q.get("index")
        if idx in dropped_indices:
            continue
        question_text = (q.get("question") or "").strip()
        if not question_text:
            continue
        d = by_index_distractors.get(idx)
        if d is None:
            continue
        items.append(
            QuizItem(
                index=idx,
                question=question_text,
                correct_answer=q.get("correct_answer", ""),
                supporting_fact=q.get("supporting_fact", ""),
                distractors=d.get("distractors", []),
                source_chunk_ids=[c.id for c in context_chunks],
            )
        )
    return items
