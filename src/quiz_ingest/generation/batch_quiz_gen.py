"""
generation/batch_quiz_gen.py

Generates B questions, then B sets of distractors, each in ONE call.
Both calls share the exact same `shared_prefix` (system instruction +
retrieved context) -- built once by build_shared_prefix() and passed
unchanged to both calls, so vLLM's prefix cache hits on the second call.

For groups after the first, generate_question_batch appends an
"already asked" addendum AFTER shared_prefix (see
build_avoid_repeat_addendum) so separate batch calls against the same
context don't independently rediscover the same salient fact and produce
near-duplicate questions -- prefix caching only reuses matching INPUT
tokens between calls, it carries no memory of what a previous call's
OUTPUT was, so that has to be threaded through explicitly.

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


def build_avoid_repeat_addendum(already_asked_questions: list[str]) -> str:
    """
    Appended AFTER the base shared_prefix (context block) for
    generate_question_batch calls in groups after the first -- so later
    groups don't independently rediscover the same most-salient fact.
    Real observed failure this fixes: with num_questions=10 and
    batch_size=8, group 1 (8 calls' worth) and group 2 (the remaining 2)
    are two separate, stateless LLM calls sharing the same context. Both
    independently picked the paper's single most prominent claim,
    producing two near-duplicate questions with nothing to prevent it --
    KV-cache/prefix-cache only reuses matching INPUT tokens, it carries no
    memory of a previous call's OUTPUT. This addendum is how that missing
    state gets passed along manually.

    Deliberately appended AFTER the base prefix, not interleaved into it,
    so vLLM's prefix cache still hits on the shared base portion (system
    instruction + retrieved context) across every call in every group --
    only this tail differs per group, and it's cheap: it doesn't touch
    the (much larger) context block that's the expensive part to
    re-prefill.
    """
    if not already_asked_questions:
        return ""
    listed = "\n".join(f"- {q}" for q in already_asked_questions)
    return (
        "\n\nThe following questions have ALREADY been asked in this quiz -- "
        "do NOT repeat them or write a close rephrasing of any of them. "
        f"Pick a genuinely different fact from the context instead:\n{listed}"
    )


async def generate_question_batch(
    backend: LLMBackend,
    *,
    shared_prefix: str,
    batch_size: int,
    already_asked_questions: list[str] | None = None,
) -> tuple[list[dict], CallTelemetry, list[int], str]:
    prompt_prefix = shared_prefix + build_avoid_repeat_addendum(already_asked_questions or [])
    item_prompts = [
        f"Write one multiple-choice question grounded in the reference context above."
        for _ in range(batch_size)
    ]
    result = await backend.generate_json_batch(
        shared_prefix=prompt_prefix,
        item_prompts=item_prompts,
        schema_hint=QUESTION_SCHEMA_HINT,
        max_tokens=200 * batch_size,
    )
    return result.items, result.telemetry, result.parse_failures, result.raw_text


async def generate_distractor_batch(
    backend: LLMBackend,
    *,
    shared_prefix: str,
    questions: list[dict],
) -> tuple[list[dict], CallTelemetry, list[int], str]:
    item_prompts = [
        f"For the question \"{q.get('question', '')}\" with correct answer "
        f"\"{q.get('correct_answer', '')}\", write 3 DIFFERENT wrong options: "
        f"one near-miss (plausible but factually incorrect -- NOT a reworded "
        f"or copied version of the correct answer), one common misconception, "
        f"one plausible-but-unrelated. Every option's wording must be clearly "
        f"distinguishable from the correct answer above -- do not repeat it."
        for q in questions
    ]
    result = await backend.generate_json_batch(
        shared_prefix=shared_prefix,
        item_prompts=item_prompts,
        schema_hint=DISTRACTOR_SCHEMA_HINT,
        max_tokens=150 * len(questions),
    )
    return result.items, result.telemetry, result.parse_failures, result.raw_text


def _distractor_duplicates_correct(correct_answer: str, distractors: list[dict]) -> bool:
    """
    True if any distractor is a literal (whitespace/case-normalized) copy
    of the correct answer -- see assemble_quiz_items and
    repair_duplicate_distractors docstrings for why this must never reach
    delivery, and why it's caught here instead of by one of the 3 eval
    metrics (none of which compare a distractor against correct_answer).
    """
    correct_norm = " ".join(correct_answer.split()).lower()
    if not correct_norm:
        return False
    for d in distractors:
        if " ".join(d.get("text", "").split()).lower() == correct_norm:
            return True
    return False


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

    Does NOT check for a distractor duplicating the correct answer -- that
    check now lives in repair_duplicate_distractors(), which gets a chance
    to fix it with a targeted regeneration before anything is dropped for
    that reason. This function only drops items that are structurally
    unusable (missing pieces), never items that are merely low-quality.
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


async def repair_duplicate_distractors(
    backend: LLMBackend,
    *,
    shared_prefix: str,
    items: list[QuizItem],
    max_retries: int = 1,
) -> tuple[list[QuizItem], list[CallTelemetry]]:
    """
    Finds items whose distractors literally duplicate the correct answer
    (see _distractor_duplicates_correct) and regenerates ONLY those items'
    distractors, instead of dropping the whole item outright.

    This matters because generate_distractor_batch produces distractors
    for a whole group in ONE completion -- if the model settles into the
    copy-the-answer pattern during that one completion, it tends to do it
    for every item in the group at once (observed directly: one real run
    had every item in a 5-question batch hit this simultaneously,
    delivering 0/5 despite every LLM call succeeding). Repairing only the
    affected subset, in a second smaller batched call, is both cheaper
    than dropping the whole group and avoids that correlated-failure
    pattern repeating on the retry (different, smaller batch composition).

    Items still duplicated after `max_retries` repair attempts are
    dropped -- never delivered with a broken option (same principle as
    assemble_quiz_items and the eval-score-exclusion logic in
    eval/ragas_eval.py: a defective item must never look like a normal,
    scoreable one).
    """
    telemetries: list[CallTelemetry] = []
    ok = [it for it in items if not _distractor_duplicates_correct(it.correct_answer, it.distractors)]
    bad = [it for it in items if it.index not in {o.index for o in ok}]

    for _attempt in range(max_retries):
        if not bad:
            break
        questions_payload = [
            {"index": it.index, "question": it.question, "correct_answer": it.correct_answer}
            for it in bad
        ]
        new_distractors, telemetry, failures, _raw = await generate_distractor_batch(
            backend, shared_prefix=shared_prefix, questions=questions_payload
        )
        telemetries.append(telemetry)
        by_index = {d.get("index"): d.get("distractors", []) for d in new_distractors}

        still_bad = []
        for it in bad:
            new_d = by_index.get(it.index)
            if new_d is None:
                still_bad.append(it)  # parse failure on the retry -- try again next loop (or drop)
                continue
            it.distractors = new_d
            if _distractor_duplicates_correct(it.correct_answer, it.distractors):
                still_bad.append(it)
            else:
                ok.append(it)
        bad = still_bad

    # Anything still bad after all retries is dropped, not delivered.
    return ok, telemetries
