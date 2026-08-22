import pytest

from quiz_ingest.generation.batch_quiz_gen import assemble_quiz_items
from quiz_ingest.ingest.chunking import Chunk
from quiz_ingest.llm.base import CallTelemetry, GenerateJsonBatchResult


def test_assemble_drops_indices_missing_distractors():
    questions = [
        {"index": 0, "question": "Q0", "correct_answer": "A0", "supporting_fact": "F0"},
        {"index": 1, "question": "Q1", "correct_answer": "A1", "supporting_fact": "F1"},
    ]
    distractors = [{"index": 0, "distractors": [{"text": "d1"}, {"text": "d2"}]}]  # index 1 missing
    items = assemble_quiz_items(questions, distractors, dropped_indices=set(), context_chunks=[])
    assert len(items) == 1
    assert items[0].index == 0


def test_assemble_drops_blank_question():
    questions = [{"index": 0, "question": "  ", "correct_answer": "A0", "supporting_fact": "F0"}]
    distractors = [{"index": 0, "distractors": []}]
    items = assemble_quiz_items(questions, distractors, dropped_indices=set(), context_chunks=[])
    assert items == []


def test_assemble_drops_explicitly_dropped_indices():
    questions = [{"index": 0, "question": "Q0", "correct_answer": "A0", "supporting_fact": "F0"}]
    distractors = [{"index": 0, "distractors": [{"text": "d"}]}]
    items = assemble_quiz_items(questions, distractors, dropped_indices={0}, context_chunks=[])
    assert items == []


def test_assemble_no_longer_drops_for_duplicate_distractor():
    # assemble_quiz_items now only does structural checks -- the
    # duplicate-distractor check moved to repair_duplicate_distractors
    # (tested below), which gets a chance to fix it before anything drops.
    questions = [{
        "index": 0,
        "question": "Q0",
        "correct_answer": "The Transformer uses self-attention instead of recurrence.",
        "supporting_fact": "F0",
    }]
    distractors = [{
        "index": 0,
        "distractors": [
            {"text": "The Transformer uses self-attention instead of recurrence.", "type": "near_miss"},
            {"text": "A genuinely different wrong statement.", "type": "misconception"},
            {"text": "Another unrelated wrong statement.", "type": "plausible_unrelated"},
        ],
    }]
    items = assemble_quiz_items(questions, distractors, dropped_indices=set(), context_chunks=[])
    assert len(items) == 1  # still present -- repair happens later, not here


def test_assemble_keeps_item_with_genuinely_distinct_distractors():
    questions = [{
        "index": 0, "question": "Q0", "correct_answer": "Answer A", "supporting_fact": "F0",
    }]
    distractors = [{
        "index": 0,
        "distractors": [
            {"text": "Answer B", "type": "near_miss"},
            {"text": "Answer C", "type": "misconception"},
            {"text": "Answer D", "type": "plausible_unrelated"},
        ],
    }]
    items = assemble_quiz_items(questions, distractors, dropped_indices=set(), context_chunks=[])
    assert len(items) == 1


class FakeBackend:
    """Minimal LLMBackend stub for eval tests -- returns canned responses."""

    def __init__(self, generate_responses, embed_vectors):
        self._responses = iter(generate_responses)
        self._embed_vectors = embed_vectors
        self.model_name = "fake-model"

    async def generate_json_batch(self, **kwargs):
        items, failures = next(self._responses)
        telemetry = CallTelemetry(
            call_type="generate_json", model="fake", prompt_tokens=1,
            completion_tokens=1, ttft_s=0.01, decode_tokens_per_second=10.0, wall_time_s=0.1,
        )
        return GenerateJsonBatchResult(items=items, raw_text="", telemetry=telemetry, parse_failures=failures)

    async def embed(self, texts):
        telemetry = CallTelemetry(
            call_type="embed", model="fake", prompt_tokens=1,
            completion_tokens=0, ttft_s=None, decode_tokens_per_second=None, wall_time_s=0.05,
        )
        return self._embed_vectors[: len(texts)], telemetry

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_faithfulness_excludes_parse_failures_not_zero_score():
    from quiz_ingest.eval.ragas_eval import score_faithfulness_and_relevance
    from quiz_ingest.generation.batch_quiz_gen import QuizItem

    items = [
        QuizItem(index=0, question="Q0", correct_answer="A0", supporting_fact="F0", distractors=[], source_chunk_ids=[]),
        QuizItem(index=1, question="Q1", correct_answer="A1", supporting_fact="F1", distractors=[], source_chunk_ids=[]),
    ]

    # decompose: both succeed / verify: both succeed / reverse-q: index 1 fails to parse
    responses = [
        ([{"index": 0, "statements": ["s0"]}, {"index": 1, "statements": ["s1"]}], []),
        ([{"index": 0, "verdicts": [1]}, {"index": 1, "verdicts": [0]}], []),
        ([{"index": 0, "reverse_questions": ["r0a", "r0b", "r0c"]}, {}], [1]),
    ]
    embed_vectors = [[1.0, 0.0]] * 10
    backend = FakeBackend(responses, embed_vectors)

    result = await score_faithfulness_and_relevance(
        backend, backend, shared_prefix="ctx", items=items
    )

    assert 1 in result.excluded_indices
    assert 1 not in result.answer_relevance
    assert 0 in result.answer_relevance
    assert 0 in result.faithfulness


class FakeDistractorRepairBackend:
    """Returns a canned distractors-batch response on each call, in order --
    used to test repair_duplicate_distractors' retry loop deterministically."""

    def __init__(self, distractor_responses):
        self._responses = iter(distractor_responses)
        self.model_name = "fake-model"

    async def generate_json_batch(self, **kwargs):
        items, failures = next(self._responses)
        telemetry = CallTelemetry(
            call_type="generate_json", model="fake", prompt_tokens=1,
            completion_tokens=1, ttft_s=0.01, decode_tokens_per_second=10.0, wall_time_s=0.1,
        )
        return GenerateJsonBatchResult(items=items, raw_text="", telemetry=telemetry, parse_failures=failures)


@pytest.mark.asyncio
async def test_repair_fixes_duplicate_distractor_on_retry():
    from quiz_ingest.generation.batch_quiz_gen import QuizItem, repair_duplicate_distractors

    items = [
        QuizItem(
            index=0, question="Q0", correct_answer="Correct answer text.",
            supporting_fact="F0",
            distractors=[
                {"text": "Correct answer text.", "type": "near_miss"},  # duplicate -- needs repair
                {"text": "Wrong B", "type": "misconception"},
                {"text": "Wrong C", "type": "plausible_unrelated"},
            ],
            source_chunk_ids=[],
        ),
        QuizItem(
            index=1, question="Q1", correct_answer="Another correct answer.",
            supporting_fact="F1",
            distractors=[
                {"text": "Genuinely wrong A", "type": "near_miss"},  # already fine, no repair needed
                {"text": "Genuinely wrong B", "type": "misconception"},
                {"text": "Genuinely wrong C", "type": "plausible_unrelated"},
            ],
            source_chunk_ids=[],
        ),
    ]

    # Only index 0 should be sent to the repair call; the response fixes it.
    repair_responses = [
        ([{"index": 0, "distractors": [
            {"text": "A real near-miss.", "type": "near_miss"},
            {"text": "Wrong B2", "type": "misconception"},
            {"text": "Wrong C2", "type": "plausible_unrelated"},
        ]}], []),
    ]
    backend = FakeDistractorRepairBackend(repair_responses)

    fixed_items, telemetries = await repair_duplicate_distractors(
        backend, shared_prefix="ctx", items=items, max_retries=1
    )

    assert len(fixed_items) == 2
    assert len(telemetries) == 1  # exactly one repair call made, not one per item
    by_index = {it.index: it for it in fixed_items}
    assert by_index[0].distractors[0]["text"] == "A real near-miss."
    assert by_index[1].distractors[0]["text"] == "Genuinely wrong A"  # untouched


@pytest.mark.asyncio
async def test_repair_drops_item_still_duplicated_after_max_retries():
    from quiz_ingest.generation.batch_quiz_gen import QuizItem, repair_duplicate_distractors

    items = [
        QuizItem(
            index=0, question="Q0", correct_answer="Correct answer text.",
            supporting_fact="F0",
            distractors=[{"text": "Correct answer text.", "type": "near_miss"}],
            source_chunk_ids=[],
        ),
    ]
    # Repair call returns the SAME duplicate again -- simulates a model that
    # can't be coaxed off the pattern within the retry budget.
    repair_responses = [
        ([{"index": 0, "distractors": [{"text": "Correct answer text.", "type": "near_miss"}]}], []),
    ]
    backend = FakeDistractorRepairBackend(repair_responses)

    fixed_items, telemetries = await repair_duplicate_distractors(
        backend, shared_prefix="ctx", items=items, max_retries=1
    )

    assert fixed_items == []  # dropped, not delivered with a broken option
    assert len(telemetries) == 1
