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


def test_assemble_drops_item_when_distractor_duplicates_correct_answer():
    # Real bug observed against Qwen3-4B-Instruct-2507: the model copied
    # correct_answer verbatim into the "near_miss" distractor slot.
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
    assert items == []  # whole item dropped, not delivered with a broken option


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
