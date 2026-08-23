import pytest

from quiz_ingest.config import PipelineConfig
from quiz_ingest.llm.base import CallTelemetry, GenerateJsonBatchResult
from quiz_ingest import pipeline as pipeline_module


class ScriptedBackend:
    """
    Drives generate_json_batch from a fixed, ordered queue of canned
    responses (one per expected call, in the exact order the pipeline
    makes them). embed() is content-based instead of queued: any text
    containing the literal marker "OFFTOPIC" gets an orthogonal vector,
    everything else (including the retrieved context chunk) gets an
    aligned one -- lets detect_off_topic_indices behave deterministically
    without hand-crafting exact vector counts for every call site.
    """

    def __init__(self, generate_responses):
        self._gen_responses = iter(generate_responses)
        self.model_name = "fake"

    async def generate_json_batch(self, *, shared_prefix, item_prompts, schema_hint, max_tokens):
        items, failures = next(self._gen_responses)
        telemetry = CallTelemetry(
            call_type="generate_json", model="fake", prompt_tokens=10, completion_tokens=10,
            ttft_s=0.01, decode_tokens_per_second=50.0, wall_time_s=0.05,
        )
        return GenerateJsonBatchResult(items=items, raw_text="[scripted]", telemetry=telemetry, parse_failures=failures)

    async def embed(self, texts):
        vectors = [[0.0, 1.0] if "OFFTOPIC" in t else [1.0, 0.0] for t in texts]
        telemetry = CallTelemetry(
            call_type="embed", model="fake", prompt_tokens=5, completion_tokens=0,
            ttft_s=None, decode_tokens_per_second=None, wall_time_s=0.02,
        )
        return vectors, telemetry

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_pipeline_drops_parse_failed_and_off_topic_without_index_bug(monkeypatch, tmp_path):
    """
    Regression test for a real indexing bug introduced while moving the
    off-topic filter earlier in the pipeline: q_failures (positions in
    the ORIGINAL questions list) were briefly being used to index into a
    list already filtered by off-topic detection, which could silently
    grab the wrong item or raise IndexError. Scripts ONE parse failure
    (position 1) and ONE off-topic item (position 2) in the SAME batch of
    3, and asserts the pipeline runs to completion delivering exactly the
    one genuinely good item (index 0) -- no crash, no wrong item smuggled
    through, no leftover placeholder.
    """
    monkeypatch.setattr(
        pipeline_module, "extract_pdf_text",
        lambda path: "This is context about a normal on-topic subject with enough text to form a chunk.",
    )

    fake = ScriptedBackend(generate_responses=[
        # generate_questions: 3 requested, position 1 fails to parse
        (
            [
                {"index": 0, "question": "Q0 on-topic", "correct_answer": "A0", "supporting_fact": "F0"},
                {},
                {"index": 2, "question": "Q2 OFFTOPIC", "correct_answer": "A2 OFFTOPIC", "supporting_fact": "F2"},
            ],
            [1],
        ),
        # generate_distractors: only called for the 1 surviving (parsed + on-topic) question
        (
            [{"index": 0, "distractors": [
                {"text": "d1", "type": "near_miss"},
                {"text": "d2", "type": "misconception"},
                {"text": "d3", "type": "plausible_unrelated"},
            ]}],
            [],
        ),
        ([{"index": 0, "statements": ["s0"]}], []),           # eval: decompose
        ([{"index": 0, "verdicts": [1]}], []),                 # eval: verify
        ([{"index": 0, "reverse_questions": ["r1", "r2", "r3"]}], []),  # eval: reverse-questions
    ])

    monkeypatch.setattr(pipeline_module, "_build_backends", lambda config: (fake, fake))

    config = PipelineConfig(backend="instance", vllm_base_url="http://fake:8000", num_questions=3, batch_size=3)

    results = await pipeline_module.run_job(
        pdf_path=str(tmp_path / "fake.pdf"), topic="on-topic subject", config=config
    )

    assert len(results) == 1
    assert results[0].question == "Q0 on-topic"
    assert results[0].faithfulness == 1.0
