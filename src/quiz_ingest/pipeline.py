"""
pipeline.py

run_job() is the single entry point: PDF path + topic hint + num_questions
in, list of scored QuizItem out. Progressive delivery (yielding a group as
soon as it's scored, instead of waiting for all num_questions) is done via
an async generator -- see run_job_streaming below -- so a CLI or future
API layer can print/display results as they arrive, matching the earlier
project's UX.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from quiz_ingest.config import PipelineConfig
from quiz_ingest.eval.distractor_eval import score_distractor_diversity
from quiz_ingest.eval.ragas_eval import score_faithfulness_and_relevance
from quiz_ingest.generation.batch_quiz_gen import (
    assemble_quiz_items,
    build_shared_prefix,
    generate_distractor_batch,
    generate_question_batch,
)
from quiz_ingest.ingest.chunking import chunk_text
from quiz_ingest.ingest.pdf_source import extract_pdf_text
from quiz_ingest.llm.base import LLMBackend
from quiz_ingest.llm.vllm_client import VLLMClient, VLLMClientConfig
from quiz_ingest.logging_setup import JobTimer, log_api_call, log_job_summary
from quiz_ingest.rag.index import RagIndex


@dataclass
class ScoredQuizItem:
    question: str
    correct_answer: str
    distractors: list[dict]
    faithfulness: float | None
    answer_relevance: float | None
    diversity: float | None
    source_chunk_ids: list[int]


def _build_backends(config: PipelineConfig) -> tuple[LLMBackend, LLMBackend]:
    """Returns (generation_backend, embed_backend). May be the same object
    if a single instance serves both roles."""
    if config.backend == "serverless":
        # Imported lazily -- see llm/vast_serverless_client.py docstring
        # for why this path is not exercised by default.
        from quiz_ingest.llm.vast_serverless_client import (
            VastServerlessClient,
            VastServerlessConfig,
        )

        gen = VastServerlessClient(
            VastServerlessConfig(endpoint_name=config.vast_endpoint_name, model=config.model)
        )
        embed = VastServerlessClient(
            VastServerlessConfig(
                endpoint_name=config.vast_embed_endpoint_name or config.vast_endpoint_name,
                model=config.embed_model,
                embed_model=config.embed_model,
            )
        )
        return gen, embed

    gen = VLLMClient(VLLMClientConfig(base_url=config.vllm_base_url, model=config.model))
    embed_url = config.embed_base_url or config.vllm_base_url
    embed = VLLMClient(
        VLLMClientConfig(
            base_url=embed_url, model=config.model, embed_model=config.embed_model
        )
    )
    return gen, embed


async def run_job(
    *, pdf_path: str, topic: str, config: PipelineConfig
) -> list[ScoredQuizItem]:
    results = []
    async for group in run_job_streaming(pdf_path=pdf_path, topic=topic, config=config):
        results.extend(group)
    return results


async def run_job_streaming(*, pdf_path: str, topic: str, config: PipelineConfig):
    """
    Async generator: yields a list[ScoredQuizItem] per completed batch
    group (progressive delivery), as soon as that group's eval scores are
    ready -- does not wait for all config.num_questions before yielding
    the first group.
    """
    gen_backend, embed_backend = _build_backends(config)
    timer = JobTimer()
    job_start = time.monotonic()
    delivered = 0
    status = "failed"  # overwritten to "completed" only if the whole loop finishes
    all_scores: dict[str, list[float]] = {"faithfulness": [], "answer_relevance": [], "diversity": []}

    try:
        timer.start_stage("ingest_and_chunk")
        raw_text = extract_pdf_text(pdf_path)
        chunks = chunk_text(raw_text, source=pdf_path, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        timer.end_stage()

        timer.start_stage("retrieve_rerank")
        rag = RagIndex(embed_backend)
        context_chunks = await rag.build_and_retrieve(chunks, topic, top_k=config.retrieve_top_k)
        timer.end_stage()

        shared_prefix = build_shared_prefix(topic, context_chunks)

        remaining = config.num_questions
        while remaining > 0:
            group_size = min(config.batch_size, remaining)

            timer.start_stage("generate_questions")
            questions, q_telemetry, q_failures = await generate_question_batch(
                gen_backend, shared_prefix=shared_prefix, batch_size=group_size
            )
            log_api_call(q_telemetry, stage="generate_questions")
            timer.end_stage()

            timer.start_stage("generate_distractors")
            distractors, d_telemetry, d_failures = await generate_distractor_batch(
                gen_backend, shared_prefix=shared_prefix, questions=questions
            )
            log_api_call(d_telemetry, stage="generate_distractors")
            timer.end_stage()

            dropped = {questions[i].get("index") for i in q_failures} | {
                questions[i].get("index") for i in d_failures if i < len(questions)
            }
            items = assemble_quiz_items(questions, distractors, dropped, context_chunks)
            if not items:
                remaining -= group_size
                continue

            timer.start_stage("eval_faithfulness_relevance")
            ragas_scores = await score_faithfulness_and_relevance(
                gen_backend, embed_backend, shared_prefix=shared_prefix, items=items
            )
            for t in ragas_scores.telemetries:
                log_api_call(t, stage="eval_faithfulness_relevance")
            timer.end_stage()

            timer.start_stage("eval_diversity")
            diversity_scores, diversity_telemetry = await score_distractor_diversity(embed_backend, items)
            if diversity_telemetry is not None:
                log_api_call(diversity_telemetry, stage="eval_diversity")
            timer.end_stage()

            group_results = []
            for it in items:
                if it.index in ragas_scores.excluded_indices:
                    continue  # parse failure -- excluded, not scored 0.0
                group_results.append(
                    ScoredQuizItem(
                        question=it.question,
                        correct_answer=it.correct_answer,
                        distractors=it.distractors,
                        faithfulness=ragas_scores.faithfulness.get(it.index),
                        answer_relevance=ragas_scores.answer_relevance.get(it.index),
                        diversity=diversity_scores.get(it.index),
                        source_chunk_ids=it.source_chunk_ids,
                    )
                )
                for key, val in (
                    ("faithfulness", ragas_scores.faithfulness.get(it.index)),
                    ("answer_relevance", ragas_scores.answer_relevance.get(it.index)),
                    ("diversity", diversity_scores.get(it.index)),
                ):
                    if val is not None:
                        all_scores[key].append(val)

            delivered += len(group_results)
            remaining -= group_size
            yield group_results

        status = "completed"
    finally:
        mean_scores = {
            k: (sum(v) / len(v) if v else 0.0) for k, v in all_scores.items()
        }
        log_job_summary(
            status=status,
            num_delivered=delivered,
            num_requested=config.num_questions,
            wall_time_s=time.monotonic() - job_start,
            stage_timings_s=timer.stage_timings_s,
            mean_scores=mean_scores,
        )
