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
from collections import defaultdict
from dataclasses import dataclass

from quiz_ingest.config import PipelineConfig
from quiz_ingest.eval.distractor_eval import score_distractor_diversity
from quiz_ingest.eval.ragas_eval import score_faithfulness_and_relevance
from quiz_ingest.generation.batch_quiz_gen import (
    assemble_quiz_items,
    build_shared_prefix,
    generate_distractor_batch,
    generate_question_batch,
    repair_duplicate_distractors,
)
from quiz_ingest.generation.topic_filter import detect_off_topic_indices
from quiz_ingest.ingest.chunking import chunk_text
from quiz_ingest.ingest.pdf_source import extract_pdf_text
from quiz_ingest.llm.base import CallTelemetry, LLMBackend
from quiz_ingest.llm.vllm_client import VLLMClient, VLLMClientConfig
from quiz_ingest.logging_setup import (
    JobTimer,
    log_api_call,
    log_job_summary,
    log_off_topic_scores,
    log_parse_failure,
)
from quiz_ingest.output_writer import build_job_output, write_job_output_json
from quiz_ingest.rag.index import RagIndex


@dataclass
class ScoredQuizItem:
    question: str
    correct_answer: str
    distractors: list[dict]
    supporting_fact: str
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


async def run_job_streaming(
    *, pdf_path: str, topic: str, config: PipelineConfig, output_json_path: str | None = None
):
    """
    Async generator: yields a list[ScoredQuizItem] per completed batch
    group (progressive delivery), as soon as that group's eval scores are
    ready -- does not wait for all config.num_questions before yielding
    the first group.

    If output_json_path is given, writes ONE consolidated JSON summary
    there when the job finishes successfully (see output_writer.py) --
    items + usage + latency (including real per-stage token/s), built
    from the exact same CallTelemetry objects also streamed to
    logs/*.jsonl via log_api_call.
    """
    gen_backend, embed_backend = _build_backends(config)
    timer = JobTimer()
    job_start = time.monotonic()
    delivered = 0
    status = "failed"  # overwritten to "completed" only if the whole loop finishes
    all_scores: dict[str, list[float]] = {"faithfulness": [], "answer_relevance": [], "diversity": []}
    telemetries_by_stage: dict[str, list[CallTelemetry]] = defaultdict(list)
    all_delivered_items: list[ScoredQuizItem] = []
    all_asked_questions: list[str] = []  # accumulated across groups, see batch_quiz_gen.build_avoid_repeat_addendum

    def _log(t: CallTelemetry, *, stage: str, extra: dict | None = None) -> None:
        log_api_call(t, stage=stage, extra=extra)
        telemetries_by_stage[stage].append(t)

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
            questions, q_telemetry, q_failures, q_raw = await generate_question_batch(
                gen_backend, shared_prefix=shared_prefix, batch_size=group_size,
                already_asked_questions=all_asked_questions,
            )
            # Logged explicitly so a run can be checked after the fact:
            # was the avoid-repeat list actually sent for this call, and
            # how big was it? Without this, a duplicate question in the
            # output is indistinguishable between "the addendum wasn't
            # sent" (stale deploy, wiring bug) and "the model ignored a
            # correctly-sent addendum" (weaker compliance) -- very
            # different problems requiring different fixes.
            _log(
                q_telemetry, stage="generate_questions",
                extra={"avoid_list_size": len(all_asked_questions)},
            )
            if q_failures:
                log_parse_failure(
                    stage="generate_questions", raw_text=q_raw,
                    failed_indices=q_failures, total_count=group_size,
                )
            timer.end_stage()

            # Off-topic check moved here (right after generate_questions,
            # before generate_distractors) rather than after full
            # assembly: two real benefits confirmed from an actual run.
            # (1) A distractor-generation call is never wasted on a
            # question about to be dropped. (2) This check used to be
            # skippable entirely -- if generate_distractors' JSON got
            # truncated (see that function's max_tokens docstring for a
            # real incident), the whole group was dropped BEFORE ever
            # reaching the off-topic filter, so an off-topic item and a
            # truncation failure could mask each other in the logs.
            # Operates on the raw `questions` dicts (not yet assembled
            # into QuizItem, since distractors don't exist yet).
            timer.start_stage("detect_off_topic")
            # valid_questions: parsed-OK items only (q_failures are
            # positions in the ORIGINAL `questions` list -- this is the
            # only place that positional indexing is used; everything
            # downstream keys off each dict's own "index" field instead,
            # to avoid exactly the kind of position-vs-filtered-list bug
            # this comment is here to warn about).
            valid_questions = [q for i, q in enumerate(questions) if i not in q_failures]
            off_topic_indices, off_topic_similarities = await detect_off_topic_indices(
                embed_backend, items=valid_questions, context_chunks=context_chunks,
                threshold=config.off_topic_similarity_threshold,
            )
            log_off_topic_scores(
                similarities=off_topic_similarities, threshold=config.off_topic_similarity_threshold
            )
            if off_topic_indices:
                log_parse_failure(
                    stage="detect_off_topic",
                    raw_text=(
                        f"[dropped {len(off_topic_indices)} question(s) with low similarity "
                        f"to any retrieved context chunk] "
                        + "; ".join(
                            f"index={q.get('index')} question={q.get('question', '')!r}"
                            for q in valid_questions if q.get("index") in off_topic_indices
                        )
                    )[:4000],
                    failed_indices=sorted(off_topic_indices),
                    total_count=len(valid_questions),
                )
            # surviving_questions: parsed-OK AND on-topic. Everything from
            # here on uses THIS list and its dicts' own "index" fields --
            # never q_failures/d_failures as positions into a list that's
            # been filtered since they were computed.
            surviving_questions = [q for q in valid_questions if q.get("index") not in off_topic_indices]
            timer.end_stage()
            if not surviving_questions:
                remaining -= group_size
                continue

            timer.start_stage("generate_distractors")
            distractors, d_telemetry, d_failures, d_raw = await generate_distractor_batch(
                gen_backend, shared_prefix=shared_prefix, questions=surviving_questions
            )
            _log(d_telemetry, stage="generate_distractors")
            if d_failures:
                log_parse_failure(
                    stage="generate_distractors", raw_text=d_raw,
                    failed_indices=d_failures, total_count=len(surviving_questions),
                )
            timer.end_stage()

            # d_failures are positions in surviving_questions (exactly
            # what was sent to generate_distractor_batch above) -- safe
            # to index with here since surviving_questions hasn't changed
            # since that call.
            dropped = {
                surviving_questions[i].get("index") for i in d_failures if i < len(surviving_questions)
            }
            items = assemble_quiz_items(surviving_questions, distractors, dropped, context_chunks)
            if not items:
                log_parse_failure(
                    stage="assemble_quiz_items",
                    raw_text=(
                        f"[no items survived assembly for this group of {group_size}] "
                        f"generate_questions raw: {q_raw[:1500]!r} ||| "
                        f"generate_distractors raw: {d_raw[:1500]!r}"
                    ),
                    failed_indices=list(range(group_size)),
                    total_count=group_size,
                )
                remaining -= group_size
                continue

            timer.start_stage("repair_distractors")
            items, repair_telemetries = await repair_duplicate_distractors(
                gen_backend, shared_prefix=shared_prefix, items=items
            )
            for t in repair_telemetries:
                _log(t, stage="repair_distractors")
            timer.end_stage()
            if not items:
                remaining -= group_size
                continue

            # Record this group's questions so the NEXT group's
            # generate_question_batch call can be told to avoid repeating
            # them -- must happen before eval (which may exclude some
            # items) since the model already "spent" these facts on this
            # group regardless of whether eval later drops one.
            all_asked_questions.extend(it.question for it in items)

            timer.start_stage("eval_faithfulness_relevance")
            ragas_scores = await score_faithfulness_and_relevance(
                gen_backend, embed_backend, shared_prefix=shared_prefix, items=items
            )
            for t in ragas_scores.telemetries:
                _log(t, stage="eval_faithfulness_relevance")
            timer.end_stage()

            timer.start_stage("eval_diversity")
            diversity_scores, diversity_telemetry = await score_distractor_diversity(embed_backend, items)
            if diversity_telemetry is not None:
                _log(diversity_telemetry, stage="eval_diversity")
            timer.end_stage()

            group_results = []
            for it in items:
                if it.index in ragas_scores.excluded_indices:
                    continue  # parse failure -- excluded, not scored 0.0
                scored = ScoredQuizItem(
                    question=it.question,
                    correct_answer=it.correct_answer,
                    distractors=it.distractors,
                    supporting_fact=it.supporting_fact,
                    faithfulness=ragas_scores.faithfulness.get(it.index),
                    answer_relevance=ragas_scores.answer_relevance.get(it.index),
                    diversity=diversity_scores.get(it.index),
                    source_chunk_ids=it.source_chunk_ids,
                )
                group_results.append(scored)
                for key, val in (
                    ("faithfulness", ragas_scores.faithfulness.get(it.index)),
                    ("answer_relevance", ragas_scores.answer_relevance.get(it.index)),
                    ("diversity", diversity_scores.get(it.index)),
                ):
                    if val is not None:
                        all_scores[key].append(val)

            delivered += len(group_results)
            all_delivered_items.extend(group_results)
            remaining -= group_size
            yield group_results

        status = "completed"
    finally:
        wall_time_s = time.monotonic() - job_start
        mean_scores = {
            k: (sum(v) / len(v) if v else 0.0) for k, v in all_scores.items()
        }
        log_job_summary(
            status=status,
            num_delivered=delivered,
            num_requested=config.num_questions,
            wall_time_s=wall_time_s,
            stage_timings_s=timer.stage_timings_s,
            mean_scores=mean_scores,
        )
        if output_json_path is not None and status == "completed":
            output = build_job_output(
                source_mode="pdf",
                source=pdf_path,
                scored_items=all_delivered_items,
                wall_time_s=wall_time_s,
                stage_timings_s=timer.stage_timings_s,
                telemetries_by_stage=telemetries_by_stage,
            )
            write_job_output_json(output, output_json_path)
