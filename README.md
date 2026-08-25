# quiz-ingest-vllm

Document (PDF) → 4-option multiple-choice quiz, scored with 3
reference-free metrics (RAGAS Faithfulness, RAGAS Answer Relevance,
Distractor Diversity), running entirely on a **self-hosted vLLM GPU
instance** rented from Vast.ai. No Gemini/GCP dependency anywhere in this
repo.

This is a narrower sibling of an earlier project (`quizrag-scale`) --
this repo drops the web-research/topic-ingestion path, the Gemini
backend, the credits/rate-limiting layer, and the admission-gate/circuit-
breaker concurrency machinery (all of that existed to survive a *shared,
rate-limited third-party API key*; a rented GPU instance you fully
control doesn't have that problem). What's kept and expanded: batched
generation, the 3 eval metrics, and — new in this repo — **real
per-stage TTFT/decode-tokens-per-second telemetry**, because a
self-hosted vLLM server can report it directly instead of approximating.

## Status

Built for **instance mode** (a single rented GPU, vLLM's own HTTP
server, called directly) — this is the mode to validate first, per the
staged plan: rent 1 instance → deploy vLLM → confirm quality/speed →
only then move to a Vast Serverless Endpoint.

`llm/vast_serverless_client.py` exists and satisfies the same `LLMBackend`
Protocol, but is **not wired into `scripts/` by default** and has **not
been run against a real Vast Serverless Endpoint yet** — see that file's
docstring for exactly what to confirm before flipping `--backend
serverless` on for real (in particular: whether the current `vastai`
package's `Serverless.request()` streams or not, which determines
whether TTFT is measurable the same way it is in instance mode).

## Architecture

```
PDF path + topic hint
        │
        ▼
1. extract_pdf_text()      -- cap 60MB raw / 200K extracted chars
        │
        ▼
2. chunk_text()             -- char-window chunks, configurable overlap
        │
        ▼
3. RagIndex.build_and_retrieve()
   - BM25 prefilter (local, free) → top MAX_CHUNKS_TO_EMBED candidates
   - ONE embed() call for query + all candidates
   - cosine rerank → top_k chunks
        │
        ▼
4. build_shared_prefix()    -- system instruction + retrieved context,
                                built ONCE, reused byte-identical across
                                every call below (this is what prefix
                                caching keys on)
        │
        ▼
5. Loop over groups of `batch_size` questions:
   a. generate_question_batch()      -- 1 call/group
   b. generate_distractor_batch()    -- 1 call/group
   c. score_faithfulness_and_relevance()  -- 3 calls/group + 1 embed call
   d. score_distractor_diversity()   -- 1 embed call/group
   e. yield the scored group immediately (progressive delivery)
        │
        ▼
6. Every LLM call logs an `api_call` JSONL event with REAL
   ttft_s/decode_tokens_per_second (streamed, not approximated -- see
   llm/vllm_client.py). Every job logs a `job_summary` event with
   stage_timings_s broken out per stage.
```

## Running 2 vLLM processes on 1 GPU (instance mode, first test)

Generation and embedding are separate vLLM processes even on the same
box — start both, on different ports, splitting `--gpu-memory-utilization`
between them so there's still headroom left over for the prefix-cache
KV pool (see "Prefix caching" below for why that headroom matters):

```bash
# generation, port 8000
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --port 8000 \
    --gpu-memory-utilization 0.55 \
    --max-model-len 16384 \
    --enable-prefix-caching

# embedding, port 8001 (separate process -- vLLM does not serve
# generate + embed from one process)
# NOTE: `--task embed` is deprecated as of recent vLLM releases -- use
# `--runner pooling` instead (vLLM usually auto-detects this correctly
# for a native embedding model like Qwen3-Embedding-*, but pass it
# explicitly to not depend on auto-detection).
vllm serve Qwen/Qwen3-Embedding-0.6B \
    --runner pooling \
    --port 8001 \
    --gpu-memory-utilization 0.25
```

Then:

```bash
pip install -e .
python scripts/generate_quiz_from_pdf.py \
    --pdf path/to/document.pdf \
    --topic "history of rome" \
    --num-questions 5 \
    --vllm-base-url http://localhost:8000 \
    --embed-base-url http://localhost:8001
```

(Same commands work against a remote rented instance — swap `localhost`
for the instance's external IP.)

## Prefix caching

`--enable-prefix-caching` is a **server-side** vLLM flag — nothing in
this repo's client code turns it on or off, but the client code IS
responsible for making sure the thing being cached is actually stable:

- `build_shared_prefix()` is called exactly once per job/group and the
  resulting string is passed unchanged into every one of the 4 LLM calls
  in that group (question gen, distractor gen, 2 eval steps). If you
  modify `generate_json_batch`'s prompt-building, keep `shared_prefix`
  and the per-item suffix as separate arguments all the way down — do
  not let a caller pre-concatenate them, or small formatting drift
  between calls will silently break cache hits.
- The system instruction (`NON_CONVERSATIONAL_SYSTEM_INSTRUCTION` in
  `generation/schemas.py`) is identical across every job on the process,
  so it stays cached across the whole server's lifetime, not just within
  one job.
- `--kv-cache-dtype fp8` is worth benchmarking (not enabled by default
  here) — halves KV cache memory, which grows how much prefix can stay
  cached before eviction. Verify quality doesn't regress before turning
  it on for real (see "Benchmarking" below).

## Real token/s telemetry

`llm/vllm_client.py::_stream_completion` streams every completion call
with `stream_options: {"include_usage": true}` and measures:

- `ttft_s` — wall-clock from request sent to first non-empty text chunk
- `decode_tokens_per_second` — `(completion_tokens - 1) / (t_last - t_first)`,
  matching how vLLM's own Prometheus histogram defines
  `time_per_output_token`
- `prefix_cache_hit_tokens` — read from the server's usage extension when
  present, to directly confirm prefix caching is actually hitting (not
  just assumed from lower latency)

Every call's telemetry is written to `logs/quiz-ingest-events-*.jsonl` as
an `api_call` event, tagged with `stage` (`generate_questions`,
`generate_distractors`, `eval_faithfulness_relevance`, `eval_diversity`).

## Benchmarking (batch size / model / quantization)

```bash
python scripts/bench_stage_timing.py \
    --pdf path/to/document.pdf --topic "history of rome" \
    --vllm-base-url http://localhost:8000 --embed-base-url http://localhost:8001 \
    --batch-sizes 4 8 16 32 --num-questions 16 --repeats 2 \
    --model Qwen/Qwen3-4B-Instruct-2507
```

Prints a table of `stage_timings_s` + mean 3-metric scores per
configuration, read back from `logs/` (not recomputed independently, so
the CLI's numbers and the benchmark table's numbers can never disagree).
To compare `--enable-prefix-caching` on vs off, or one model vs another,
run this script once per vLLM server config and diff the two tables —
prefix caching itself can't be toggled by this script since it requires
restarting the vLLM process with a different flag.

## Eval score integrity

If a batch JSON response only partially parses, the unparsed items are
**excluded** from that group's scores (`EvalScores.excluded_indices`),
never defaulted to `0.0`. This distinction mattered in the earlier
project: several models scored an implausibly uniform `0.000` on
Answer Relevance simultaneously, which turned out to be parse failures
being silently counted as real (terrible) scores, not genuinely bad
generations. `assemble_quiz_items()` and `score_faithfulness_and_relevance()`
both propagate failure indices explicitly instead of letting them look
like real zeros downstream.

**Faithfulness checks the real retrieved context, not the model's own
self-report.** The verify step compares decomposed statements against
`shared_prefix` (the actual RAG context sent to the generation call), not
against a per-item `supporting_fact` field the model wrote in the same
call as `correct_answer`. Checking a claim against the model's own
self-report is a self-referential loop with zero grounding guarantee --
observed directly in a real run: a negation-style question ("which is
NOT a feature...") whose `correct_answer` was objectively false (it
asserted the Transformer uses recurrent attention -- the exact opposite
of the paper's central claim) still scored `faithfulness=1.0`, because
the model's self-written `supporting_fact` agreed with its own false
claim. This is a real limitation of using the same model as both
generator and judge -- pointing the check at the real context closes the
most egregious failure mode (self-consistency masquerading as grounding)
but doesn't guarantee a small judge model never misjudges a genuinely
subtle case.

Separately, a distractor that's a literal copy of the correct answer
(observed in real runs against `Qwen3-4B-Instruct-2507` -- see
`generation/batch_quiz_gen.py`) is caught and **repaired**, not just
dropped: `repair_duplicate_distractors()` regenerates distractors for
only the affected items in one smaller batched call, and only drops an
item if it's still duplicated after the retry budget is exhausted. None
of the 3 eval metrics catch this on their own -- Faithfulness/Answer
Relevance never look at the distractors, and Diversity only compares
distractors to each other, never to the correct answer.

## Diagnosing a batch that silently produces fewer items than requested

Every `generate_json_batch` call whose response has ANY parse failures --
including a full-batch failure that used to leave zero trace -- now logs
a `parse_failure` event (`logging_setup.py::log_parse_failure`) with the
stage, how many of the batch failed, and a `raw_text_excerpt` of what the
model actually returned:

```bash
grep '"event": "parse_failure"' logs/quiz-ingest-events-*.jsonl | python3 -m json.tool
```

If a whole group of items disappears (e.g. requesting 10 delivers fewer,
with a gap in the middle), check this first -- it was previously
indistinguishable from "the model just decided not to answer," with no
diagnostic path at all.

## Cross-group question dedup

`num_questions` gets split into groups of at most `batch_size` (see
`generation/batch_quiz_gen.py::QUESTION_BATCH_SIZE`) -- each group is a
separate, stateless LLM call. Against the same context, two separate
calls can independently pick the same most-salient fact and produce
near-duplicate questions (observed directly: with `num_questions=10`,
`batch_size=8`, question 1 and question 9 -- from two different groups --
ended up asking the exact same thing). None of the 3 eval metrics catch
this, and vLLM's prefix cache doesn't help either -- it only reuses
matching *input* tokens between calls, it has no memory of a previous
call's *output*.

Fix: `generate_question_batch` accepts `already_asked_questions`, and
`pipeline.py` accumulates every group's questions across the whole job
and passes them into the next group's call. The addendum is appended
AFTER the base `shared_prefix` (see
`batch_quiz_gen.py::build_avoid_repeat_addendum`), not interleaved into
it, so the cache-relevant base portion (system instruction + retrieved
context) stays byte-identical across every call in every group -- only
this small tail differs, and only for groups after the first.

This doesn't guarantee zero repeats (it's still a prompted instruction,
not a hard constraint) -- there's no post-hoc dedup safety net yet (the
distractor-duplication problem has one via `repair_duplicate_distractors`;
question-level duplication doesn't). Worth adding the same repair pattern
here if avoid-list prompting alone isn't enough in practice.

## Consolidated per-job JSON output

Beyond the JSONL event log, `run_job_streaming(..., output_json_path=...)`
writes one consolidated JSON per job on completion (default path:
`outputs/quiz_output_<pdf-stem>_<timestamp>.json`, printed by
`generate_quiz_from_pdf.py` at the end of a run). Schema:

```json
{
  "source_mode": "pdf",
  "source": "path/to/document.pdf",
  "items": [
    {
      "question": "...",
      "options": [{"text": "...", "is_correct": false, "type": "near_miss"}, ...],
      "num_correct": 1,
      "supporting_fact": "...",
      "faithfulness": 0.95,
      "answer_relevance": 0.74,
      "diversity": 0.35,
      "source_chunk_ids": [3, 7, 12]
    }
  ],
  "usage": {
    "total_calls": 6,
    "total_prompt_tokens": 18432,
    "total_completion_tokens": 3120,
    "avg_tokens_per_question": 4310.4
  },
  "latency": {
    "wall_time_s": 23.76,
    "stage_timings_s": {"generate_questions": 6.28, "eval_faithfulness_relevance": 8.35, "...": "..."},
    "stage_token_throughput": {
      "generate_questions": {
        "call_count": 1, "total_prompt_tokens": 3091, "total_completion_tokens": 535,
        "mean_ttft_s": 0.27, "mean_decode_tokens_per_second": 98.9
      },
      "...": "..."
    }
  }
}
```

Two deliberate differences from an earlier project's per-job JSON this
schema is adapted from:
- No `estimated_cost_usd` -- self-hosted GPU has no per-token API cost;
  the real cost is $/hr instance rental over uptime, not a per-job
  number, so it isn't fabricated here.
- `stage_token_throughput` (mean TTFT + mean decode tokens/s **per
  stage**) is new -- the earlier project's vLLM backend could only
  report an approximate, non-streaming total latency. Every number here
  comes from the same `CallTelemetry` objects also written to
  `logs/*.jsonl`, so the two outputs can never disagree.

## Tests

```bash
pytest -q
```

11 tests, no GPU/network needed — chunking, the tolerant JSON-array
parser (clean / markdown-fenced / partial-failure / total-failure
cases), item-assembly drop logic, and eval-score exclusion, using a fake
`LLMBackend` so the Protocol boundary itself is what's under test.

## Not done yet

- `llm/vast_serverless_client.py` — not run against a real Endpoint.
  Streaming support in the current `vastai` SDK needs confirming before
  TTFT can be trusted in serverless mode (see that file's docstring).
- No `--kv-cache-dtype fp8` on/off comparison run yet.
- `QUESTION_BATCH_SIZE=8` (in `generation/batch_quiz_gen.py`) is carried
  over as a starting point, not re-derived for this repo's model/GPU —
  use `bench_stage_timing.py` before trusting it.
- No chunk_size/retrieve_top_k sweep yet — Faithfulness depends on
  retrieval quality as much as on the generation model; only the
  generation side has been benchmarked so far.

## Demo API (Swagger UI) -- sharing a public link

```bash
pip install -e ".[api]"

# instance mode
export QUIZ_BACKEND=instance
export VLLM_BASE_URL=http://localhost:8000
export EMBED_BASE_URL=http://localhost:8001
# OR serverless mode
export QUIZ_BACKEND=serverless
export VAST_API_KEY=...
export VAST_ENDPOINT_NAME=quiz-gen-qwen3-4b
export VAST_EMBED_ENDPOINT_NAME=quiz-embed-qwen3

# strongly recommended before exposing this on a public IP -- see
# api.py's _require_api_key docstring for why
export DEMO_API_KEY=<a secret you make up>

python scripts/demo_server.py --host 0.0.0.0 --port 8080
```

Then share `http://<this-machine's-ip>:8080/docs` -- a Swagger UI with
one `POST /generate-quiz` endpoint (upload a PDF, set topic/num_questions,
"Try it out") and `GET /health`. The response body is the exact same
consolidated JSON `generate_quiz_from_pdf.py` writes to disk (see
"Consolidated per-job JSON output" above) -- read back from that file,
not rebuilt separately, so the two can't drift into different schemas.

The backend (instance vs serverless, model names) is configured **once,
server-side**, via the environment variables above -- never something the
person hitting the API supplies. `num_questions` is still capped at
`MAX_NUM_QUESTIONS` (10) and validated by FastAPI itself (a request over
the cap gets a `422`, not silently clamped).

**This is a demo server, not a hardened production API**: no request
queueing (concurrent requests all hit the backend directly -- fine for
one person trying it, not for real traffic), no rate limiting, `DEMO_API_KEY`
is a single shared secret (not per-user), and no HTTPS termination (put
this behind a reverse proxy -- e.g. Caddy or nginx -- for anything beyond
a quick demo link). `tests/test_api.py` covers the routing/validation
logic with a mocked pipeline -- it does not exercise a real backend.

## Load testing (concurrent users)

```bash
python scripts/load_test.py \
    --pdf test_docs/sample.pdf \
    --vllm-base-url http://localhost:8000 --embed-base-url http://localhost:8001 \
    --concurrency-levels 1 2 4 8 16 32 \
    --requests-per-worker 5
```

**Methodology** (fixed after a real run exposed a measurement artifact
in an earlier version): total requests SCALE with concurrency
(`concurrency * requests_per_worker`), not a fixed pool shared across
every level. `concurrency` worker coroutines run in parallel, each
firing its own requests sequentially -- so exactly `concurrency`
requests are genuinely in flight at any instant (true steady state). An
earlier version fired a fixed total at every level through one shared
queue; at low concurrency most of that fixed pool spent most of its
measured time waiting in line, not being processed, which made latency
look like it improved dramatically as concurrency increased -- it
didn't, the queue just got shorter. Verified in
`tests/test_load_test.py` (`test_run_level_total_requests_scales_with_concurrency`).

**`delivery_rate`, not just `success_rate`.** A job that completes
without raising an exception can still deliver ZERO of the questions it
was asked for (parse failures, off-topic content dropped, distractors
that couldn't be repaired). A real run surfaced this directly: every
job reported `success_rate=1.0` while 71% of jobs delivered nothing --
`delivery_rate` (`delivered_total / requested_total`) is now a primary,
always-printed column and part of the auto-stop degradation check
(`--degradation-delivery-rate`, default 0.5), not an afterthought.
Regression-tested in `test_summarize_level_delivery_rate_can_be_zero_despite_perfect_success_rate`.

Each simulated request still gets a randomized topic (varies retrieved
context size) and randomized `num_questions`/`retrieve_top_k` (varies
prompt size) -- deliberately not identical repeated requests, which
would give an unrealistically rosy result via prefix-cache reuse that
real, distinct traffic wouldn't get.

Per level, reports success rate, delivery rate, p50/p95/max latency
(true per-request service time now, not queue-wait-inflated), and mean
decode tokens/s (read back from `logs/*.jsonl`'s real per-call
telemetry). Writes a per-level JSON summary to
`logs/load_test_summary_<ts>.json`. If `delivery_rate` looks low at any
level, grep `logs/*.jsonl` for `"event": "parse_failure"` -- every entry
includes the raw model output that failed, not just a pass/fail flag.

## Off-topic / hallucination detection

A load test with randomized topics and low `retrieve_top_k` surfaced a
real failure mode no eval metric was catching: a request grounded in
the sample PDF (topic "training efficiency of the transformer")
produced a fully-formed, validly-structured question about
*mitochondria*. Faithfulness and Answer Relevance both operate on an
item in isolation and don't catch this -- a hallucinated item can be
perfectly self-consistent.

`generation/topic_filter.py::detect_off_topic_indices` catches this with
one batched embed() call per group: each candidate question's
question+correct_answer is compared (max cosine similarity) against
every retrieved context chunk, and anything below
`PipelineConfig.off_topic_similarity_threshold` (default 0.3, loosely
calibrated -- tune against your own runs) is dropped.

**Runs right after `generate_questions`, before `generate_distractors`**
(moved there after a real run showed why placement matters): if this
check only ran after the full item was assembled, a downstream JSON
truncation failure (see "Token budget" below) could drop the whole group
before the off-topic filter ever got a chance to run, silently masking
which of the two problems actually caused a given failure. Running it
earlier also means a distractor-generation call is never wasted on a
question that's about to be dropped anyway. Locked in end to end with a
scripted-backend integration test
(`tests/test_pipeline_integration.py`) that deliberately combines a
parse failure AND an off-topic item in the same batch -- this also
caught a real indexing bug introduced during the move (stale positional
indices from before the off-topic filter being reused to index into the
list *after* filtering), now fixed and regression-tested.

## Token budget -- a confirmed real incident

A load test found `generate_distractors` calls whose `completion_tokens`
landed on EXACTLY `150 * num_questions` for several failed batches (7,
8, and 9 questions) -- not a coincidence: the model was being cut off
mid-JSON by `max_tokens` before the array could close, so the entire
batch failed to parse (100% loss, not partial) purely from budget being
too tight for 3 distractor texts + JSON structural overhead per item.
Bumped to `300 * num_questions` for distractors and `260 * num_questions`
for questions (which wasn't failing outright but was running close,
~78% utilization observed in the same run).

`CallTelemetry.hit_token_limit` (`completion_tokens >= max_tokens sent`)
is now computed automatically for every call in both `VLLMClient` and
`VastServerlessClient`, and shows up in every `api_call` log line --
diagnosing this kind of failure no longer requires manually
cross-referencing completion_tokens against each call site's budget
formula by hand.


## Closing the diagnostic gap: eval-stage and repair-exhausted failures now logged

A load test's `delivered/requested` gap (34 items missing) vastly
exceeded its `parse_failure` event count (1) -- most of the loss was
happening in two code paths that detected failures internally but never
logged them:

- `eval/ragas_eval.py`'s 3 internal calls (decompose, verify,
  reverse-question) tracked parse failures via `excluded_indices` but
  never called `log_parse_failure` -- an item silently excluded from
  scoring during eval left no trace of *why*. Now logs
  `stage: "eval_decompose"` / `"eval_verify"` / `"eval_reverse_question"`
  with the raw model output, same as every other stage.
- `repair_duplicate_distractors` silently dropped items still duplicated
  after exhausting `max_retries` -- now logs
  `stage: "repair_distractors_exhausted"` with the dropped questions.

If `delivered_total` is still well below `requested_total` after
upgrading, `grep '"event": "parse_failure"' logs/*.jsonl | python3 -c
"import json,sys,collections; print(collections.Counter(json.loads(l)['stage'] for l in sys.stdin))"`
now gives a complete stage-by-stage breakdown instead of missing most of
the loss.

## Benchmarking multiple models (scripts/benchmark_models.py)

```bash
python scripts/benchmark_models.py \
    --pdf test_docs/sample.pdf \
    --jobs-per-pair 8 --num-questions 5
```

Manages the vLLM server lifecycle itself: starts a generation + embedding
server pair, waits for both `/health` checks, runs `jobs_per_pair` real
end-to-end jobs against the actual pipeline (retrieval, batched
generation, eval, repair, off-topic filtering -- not a synthetic
completion benchmark), tears both servers down, cools down for GPU
memory to release, then moves to the next pair. One combination loaded
at a time.

**Candidates and reasoning** are documented directly in the script
(`GENERATION_CANDIDATES` / `EMBEDDING_CANDIDATES`) -- text-only dense
models only (no multimodal variants, per this repo's earlier Gemma3
lesson), capped around 8-9B to fit a 24GB card alongside a separate
embedding process, spanning both same-family-different-size (Qwen3-4B
vs Qwen3-8B) and different-family-same-size (Qwen3-8B vs GLM-4-9B vs
Granite-4.1-8B vs Llama-3.1-8B) so you can tell whether size or training
lineage matters more for this specific batched-JSON-generation task --
public leaderboards don't answer that.

**Design: one-factor-at-a-time, not a full cross product** -- by
default, every generation candidate is tested against the baseline
embedding model, and every other embedding candidate is tested against
the baseline generation model. Use `--only <id> <id>...` to restrict to
specific candidates (still paired against the relevant baseline, not
combined with each other), or `--skip-generation-sweep` /
`--skip-embedding-sweep` to run just one axis.

`--gpu-memory-utilization` for each server is computed automatically
from the candidate's `vram_gb` estimate (`resolve_gpu_utilization_pair`)
-- a rough starting point, not exact science; watch `nvidia-smi` on a
new candidate's first run and adjust `vram_gb` in the candidate table or
`--total-vram-gb` if a server OOMs.

**Not yet run against real hardware in this environment** -- the
subprocess start/health-check/teardown mechanics (`VLLMServerProcess`)
are implemented defensively and unit-tested with a dummy process
(`tests/test_benchmark_models.py`), but the actual `vllm serve` startup
timing, health-check readiness, and OOM behavior have not been verified
end to end. Run it on your instance and adjust `--startup-timeout-s` /
`--cooldown-s` if needed -- report back what breaks.

