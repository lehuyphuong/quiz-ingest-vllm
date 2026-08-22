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
vllm serve google/gemma-3-4b-it \
    --port 8000 \
    --gpu-memory-utilization 0.55 \
    --max-model-len 32768 \
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
    --model google/gemma-3-4b-it
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
