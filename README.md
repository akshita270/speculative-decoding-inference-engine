# Speculative Decoding Inference Engine

A mini LLM inference system built to practice **production serving concerns**
around a **speculative decoding engine** — not just "a model that works," but
the layers a real inference service needs around it: request queueing,
semantic caching, load-based routing, fallback/reliability, and
observability.

Built as a weekend project while transitioning from Data Analyst into
AI/LLM Engineering. Everything below — architecture, numbers, limitations —
reflects what was actually built and measured on a single Apple M2 laptop,
not theoretical claims.

## What is speculative decoding, and why does it matter here?

Normal autoregressive decoding generates one token per forward pass through
the (large, expensive) model — latency scales linearly with output length.

Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023) speeds this
up without changing the output distribution:

1. A small, cheap **draft model** proposes `k` tokens ahead, one at a time.
2. The large **target model** checks all `k` proposed tokens in a **single
   batched forward pass** — verifying k tokens costs about the same
   wall-clock time as generating one token normally.
3. **Accept/reject**: walk the k draft tokens left to right. Accept token `i`
   with probability `min(1, p_target(x_i) / p_draft(x_i))`. On the first
   rejection, discard every draft token from that point on (they were
   conditioned on a token the target never would have produced) and
   **resample** a replacement from the residual distribution
   `max(0, p_target - p_draft)`, renormalized. If all k are accepted, draw
   one bonus token for free from the target's next-token distribution.

This specific accept/reject rule is what makes the output **mathematically
identical** to sampling from the target model alone — it's not an
approximation, it's an exact algorithm that happens to be faster when the
draft and target models agree often enough. The implementation lives in
[engine/speculative_decode.py](engine/speculative_decode.py), heavily commented
since this is the part I need to be able to explain in interviews.

## Architecture

```
                          ┌────────────┐
                          │   Client   │
                          └─────┬──────┘
                                │ POST /generate
                                ▼
                    ┌────────────────────────┐
                    │  FastAPI (serving/api.py) │
                    └───────────┬─────────────┘
                                │ enqueue
                                ▼
                ┌──────────────────────────────────┐
                │  Request Queue + Batcher          │
                │  (serving/queue_batcher.py)       │
                │  groups requests arriving within  │
                │  ~50ms into a batch, so concurrent│
                │  requests queue instead of drop   │
                └────────────────┬───────────────────┘
                                 │ processed one at a time
                                 ▼
                   ┌───────────────────────────────┐
                   │  Semantic Cache check          │
                   │  (cache/semantic_cache.py)     │
                   │  cosine similarity >= 0.92 ?   │
                   └──────┬─────────────────┬───────┘
                     hit  │                 │  miss
                          ▼                 ▼
                  return cached   ┌──────────────────────────┐
                     response     │  Load-Based Router         │
                                  │  (serving/router.py)       │
                                  │  high load  -> fast draft  │
                                  │  low load   -> better draft│
                                  └─────────────┬───────────────┘
                                                ▼
                                ┌───────────────────────────────┐
                                │  Speculative Decoding Engine    │
                                │  (engine/speculative_decode.py) │
                                │  draft.propose -> k tokens      │
                                │  target.verify -> 1 fwd pass    │
                                │  accept / reject / resample     │
                                └───────────────┬───────────────────┘
                                                │ on exception
                                                ▼
                                ┌───────────────────────────────┐
                                │  Fallback (serving/fallback.py) │
                                │  target-only normal decoding    │
                                └───────────────┬───────────────────┘
                                                ▼
                                store new prompt+response in cache
                                                │
                                                ▼
                          log metrics -> observability/metrics_store.csv
                                                │
                                                ▼
                                  response returned to client

     Streamlit dashboard (dashboard/app.py) independently tails
     metrics_store.csv and auto-refreshes while traffic flows.
```

## Models used

Draft and target models must share the same tokenizer/vocabulary for the
token-level accept/reject math to be valid — all three below are GPT-2
family models (same byte-level BPE vocab, 50257 tokens):

| Role                    | Model         | Params | Why                                             |
|-------------------------|---------------|--------|--------------------------------------------------|
| Draft (fast, low load)  | `gpt2`        | 124M   | Cheapest possible proposer                       |
| Draft (better, default) | `gpt2-medium` | 355M   | Higher acceptance rate, costs more per proposal  |
| Target                  | `gpt2-large`  | 774M   | "Ground truth" model, only ever runs batched verify passes |

## Project structure

```
engine/          draft_model.py, target_model.py, speculative_decode.py (core accept/reject),
                  normal_decode.py (baseline), model_utils.py (device selection)
serving/         api.py (FastAPI), queue_batcher.py, router.py, fallback.py
cache/           semantic_cache.py
observability/   logger.py, metrics_store.csv (generated at runtime)
load_test/       load_test.py, results.json (generated by running the script)
dashboard/       app.py (Streamlit)
```

## How to run it locally

Requires Python 3.9+. GPU is optional but strongly recommended if available
(CUDA or Apple Silicon MPS) — CPU-only will be noticeably slower per the
numbers below.

```bash
# 1. Set up environment (reuses already-installed packages if present)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the API (downloads gpt2, gpt2-medium, gpt2-large on first run, ~5GB total)
python -m uvicorn serving.api:app --host 127.0.0.1 --port 8000

# 3. Send a request
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of artificial intelligence is", "max_new_tokens": 40, "k": 4}'

# 4. Simulate concurrent traffic (in a separate terminal, API must be running)
python load_test/load_test.py --levels 1 10 50 100 --max-new-tokens 20

# 5. Launch the live dashboard (in a separate terminal)
python -m streamlit run dashboard/app.py
```

The dashboard reads `observability/metrics_store.csv` directly, so it updates
as real requests (including from the load test) come in.

> **Gotcha:** always run `python -m uvicorn ...` / `python -m streamlit ...`
> (not the bare `uvicorn`/`streamlit` commands) after activating the venv.
> Because the venv was created with `--system-site-packages`, packages like
> `streamlit` and `uvicorn` are importable but don't get entry-point scripts
> inside `.venv/bin/`. The bare command name then falls through your shell's
> `PATH` to any other `streamlit`/`uvicorn` installed on the system (e.g. a
> Homebrew one) — a *different* Python environment that won't have this
> project's dependencies (like `plotly`) installed, causing a confusing
> `ModuleNotFoundError` even though `pip install -r requirements.txt`
> succeeded. `python -m <module>` avoids this because it always resolves
> against the active venv's `python`.

## Actual measured results (Apple M2, 16GB unified memory, MPS backend)

### Speculative vs. normal decoding

Averaged over 5 prompts, `max_new_tokens=40`, `k=4`, `temperature=0.7`,
`torch.manual_seed(42)`:

| Method                          | Tokens/sec | Draft acceptance rate | Speedup vs. normal |
|----------------------------------|-----------:|-----------------------:|--------------------:|
| Normal decoding (gpt2-large only) | 10.13      | n/a                     | 1.00x                |
| Speculative (draft = gpt2)        | 11.22      | 0.47                    | **1.11x**            |
| Speculative (draft = gpt2-medium) | 10.82      | 0.56                    | 1.07x                |

Interesting finding: the **faster, less accurate draft model produced a
bigger speedup** than the "better" one, despite a lower acceptance rate
(0.47 vs 0.56). `gpt2-medium` proposes higher-quality tokens, but it's ~3x
more expensive per proposal step than `gpt2`, and on this hardware that extra
cost outweighs the acceptance-rate gain. This directly motivates the
load-based routing rule (favor the cheap draft model under load) and is
exactly the kind of tradeoff a real speculative decoding deployment has to
tune for its specific hardware.

Per-prompt numbers varied more than the averages suggest — first-call
latency in particular was inflated by MPS kernel warmup (see Limitations).

### Load test — throughput and latency vs. concurrency

`load_test/load_test.py --levels 1 10 50 100 --max-new-tokens 20`, mixed
traffic (~55% repeated/reworded "hot" prompts to exercise the cache, ~45%
unique "long-tail" prompts to force real generation):

| Concurrency | Wall time (s) | Throughput (req/s) | Avg latency (s) | p95 latency (s) | Cache hit rate | Fallbacks |
|------------:|---------------:|---------------------:|------------------:|-------------------:|-----------------:|------------:|
| 1           | 10.26          | 0.10                  | 10.26              | 10.26               | 0%                | 0           |
| 10          | 16.66          | 0.60                  | 9.86                | 16.65               | 30%               | 0           |
| 50          | 22.32          | 2.24                  | 12.61               | 20.70               | 72%               | 0           |
| 100         | 41.53          | 2.41                  | 22.43               | 41.27               | 77%               | 0           |

All 161 requests across all levels succeeded (0 failures). (These numbers
come from `load_test.py`'s own client-side timer, which always measured full
round-trip time including queue wait. A separate bug meant the *dashboard's*
per-request log undercounted latency, since its internal timer started only
once a request left the queue — fixed in `serving/api.py` by starting the
timer at submission instead of at dequeue. The table above is a fresh
re-run confirming both measurements now agree.) What this shows honestly:

- **Throughput scales with concurrency, but the cache — not the engine — is
  doing most of that work.** As more requests pile up, the chance any given
  one matches something already cached goes up (0% → 77%), and cache hits
  return in well under a second. That's the main reason throughput climbs.
- **Latency grows with concurrency because the engine itself is a single
  sequential worker.** The request batcher groups arrivals, but within a
  batch, prompts are still processed through the model one at a time (see
  Limitations) — so p95 latency at concurrency=100 (41s) reflects genuine
  queueing wait, not per-request slowness.
- **Fallback never triggered under normal load** (draft models don't
  naturally fail on well-formed input). I verified the fallback path
  directly by injecting a simulated draft-model failure — it caught the
  exception, logged the reason (`RuntimeError: simulated draft model
  failure`), and successfully completed the request via target-only
  decoding.

## Honest limitations

- **Small models, modest speedup.** GPT-2/GPT-2-medium/GPT-2-large are tiny
  by modern standards. The ~1.1x speedup measured here is real but far below
  the 2-3x+ speedups reported in papers using much larger target models
  (e.g. 13B-70B) — the bigger the target model relative to the draft, the
  more a single verify pass saves versus token-by-token generation. On this
  hardware/model pairing, per-forward-pass fixed overhead (especially MPS
  dispatch overhead — see below) eats into a lot of the theoretical gain.
- **MPS (Apple Silicon GPU) has real per-op dispatch overhead.** An early
  version of the accept/reject loop called `.item()` on GPU tensors inside a
  per-token Python loop, forcing a CPU↔GPU sync on every call — this alone
  made speculative decoding *slower* than normal decoding (0.39x) until I
  vectorized the accept/reject decision into one batched comparison per
  round. This is a genuine, measured lesson about naive GPU code, not a
  hypothetical one.
- **No KV-cache reuse across speculative rounds.** For simplicity and
  correctness, both the draft and target models recompute from the full
  growing sequence each round rather than carrying forward a truncatable KV
  cache. This is the single biggest performance leaf on the table — a
  production implementation would maintain and truncate the cache across
  rounds, which would meaningfully increase the speedup.
- **"Batching" groups arrivals, but doesn't fuse them into one forward
  pass.** The request batcher (`queue_batcher.py`) collects requests that
  arrive within ~50ms so nothing gets dropped under concurrent load, but
  requests within a batch still run through the engine sequentially, one
  prompt at a time. True batched (padded, masked) speculative decoding
  across multiple different prompts — with per-sequence accept/reject
  bookkeeping — is what real systems like vLLM/TGI do, and is a substantially
  bigger engineering effort than this project's scope.
- **Single in-process model worker.** There's one draft/target model pair
  loaded once at startup, and the batcher processes one request at a time
  against it — no multi-worker/multi-GPU parallelism. This is the direct
  cause of the latency growth seen at concurrency=100 in the load test.
- **No real users.** `load_test.py` fires simulated concurrent traffic with
  a mix of repeated and unique prompts. I don't have a way to generate real
  multi-user traffic patterns for this project, so concurrency behavior is
  inferred from synthetic load, not production telemetry.
- **Cache is in-memory and unbounded across restarts.** `SemanticCache`
  keeps embeddings in a plain Python list/NumPy array with a hard cap
  (500 entries, oldest evicted first) and nothing persists across process
  restarts — fine at this scale, not how you'd do it in production (vector
  DB, disk persistence, per-tenant isolation).
- **Metrics are a flat CSV**, not a real time-series database — deliberate,
  since a full DB was explicitly out of scope for a project this size, but
  it means the dashboard re-reads and re-parses the whole file on every
  refresh rather than querying incrementally.
