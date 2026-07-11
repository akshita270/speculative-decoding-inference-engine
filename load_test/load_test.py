"""Simulated concurrent traffic for the inference API.

There are no real users for this project, so this script IS the traffic: it
fires batches of concurrent requests at increasing concurrency levels
(1, 10, 50, 100 by default) using a mix of:
  - "hot" prompts, repeated verbatim or lightly reworded -- these exercise the
    semantic cache and the request batcher (many arrive close together).
  - "long-tail" prompts, mostly unique per request -- these force the engine
    (draft + target models) to actually run, so we can see how latency and
    throughput hold up as concurrent demand increases.

Usage:
    python load_test/load_test.py --levels 1 10 50 100 --max-new-tokens 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp

API_URL = "http://127.0.0.1:8000/generate"

HOT_PROMPTS = [
    "The best way to learn programming is",
    "Artificial intelligence will change the world by",
    "The key to writing clean code is",
    "In the future, machine learning models will",
    "A good leader should always",
]

LONG_TAIL_PROMPTS = [
    "The most important skill for a data analyst is",
    "Climate change is a serious problem because",
    "The history of computers began when",
    "The internet has changed how we",
    "My favorite way to relax on weekends is",
    "The stock market moved today because",
    "Space exploration matters because",
    "The recipe for a good cup of coffee is",
    "Renewable energy is important since",
    "The biggest challenge in software engineering is",
    "A healthy morning routine includes",
    "The future of remote work looks like",
    "Learning a new language requires",
    "The best advice I ever received was",
    "Cities of the future will need to",
    "The role of a good manager is to",
    "Scientists recently discovered that",
    "The economy is likely to shift as",
    "Building a strong team means",
    "The next big breakthrough in technology will be",
]


def make_prompt_pool(n: int, request_index_offset: int = 0) -> list:
    pool = []
    for i in range(n):
        if random.random() < 0.55:
            base = random.choice(HOT_PROMPTS)
            # Half the time reword slightly -- tests the *semantic* cache match,
            # not just exact-string caching.
            prompt = base if random.random() < 0.5 else base.rstrip(".") + ", in my opinion,"
        else:
            base = random.choice(LONG_TAIL_PROMPTS)
            # Tag with a unique index so most long-tail prompts are genuinely
            # novel and force real generation instead of a cache hit.
            prompt = f"{base} (case #{request_index_offset + i})"
        pool.append(prompt)
    return pool


async def fire_request(session: aiohttp.ClientSession, prompt: str, max_new_tokens: int) -> dict:
    t0 = time.perf_counter()
    try:
        async with session.post(
            API_URL,
            json={"prompt": prompt, "max_new_tokens": max_new_tokens, "k": 4, "temperature": 0.7},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            return {"ok": True, "latency": time.perf_counter() - t0, **data}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "latency": time.perf_counter() - t0, "error": str(e)}


async def run_level(concurrency: int, max_new_tokens: int, offset: int) -> dict:
    prompts = make_prompt_pool(concurrency, offset)
    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        results = await asyncio.gather(*[fire_request(session, p, max_new_tokens) for p in prompts])
        wall_time = time.perf_counter() - start

    ok = [r for r in results if r["ok"]]
    failed = len(prompts) - len(ok)
    latencies = sorted(r["latency"] for r in ok)
    cache_hits = sum(1 for r in ok if r.get("cache_hit"))
    acc_rates = [r["acceptance_rate"] for r in ok if r.get("acceptance_rate") is not None]
    fallbacks = sum(1 for r in ok if r.get("fallback_triggered"))

    def pct(p):
        if not latencies:
            return None
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return round(latencies[idx], 3)

    return {
        "concurrency": concurrency,
        "num_requests": len(prompts),
        "num_ok": len(ok),
        "num_failed": failed,
        "wall_time_s": round(wall_time, 2),
        "throughput_req_per_s": round(len(ok) / wall_time, 3) if wall_time > 0 else 0,
        "latency_avg_s": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_p50_s": pct(0.50),
        "latency_p95_s": pct(0.95),
        "cache_hit_rate": round(cache_hits / len(ok), 3) if ok else None,
        "avg_acceptance_rate": round(statistics.mean(acc_rates), 3) if acc_rates else None,
        "fallback_count": fallbacks,
    }


async def main(levels: list, max_new_tokens: int) -> list:
    summaries = []
    offset = 0
    for level in levels:
        print(f"\n=== Concurrency level: {level} ===")
        summary = await run_level(level, max_new_tokens, offset)
        offset += level
        for k, v in summary.items():
            print(f"  {k}: {v}")
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--out", default="load_test/results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    summaries = asyncio.run(main(args.levels, args.max_new_tokens))

    with open(args.out, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved results to {args.out}")
