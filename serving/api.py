from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from cache.semantic_cache import SemanticCache
from engine.draft_model import DraftModel
from engine.model_utils import best_device
from engine.target_model import TargetModel
from observability.logger import log_request
from serving.fallback import run_with_fallback
from serving.queue_batcher import RequestBatcher
from serving.router import LoadBasedRouter

DEVICE = best_device()

# Models are loaded once at process startup and reused for every request.
TARGET = TargetModel("gpt2-large", device=DEVICE)
DRAFTS = {
    "gpt2": DraftModel("gpt2", device=DEVICE),
    "gpt2-medium": DraftModel("gpt2-medium", device=DEVICE),
}
ROUTER = LoadBasedRouter(fast_model="gpt2", better_model="gpt2-medium", high_load_threshold=3)
CACHE = SemanticCache(threshold=0.92)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    k: int = 4
    temperature: float = 0.8


class GenerateResponse(BaseModel):
    text: str
    latency_s: float
    tokens_per_sec: float
    acceptance_rate: Optional[float] = None
    cache_hit: bool
    draft_model: Optional[str] = None
    fallback_triggered: bool
    fallback_reason: Optional[str] = None
    batch_size: int


async def _process_one(payload: tuple, batch_size: int) -> GenerateResponse:
    # `start` is captured by the caller at submission time, before the
    # request even entered the queue -- so latency_s below measures the full
    # wait (queue + processing), not just processing time.
    req, start = payload

    cached, _similarity = CACHE.get(req.prompt)
    if cached is not None:
        resp = GenerateResponse(
            text=cached["text"],
            latency_s=time.perf_counter() - start,
            tokens_per_sec=cached.get("tokens_per_sec", 0.0),
            acceptance_rate=cached.get("acceptance_rate"),
            cache_hit=True,
            draft_model=cached.get("draft_model"),
            fallback_triggered=False,
            fallback_reason=None,
            batch_size=batch_size,
        )
    else:
        # load-based routing: pick the draft model based on how many requests
        # are currently in the system (queued + processing) right now.
        draft_name = ROUTER.choose_draft_model()
        draft = DRAFTS[draft_name]

        loop = asyncio.get_running_loop()
        result, fallback_triggered, fallback_reason = await loop.run_in_executor(
            None, run_with_fallback, req.prompt, draft, TARGET, req.max_new_tokens, req.k, req.temperature
        )

        CACHE.put(req.prompt, result)
        resp = GenerateResponse(
            text=result["text"],
            latency_s=time.perf_counter() - start,
            tokens_per_sec=result["tokens_per_sec"],
            acceptance_rate=result.get("acceptance_rate"),
            cache_hit=False,
            draft_model=result.get("draft_model"),
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
            batch_size=batch_size,
        )

    log_request(
        prompt=req.prompt,
        latency_s=resp.latency_s,
        tokens_per_sec=resp.tokens_per_sec,
        acceptance_rate=resp.acceptance_rate,
        cache_hit=resp.cache_hit,
        draft_model=resp.draft_model,
        fallback_triggered=resp.fallback_triggered,
        fallback_reason=resp.fallback_reason,
        batch_size=resp.batch_size,
    )
    return resp


BATCHER = RequestBatcher(process_one=_process_one, batch_window_s=0.05, max_batch_size=8)


@asynccontextmanager
async def lifespan(app: FastAPI):
    BATCHER.start()
    yield


app = FastAPI(title="Speculative Decoding Inference Engine", lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    # Timer starts here, before queueing, so the logged latency reflects
    # what a caller actually experiences -- queue wait included.
    start = time.perf_counter()

    # "load" is the count of requests currently between submission and
    # completion -- queued or processing. Tracked for the whole request
    # lifecycle (not just the inference call) since the queue wait is part
    # of what "the system is under load" means.
    ROUTER.enter()
    try:
        return await BATCHER.submit((req, start))
    finally:
        ROUTER.exit()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "current_load": ROUTER.current_load,
        "cache_size": CACHE.size(),
    }
