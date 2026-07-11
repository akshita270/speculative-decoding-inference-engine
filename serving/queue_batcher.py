from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass
class _QueuedRequest:
    payload: Any
    future: "asyncio.Future"


class RequestBatcher:
    """Groups requests that arrive within a short time window into a batch
    before running them through the engine, so concurrent requests queue up
    instead of getting dropped or fighting each other for the model.

    Simplification (documented, not hidden): within a batch, requests are
    still run through the engine sequentially, one prompt at a time. True
    fused matrix-batched generation across different prompts would require
    padding + per-sequence accept/reject bookkeeping in the speculative
    decoding loop -- a much bigger lift than this weekend project calls for.
    What batching buys us here: (1) no dropped requests under concurrent
    load, everything queues instead, and (2) a natural place to observe and
    log batch sizes, which is what the dashboard's batching story is about.
    """

    def __init__(
        self,
        process_one: Callable[[Any, int], Awaitable[Any]],
        batch_window_s: float = 0.05,
        max_batch_size: int = 8,
    ):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._process_one = process_one
        self.batch_window_s = batch_window_s
        self.max_batch_size = max_batch_size
        self._worker_task = None

    def start(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def submit(self, payload: Any) -> Any:
        fut = asyncio.get_event_loop().create_future()
        await self._queue.put(_QueuedRequest(payload, fut))
        return await fut

    async def _worker_loop(self):
        while True:
            first = await self._queue.get()
            batch = [first]

            # keep absorbing more requests until the arrival window closes or
            # we hit the max batch size -- this is the "group nearby-in-time
            # requests together" behavior.
            deadline = time.monotonic() + self.batch_window_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            batch_size = len(batch)
            for item in batch:
                try:
                    result = await self._process_one(item.payload, batch_size)
                    if not item.future.done():
                        item.future.set_result(result)
                except Exception as e:  # noqa: BLE001
                    if not item.future.done():
                        item.future.set_exception(e)
