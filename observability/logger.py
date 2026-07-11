from __future__ import annotations

import csv
import os
import threading
from datetime import datetime, timezone

CSV_PATH = os.path.join(os.path.dirname(__file__), "metrics_store.csv")

FIELDS = [
    "timestamp",
    "prompt",
    "latency_s",
    "tokens_per_sec",
    "acceptance_rate",
    "cache_hit",
    "draft_model",
    "fallback_triggered",
    "fallback_reason",
    "batch_size",
]

_lock = threading.Lock()


def _ensure_header():
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def log_request(
    prompt: str,
    latency_s: float,
    tokens_per_sec: float,
    acceptance_rate: float | None,
    cache_hit: bool,
    draft_model: str | None,
    fallback_triggered: bool,
    fallback_reason: str | None,
    batch_size: int,
):
    """Append one row of per-request metrics to the CSV metrics store.
    A simple file lock keeps concurrent requests from interleaving writes --
    good enough at this scale; a real system would use a proper queue/DB.
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt[:200],
        "latency_s": round(latency_s, 4),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "acceptance_rate": round(acceptance_rate, 4) if acceptance_rate is not None else "",
        "cache_hit": cache_hit,
        "draft_model": draft_model or "",
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason or "",
        "batch_size": batch_size,
    }
    with _lock:
        _ensure_header()
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
