from __future__ import annotations

import threading


class LoadBasedRouter:
    """Rule-based load balancing between two draft models.

    Under light load we use the "better" (larger, more accurate) draft model,
    since it gets a higher acceptance rate and there's spare capacity to afford
    its extra cost. Under heavy load we switch to the "fast" (smaller, cheaper)
    draft model to keep per-request latency down, trading some acceptance rate
    for lower cost per proposal step.

    "Load" here is simply the number of requests currently in flight (being
    processed by the engine) -- tracked in-process, no external metrics system.
    """

    def __init__(self, fast_model: str, better_model: str, high_load_threshold: int = 3):
        self.fast_model = fast_model
        self.better_model = better_model
        self.high_load_threshold = high_load_threshold
        self._in_flight = 0
        self._lock = threading.Lock()

    def enter(self) -> int:
        with self._lock:
            self._in_flight += 1
            return self._in_flight

    def exit(self):
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    @property
    def current_load(self) -> int:
        with self._lock:
            return self._in_flight

    def choose_draft_model(self) -> str:
        load = self.current_load
        if load >= self.high_load_threshold:
            return self.fast_model
        return self.better_model
