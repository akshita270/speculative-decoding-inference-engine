from __future__ import annotations

import threading

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticCache:
    """Cache keyed by *meaning*, not exact string match. Embeds every incoming
    prompt with a small sentence-transformer model and compares it against
    previously seen prompts via cosine similarity. If something close enough
    has already been answered, we skip the LLM entirely and return the stored
    response.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.92, max_entries: int = 500):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.max_entries = max_entries
        self._prompts: list[str] = []
        self._embeddings: np.ndarray | None = None  # [N, D], L2-normalized
        self._responses: list[dict] = []
        self._lock = threading.Lock()

    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def get(self, prompt: str):
        """Return (cached_response, similarity) if a close-enough match exists,
        else (None, best_similarity_seen) so callers can log near-misses too.
        """
        with self._lock:
            if self._embeddings is None or len(self._prompts) == 0:
                return None, 0.0
            query = self._embed(prompt)
            # embeddings are L2-normalized, so dot product == cosine similarity
            sims = self._embeddings @ query
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= self.threshold:
                return self._responses[best_idx], best_sim
            return None, best_sim

    def put(self, prompt: str, response: dict):
        with self._lock:
            vec = self._embed(prompt)
            if self._embeddings is None:
                self._embeddings = vec.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, vec])
            self._prompts.append(prompt)
            self._responses.append(response)

            # Simple bounded cache: drop the oldest entry once we hit max_entries.
            if len(self._prompts) > self.max_entries:
                self._prompts.pop(0)
                self._responses.pop(0)
                self._embeddings = self._embeddings[1:]

    def size(self) -> int:
        return len(self._prompts)
