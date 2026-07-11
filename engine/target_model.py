from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.model_utils import best_device


class TargetModel:
    """The large, expensive, "ground truth" model. In speculative decoding it never
    generates tokens one at a time -- it only ever VERIFIES a whole batch of draft
    tokens in a single forward pass. That single batched pass costs roughly the same
    wall-clock time as generating one token normally would, which is the entire
    source of the speedup: we get up to K tokens for the price of ~1.
    """

    def __init__(self, model_name: str = "gpt2-large", device: str | None = None):
        self.name = model_name
        self.device = device or best_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def verify(self, input_ids: torch.Tensor, draft_tokens: torch.Tensor, temperature: float = 1.0):
        """Run ONE forward pass over [input_ids + draft_tokens] and return the target
        model's probability distribution at every position needed to judge the draft:
        one distribution per draft token (what the target thinks should come next,
        given everything up to but not including that draft token), plus one extra
        "bonus" distribution for the token right after the last draft token -- used
        if every draft token gets accepted.

        Returns:
            dists: FloatTensor [1, k+1, V]
        """
        full = torch.cat([input_ids, draft_tokens], dim=-1)
        out = self.model(full, use_cache=False)
        logits = out.logits  # [1, seq_len, vocab]

        k = draft_tokens.shape[-1]
        start = input_ids.shape[-1] - 1  # logits here predict the first draft token
        relevant_logits = logits[:, start:start + k + 1, :]
        dists = torch.softmax(relevant_logits / max(temperature, 1e-5), dim=-1)
        return dists

    @torch.no_grad()
    def sample_next(self, input_ids: torch.Tensor, temperature: float = 1.0):
        """Single-token generation step, used for normal (non-speculative) decoding
        and as the fallback path when the draft model fails.
        """
        out = self.model(input_ids, use_cache=False)
        logits = out.logits[:, -1, :]
        dist = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
        if temperature <= 1e-5:
            next_id = torch.argmax(dist, dim=-1, keepdim=True)
        else:
            next_id = torch.multinomial(dist, num_samples=1)
        return next_id
