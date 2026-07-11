from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.model_utils import best_device


class DraftModel:
    """The small, cheap model. Its only job is to propose K tokens ahead quickly.
    It does NOT need to be right -- the target model checks its work. It just needs
    to be fast and "roughly plausible" so that a decent fraction of its guesses hold up.
    """

    def __init__(self, model_name: str = "gpt2", device: str | None = None):
        self.name = model_name
        self.device = device or best_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def propose(self, input_ids: torch.Tensor, k: int, temperature: float = 1.0):
        """Sample k tokens ahead of input_ids, one at a time (draft models are cheap,
        but generation is still inherently sequential -- that's exactly why we hand this
        job to a small model instead of the big one).

        Uses the model's own KV cache for this one proposal round only (not shared
        across speculative rounds) -- keeps the code simple while still avoiding
        redundant recomputation of the k tokens we're proposing within this round.

        Returns:
            tokens: LongTensor [1, k]      -- the proposed token ids
            dists:  FloatTensor [1, k, V]  -- draft's full probability distribution
                                               at each of the k proposal steps (needed
                                               later for the exact accept/reject math)
        """
        tokens = []
        dists = []
        cur = input_ids
        past = None
        for _ in range(k):
            model_input = cur if past is None else cur[:, -1:]
            out = self.model(model_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            dist = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
            if temperature <= 1e-5:
                next_id = torch.argmax(dist, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(dist, num_samples=1)
            tokens.append(next_id)
            dists.append(dist.unsqueeze(1))
            cur = torch.cat([cur, next_id], dim=-1)
        return torch.cat(tokens, dim=-1), torch.cat(dists, dim=1)
