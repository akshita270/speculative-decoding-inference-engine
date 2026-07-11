from __future__ import annotations

import time

import torch

from engine.target_model import TargetModel


@torch.no_grad()
def normal_generate(
    prompt: str,
    target: TargetModel,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
) -> dict:
    """Baseline: plain autoregressive decoding with the target model only, one token
    per forward pass. This is what speculative decoding is being compared against --
    same model, same output distribution, no draft model involved.
    """
    tokenizer = target.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)
    prompt_len = input_ids.shape[-1]
    generated = input_ids

    start = time.perf_counter()
    for _ in range(max_new_tokens):
        next_id = target.sample_next(generated, temperature)
        generated = torch.cat([generated, next_id], dim=-1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    elapsed = time.perf_counter() - start

    new_tokens = generated.shape[-1] - prompt_len
    text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)

    return {
        "text": text,
        "new_tokens": new_tokens,
        "elapsed_s": elapsed,
        "tokens_per_sec": new_tokens / elapsed if elapsed > 0 else 0.0,
        "target_model": target.name,
    }
