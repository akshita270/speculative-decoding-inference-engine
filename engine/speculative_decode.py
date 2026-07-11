from __future__ import annotations

import time

import torch

from engine.draft_model import DraftModel
from engine.target_model import TargetModel


@torch.no_grad()
def speculative_generate(
    prompt: str,
    draft: DraftModel,
    target: TargetModel,
    max_new_tokens: int = 50,
    k: int = 4,
    temperature: float = 1.0,
) -> dict:
    """Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023).

    Loop, each round:
      1. PROPOSE  -- draft model samples k tokens ahead, one at a time (cheap, sequential).
      2. VERIFY   -- target model checks all k tokens in a single forward pass (expensive,
                      but paid for only once per round instead of once per token).
      3. ACCEPT/REJECT -- walk the k draft tokens left to right:
           - accept token i with probability min(1, p_target(x_i) / p_draft(x_i)).
             If the target agrees the token was at least as likely as the draft thought,
             it's always accepted (ratio >= 1). If the target finds it less likely, we
             still sometimes accept it, proportionally to how close the two models are --
             this specific accept rule is what makes the overall sampling distribution
             *mathematically identical* to sampling from the target model alone.
           - on the FIRST rejection, every draft token from that point on is discarded
             (they were generated conditioned on a token the target never would have
             produced, so they're not trustworthy) and we resample a replacement token
             from the residual distribution max(0, p_target - p_draft), renormalized.
             This residual sampling is what keeps the math exact rather than just
             "falling back to the target's argmax".
           - if ALL k tokens are accepted, we get a k+1'th "bonus" token for free,
             sampled from the target's distribution at the position right after the
             last draft token (we already paid for that forward pass, so may as well
             use it).
      4. Append whatever tokens were accepted/resampled/bonus this round and repeat
         until max_new_tokens is reached or EOS is generated.

    Returns a dict with the generated text and measured performance stats
    (tokens/sec, draft acceptance rate) -- both are exactly what we log per-request.
    """
    tokenizer = target.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)
    prompt_len = input_ids.shape[-1]
    generated = input_ids

    total_accepted = 0
    total_proposed = 0
    start = time.perf_counter()

    while generated.shape[-1] - prompt_len < max_new_tokens:
        remaining = max_new_tokens - (generated.shape[-1] - prompt_len)
        k_round = min(k, remaining)

        draft_input = generated.to(draft.device)
        draft_tokens, draft_dists = draft.propose(draft_input, k_round, temperature)
        draft_tokens = draft_tokens.to(target.device)
        draft_dists = draft_dists.to(target.device)

        target_dists = target.verify(generated, draft_tokens, temperature)  # [1, k_round+1, V]

        # Compute the accept/reject decision for all k_round tokens as one batched,
        # on-device operation, then pull the result to Python in a single sync.
        # (Calling .item() inside this loop once per token, per probability, forces a
        # CPU<->GPU sync each time -- on MPS that dispatch overhead is large enough to
        # erase the FLOP savings speculative decoding is supposed to buy. Batching the
        # comparison keeps the same accept/reject math but pays the sync cost once.)
        idx = draft_tokens[0].unsqueeze(1)  # [k_round, 1]
        q = draft_dists[0].gather(1, idx).squeeze(1)  # [k_round]
        p = target_dists[0, :k_round].gather(1, idx).squeeze(1)  # [k_round]
        accept_prob = torch.clamp(p / q.clamp_min(1e-10), max=1.0)
        accept_mask = (torch.rand(k_round, device=target.device) < accept_prob).tolist()
        draft_tokens_list = draft_tokens[0].tolist()

        accepted_tokens = []
        n_accepted = 0
        rejected = False
        reject_idx = None
        for i in range(k_round):
            if accept_mask[i]:
                accepted_tokens.append(draft_tokens_list[i])
                n_accepted += 1
            else:
                reject_idx = i
                rejected = True
                break

        if rejected:
            residual = torch.clamp(target_dists[0, reject_idx] - draft_dists[0, reject_idx], min=0)
            residual_sum = residual.sum()
            resample_dist = residual / residual_sum if residual_sum.item() > 0 else target_dists[0, reject_idx]
            x_new = torch.multinomial(resample_dist, num_samples=1).item()
            accepted_tokens.append(x_new)
        else:
            bonus_dist = target_dists[0, k_round]
            x_bonus = torch.multinomial(bonus_dist, num_samples=1).item()
            accepted_tokens.append(x_bonus)

        total_accepted += n_accepted
        total_proposed += k_round

        new_ids = torch.tensor([accepted_tokens], device=target.device)
        generated = torch.cat([generated, new_ids], dim=-1)

        if tokenizer.eos_token_id in accepted_tokens:
            break

    elapsed = time.perf_counter() - start
    new_tokens = generated.shape[-1] - prompt_len
    text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)

    return {
        "text": text,
        "new_tokens": new_tokens,
        "elapsed_s": elapsed,
        "tokens_per_sec": new_tokens / elapsed if elapsed > 0 else 0.0,
        "acceptance_rate": total_accepted / total_proposed if total_proposed else 0.0,
        "draft_model": draft.name,
        "target_model": target.name,
    }
