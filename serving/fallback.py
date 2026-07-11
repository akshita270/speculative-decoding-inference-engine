from __future__ import annotations

import logging

from engine.draft_model import DraftModel
from engine.normal_decode import normal_generate
from engine.speculative_decode import speculative_generate
from engine.target_model import TargetModel

logger = logging.getLogger("fallback")


def run_with_fallback(
    prompt: str,
    draft: DraftModel,
    target: TargetModel,
    max_new_tokens: int = 50,
    k: int = 4,
    temperature: float = 1.0,
) -> tuple[dict, bool, str | None]:
    """Try speculative decoding first. If the draft model (or anything in the
    speculative path) raises, fall back to plain target-only decoding so the
    request still succeeds -- reliability over speed. Every fallback is logged
    with the reason so it shows up in the observability layer.

    Returns: (result_dict, fallback_triggered, fallback_reason)
    """
    try:
        result = speculative_generate(prompt, draft, target, max_new_tokens, k, temperature)
        return result, False, None
    except Exception as e:  # noqa: BLE001 - deliberately broad: any draft-path failure should fall back
        reason = f"{type(e).__name__}: {e}"
        logger.warning("Speculative decoding failed, falling back to normal decoding: %s", reason)
        result = normal_generate(prompt, target, max_new_tokens, temperature)
        result["acceptance_rate"] = None
        result["draft_model"] = None
        return result, True, reason
