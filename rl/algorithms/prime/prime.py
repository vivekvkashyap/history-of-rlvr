"""
Prime-RL style loss: masked importance sampling with stop-gradient.

Pure PyTorch implementation – no trainer dependencies.
Reference: Prime Intellect's Prime-RL (https://github.com/PrimeIntellect-ai/prime-rl)

Key design choices (different from GRPO/CISPO/DAPO):
  - MASKING instead of clipping: tokens with importance ratio outside
    [token_mask_low, token_mask_high] are zeroed out entirely rather
    than having their ratio clamped.
  - Stop-gradient on the weighting coefficient: gradients flow ONLY
    through log π_θ, not through the importance ratio or advantages.
    This makes it a weighted maximum-likelihood objective, which is
    more stable than the PPO-style surrogate.
  - No KL penalty term.

Loss formula:
    coeff = ratio * advantages   (detached)
    loss  = -Σ (coeff · log π_θ)[keep_mask] / Σ loss_mask

where keep_mask = loss_mask AND NOT (ratio < low OR ratio > high)
"""

import torch
from typing import Tuple


def prime_loss(
    trainer_log_probs: torch.Tensor,
    inference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    token_mask_low: float = 0.125,
    token_mask_high: float = 8.0,
    max_log_ratio: float = 10.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the Prime-RL style masked IS loss.

    Tokens whose importance ratio falls outside [token_mask_low, token_mask_high]
    are masked out (contribute zero to both loss and gradient). Remaining tokens
    get a weighted MLE loss where the weight is (ratio * advantage).detach().

    Args:
        trainer_log_probs:   (B*G, seq_len) per-token log probs from current policy
        inference_log_probs: (B*G, seq_len) per-token log probs from old policy (vLLM)
        advantages:          (B*G,) group-relative advantages
        completion_mask:     (B*G, seq_len) binary mask (1 for completion tokens)
        token_mask_low:      mask tokens with ratio < this value (default 0.125)
        token_mask_high:     mask tokens with ratio > this value (default 8.0)
        max_log_ratio:       clamp log ratios to ±this value (prevents exp overflow)

    Returns:
        loss: scalar loss tensor
        stats: dict with auxiliary metrics for logging
    """
    # ── Importance ratio ───────────────────────────────────────────────
    log_ratio = trainer_log_probs - inference_log_probs
    log_ratio = torch.clamp(log_ratio, min=-max_log_ratio, max=max_log_ratio)
    ratio = torch.exp(log_ratio)  # (B*G, seq_len)

    # ── Token-level masking (NOT clipping) ─────────────────────────────
    # Tokens with ratio outside [low, high] are excluded entirely.
    is_masked = (ratio < token_mask_low) | (ratio > token_mask_high)
    keep_mask = completion_mask.bool() & ~is_masked

    # ── Broadcast advantages to token level ────────────────────────────
    adv = advantages.unsqueeze(1)  # (B*G, 1)

    # ── Weighted MLE loss with stop-gradient on coefficient ────────────
    # coeff = ratio * advantages — detached so gradients only flow
    # through trainer_log_probs (i.e., log π_θ).
    coeff = (ratio * adv).detach()
    per_token_loss = -(coeff * trainer_log_probs)

    # ── Average over ALL valid completion tokens (not just kept) ───────
    # Normalizing by total completion tokens (not kept tokens) prevents
    # the loss scale from fluctuating as the masking fraction changes.
    num_valid_tokens = completion_mask.sum()
    loss = (per_token_loss * keep_mask.float()).sum() / (num_valid_tokens + 1e-8)

    # ── Mismatch KL (for monitoring off-policyness) ────────────────────
    with torch.no_grad():
        token_mismatch_kl = ratio - log_ratio - 1.0  # D_KL(π_old || π_θ)

    # ── Logging stats ──────────────────────────────────────────────────
    with torch.no_grad():
        masked_pg = (per_token_loss * keep_mask.float()).sum() / (num_valid_tokens + 1e-8)
        mean_ratio = (ratio * completion_mask).sum() / (num_valid_tokens + 1e-8)
        mask_frac = is_masked[completion_mask.bool()].float().mean() if completion_mask.sum() > 0 else torch.tensor(0.0)
        masked_low_frac = (ratio < token_mask_low)[completion_mask.bool()].float().mean() if completion_mask.sum() > 0 else torch.tensor(0.0)
        masked_high_frac = (ratio > token_mask_high)[completion_mask.bool()].float().mean() if completion_mask.sum() > 0 else torch.tensor(0.0)
        mismatch_kl = (token_mismatch_kl * completion_mask).sum() / (num_valid_tokens + 1e-8)

    stats = {
        "loss/total": loss.item(),
        "loss/pg_loss": masked_pg.item(),
        "loss/mismatch_kl": mismatch_kl.item(),
        "train/importance_ratio": mean_ratio.item(),
        "train/mask_fraction": mask_frac.item(),
        "train/mask_fraction_low": masked_low_frac.item(),
        "train/mask_fraction_high": masked_high_frac.item(),
    }

    return loss, stats
