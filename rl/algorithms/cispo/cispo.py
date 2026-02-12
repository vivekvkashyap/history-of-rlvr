"""
Standalone CISPO (Clipped IS-weight Policy Optimization) loss functions.

Pure PyTorch implementation – no trainer dependencies.
Reference: https://arxiv.org/abs/2506.13585

J_CISPO(θ) = E[ 1/Σ|o_i| Σ_i Σ_t  sg(r̂_i,t(θ)) · Â_i,t · log π_θ(o_i,t | q, o_i,<t) ]

where  r̂_i,t(θ) = clip(r_i,t(θ), 1 - ε_IS_low, 1 + ε_IS_high)
       r_i,t(θ)  = π_θ(o_i,t|...) / π_old(o_i,t|...)

Key differences from GRPO:
  - Clips the IS weight directly instead of the ratio-advantage product
  - No KL penalty term
  - No min() operation between clipped and unclipped objectives
  - Preserves gradient contributions from all tokens (no token dropping)
"""

import torch
from typing import Tuple


def cispo_loss(
    trainer_log_probs: torch.Tensor,
    inference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon_lower: float = 10.0,
    epsilon_upper: float = 0.2,
    max_log_ratio: float = 10.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the CISPO loss (Clipped IS-weight Policy Optimization).

    Unlike GRPO which clips the ratio-advantage product via
        min(r·A, clip(r)·A),
    CISPO clips the importance sampling weight directly:
        loss = -sg(clip(r, 1-ε_low, 1+ε_high)) · A · log π_θ

    The stop-gradient (sg) on the clipped ratio means the ratio acts purely
    as a scalar weight and does not contribute to the gradient — only the
    log π_θ term is differentiated. There is no KL penalty.

    Args:
        trainer_log_probs:   (B*G, seq_len) per-token log probs from current policy
        inference_log_probs: (B*G, seq_len) per-token log probs from old policy (vLLM)
        advantages:          (B*G,) group-relative advantages
        completion_mask:     (B*G, seq_len) binary mask (1 for completion tokens)
        epsilon_lower:       lower clip bound for IS weight: r >= 1 - ε_low
                             (paper sets this to a large value to effectively disable)
        epsilon_upper:       upper clip bound for IS weight: r <= 1 + ε_high
                             (main tuning parameter)
        max_log_ratio:       clamp log ratios to ±this value (prevents exp overflow)

    Returns:
        loss: scalar loss tensor (averaged over valid tokens)
        stats: dict with auxiliary metrics for logging
    """
    # ── Importance ratio ───────────────────────────────────────────────
    log_ratio = trainer_log_probs - inference_log_probs
    log_ratio = torch.clamp(log_ratio, min=-max_log_ratio, max=max_log_ratio)
    ratio = torch.exp(log_ratio)  # (B*G, seq_len)

    # ── Clip the IS weight directly (Eq. 5) ────────────────────────────
    clipped_ratio = torch.clamp(
        ratio, 1.0 - epsilon_lower, 1.0 + epsilon_upper,
    )

    # ── Broadcast advantages to token level ────────────────────────────
    adv = advantages.unsqueeze(1)  # (B*G, 1)

    # ── CISPO objective (Eq. 4) ────────────────────────────────────────
    # sg(r̂) · Â · log π_θ  — stop-gradient on the clipped ratio
    # The clipped_ratio.detach() ensures gradients only flow through log π_θ
    # (trainer_log_probs), not through the ratio itself.
    per_token_loss = -(clipped_ratio.detach() * adv * trainer_log_probs)

    # ── Mask and average ───────────────────────────────────────────────
    per_token_loss = per_token_loss * completion_mask

    # Average over valid completion tokens
    num_valid_tokens = completion_mask.sum()
    loss = per_token_loss.sum() / (num_valid_tokens + 1e-8)

    # ── Logging stats ──────────────────────────────────────────────────
    with torch.no_grad():
        masked_pg = (per_token_loss).sum() / (num_valid_tokens + 1e-8)
        mean_ratio = (ratio * completion_mask).sum() / (num_valid_tokens + 1e-8)
        mean_clipped_ratio = (clipped_ratio * completion_mask).sum() / (num_valid_tokens + 1e-8)
        clipped = ((ratio < 1.0 - epsilon_lower) | (ratio > 1.0 + epsilon_upper))
        clip_frac = (clipped.float() * completion_mask).sum() / (num_valid_tokens + 1e-8)

    stats = {
        "loss/total": loss.item(),
        "loss/pg_loss": masked_pg.item(),
        "train/importance_ratio": mean_ratio.item(),
        "train/clipped_ratio_mean": mean_clipped_ratio.item(),
        "train/clip_fraction": clip_frac.item(),
    }

    return loss, stats
