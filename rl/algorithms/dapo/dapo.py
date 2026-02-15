"""
Standalone DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization).

Pure PyTorch implementation – no trainer dependencies.
Reference: ByteDance DAPO (https://arxiv.org/abs/2503.14476)

J_DAPO(θ) = 1/(Σ|o_i|) Σ_i Σ_t min(r_t·A, clip(r_t, 1-ε_low, 1+ε_high)·A)

Key differences from GRPO:
  1. Clip-Higher: decoupled ε_low (0.2) and ε_high (0.28) to prevent
     entropy collapse by leaving more room for low-probability exploration
     tokens to be uplifted.
  2. Token-Level Loss: averaged over total completion tokens across all
     samples (not per-sample then across samples). Gives longer sequences
     proportionally more influence, which is critical for long-CoT RL.
  3. No KL penalty: β=0 (removed entirely). In long-CoT reasoning, the
     policy can diverge significantly from the initial model, so the KL
     restriction is unnecessary.
  4. Reuses compute_group_advantages from GRPO (same advantage formula).

Dynamic Sampling and Overlong Reward Shaping are implemented in the
orchestrator (automatic when algorithm="dapo") and are not part of this
loss function.
"""

import torch
from typing import Tuple


def dapo_loss(
    trainer_log_probs: torch.Tensor,
    inference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon_lower: float = 0.2,
    epsilon_upper: float = 0.28,
    max_log_ratio: float = 10.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the DAPO loss (Decoupled Clip and Dynamic Sampling Policy Optimization).

    Like GRPO but with three key changes:
      - Decoupled clipping: ε_low=0.2, ε_high=0.28 (Clip-Higher)
      - Token-level averaging: loss divided by total completion tokens
      - No KL penalty term (β=0)

    loss = -1/(Σ|o_i|) Σ_i Σ_t min(ratio·A, clip(ratio, 1-ε_low, 1+ε_high)·A)

    Args:
        trainer_log_probs:   (B*G, seq_len) per-token log probs from current policy
        inference_log_probs: (B*G, seq_len) per-token log probs from vLLM (old policy)
        advantages:          (B*G,) group-relative advantages
        completion_mask:     (B*G, seq_len) binary mask (1 for completion tokens)
        epsilon_lower:       lower clipping bound (default: 0.2)
        epsilon_upper:       upper clipping bound (default: 0.28, Clip-Higher)
        max_log_ratio:       clamp log ratios to ±this value (prevents exp overflow)

    Returns:
        loss: scalar loss tensor (token-level averaged)
        stats: dict with auxiliary metrics for logging
    """
    # ── Importance ratio ───────────────────────────────────────────────
    log_ratio = trainer_log_probs - inference_log_probs
    log_ratio = torch.clamp(log_ratio, min=-max_log_ratio, max=max_log_ratio)
    ratio = torch.exp(log_ratio)  # (B*G, seq_len)

    # ── Broadcast advantages to token level ────────────────────────────
    adv = advantages.unsqueeze(1)  # (B*G, 1)

    # ── Clipped surrogate objective (Eq. 10) ───────────────────────────
    # Same min() as GRPO but with decoupled ε_low and ε_high
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * torch.clamp(ratio, 1.0 - epsilon_lower, 1.0 + epsilon_upper)
    pg_loss = torch.max(pg_loss1, pg_loss2)  # (B*G, seq_len)

    # ── No KL penalty (β=0 in DAPO) ───────────────────────────────────
    # Unlike GRPO, DAPO removes the KL divergence term entirely.
    # In long-CoT reasoning, the policy can diverge significantly from
    # the initial model, so the KL restriction is unnecessary.

    # ── Mask and token-level average (Eq. 12) ──────────────────────────
    # Token-level loss: 1/(Σ|o_i|) Σ_i Σ_t loss_i,t
    # This gives longer sequences proportionally more influence on the
    # gradient, unlike GRPO's sample-level averaging.
    per_token_loss = pg_loss * completion_mask

    num_valid_tokens = completion_mask.sum()
    loss = per_token_loss.sum() / (num_valid_tokens + 1e-8)

    # ── Logging stats ──────────────────────────────────────────────────
    with torch.no_grad():
        masked_pg = per_token_loss.sum() / (num_valid_tokens + 1e-8)
        mean_ratio = (ratio * completion_mask).sum() / (num_valid_tokens + 1e-8)
        clipped = ((ratio < 1.0 - epsilon_lower) | (ratio > 1.0 + epsilon_upper))
        clip_frac = (clipped.float() * completion_mask).sum() / (num_valid_tokens + 1e-8)

    stats = {
        "loss/total": loss.item(),
        "loss/pg_loss": masked_pg.item(),
        "train/importance_ratio": mean_ratio.item(),
        "train/clip_fraction": clip_frac.item(),
    }

    return loss, stats
