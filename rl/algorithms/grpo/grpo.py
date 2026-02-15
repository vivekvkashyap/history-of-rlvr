"""
Standalone GRPO (Group Relative Policy Optimization) loss functions.

Pure PyTorch implementation – no trainer dependencies.
Reference: DeepSeek-Math (https://arxiv.org/abs/2402.03300)
           DeepSeek-R1  (https://arxiv.org/abs/2501.12948)

J_GRPO(θ) = E[ 1/|o_i| Σ_t  min(r_t·A_i, clip(r_t, 1-ε_lo, 1+ε_hi)·A_i)
                              - β·D_KL(π_θ || π_ref) ]

where  r_t = π_θ(o_t|...) / π_ref(o_t|...)
       D_KL = π_ref/π_θ - log(π_ref/π_θ) - 1
"""

import torch
from typing import Tuple


def compute_group_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute group-relative advantages from rewards.

    For each group of G completions for the same prompt, the advantage is:
        A_i = r_i - mean(r_group)

    Note: This does NOT normalize by std, matching Prime-RL's implementation.
    Standard deviation normalization can harm training with binary rewards
    and reduce gradient magnitudes excessively.

    Args:
        rewards: (B * G,) flat tensor of rewards
        num_generations: G – number of completions per prompt
        eps: small constant for numerical stability (unused, kept for API compat)

    Returns:
        advantages: (B * G,) tensor of advantages
    """
    # Reshape to (B, G)
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)      # (B, 1)
    advantages = grouped - mean                    # (B, G)
    return advantages.view(-1)                     # (B * G,)


def grpo_loss(
    trainer_log_probs: torch.Tensor,
    inference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon_lower: float = 0.2,
    epsilon_upper: float = 0.2,
    beta: float = 0.04,
    max_log_ratio: float = 10.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the GRPO loss from DeepSeek-Math / DeepSeek-R1.

    loss = -1/|tokens| Σ [ min(ratio·A, clip(ratio, 1-ε_lo, 1+ε_hi)·A) ]
           + β · D_KL(π_θ || π_ref)

    Args:
        trainer_log_probs:   (B*G, seq_len) per-token log probs from current policy
        inference_log_probs:  (B*G, seq_len) per-token log probs from vLLM (reference)
        advantages:           (B*G,) group-relative advantages
        completion_mask:      (B*G, seq_len) binary mask (1 for completion tokens)
        epsilon_lower:       lower clipping bound: ratio clipped to at least 1 - epsilon_lower
        epsilon_upper:       upper clipping bound: ratio clipped to at most 1 + epsilon_upper
        beta:                KL penalty coefficient
        max_log_ratio:       clamp log ratios to ±this value (prevents exp overflow)

    Returns:
        loss: scalar loss tensor (averaged over valid tokens)
        stats: dict with auxiliary metrics for logging
    """
    # ── Importance ratio ───────────────────────────────────────────────
    log_ratio = trainer_log_probs - inference_log_probs
    log_ratio = torch.clamp(log_ratio, min=-max_log_ratio, max=max_log_ratio)
    ratio = torch.exp(log_ratio)  # (B*G, seq_len)

    # ── Broadcast advantages to token level ────────────────────────────
    adv = advantages.unsqueeze(1)  # (B*G, 1)

    # ── Clipped surrogate objective ────────────────────────────────────
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * torch.clamp(ratio, 1.0 - epsilon_lower, 1.0 + epsilon_upper)
    pg_loss = torch.max(pg_loss1, pg_loss2)  # (B*G, seq_len)

    # ── KL divergence: D_KL(π_θ || π_ref) ─────────────────────────────
    # = π_ref/π_θ - log(π_ref/π_θ) - 1
    kl_log_ratio = inference_log_probs - trainer_log_probs
    kl_log_ratio = torch.clamp(kl_log_ratio, min=-max_log_ratio, max=max_log_ratio)
    kl = torch.exp(kl_log_ratio) - kl_log_ratio - 1.0  # (B*G, seq_len)

    # ── Combine and mask ───────────────────────────────────────────────
    per_token_loss = pg_loss + beta * kl
    per_token_loss = per_token_loss * completion_mask

    # Average over valid completion tokens
    num_valid_tokens = completion_mask.sum()
    loss = per_token_loss.sum() / (num_valid_tokens + 1e-8)

    # ── Logging stats ──────────────────────────────────────────────────
    with torch.no_grad():
        masked_kl = (kl * completion_mask).sum() / (num_valid_tokens + 1e-8)
        masked_pg = (pg_loss * completion_mask).sum() / (num_valid_tokens + 1e-8)
        mean_ratio = (ratio * completion_mask).sum() / (num_valid_tokens + 1e-8)
        clipped = ((ratio < 1.0 - epsilon_lower) | (ratio > 1.0 + epsilon_upper))
        clip_frac = (clipped.float() * completion_mask).sum() / (num_valid_tokens + 1e-8)

    stats = {
        "loss/total": loss.item(),
        "loss/pg_loss": masked_pg.item(),
        "loss/kl": masked_kl.item(),
        "train/importance_ratio": mean_ratio.item(),
        "train/clip_fraction": clip_frac.item(),
    }

    return loss, stats
