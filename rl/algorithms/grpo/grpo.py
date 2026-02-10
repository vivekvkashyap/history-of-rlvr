"""
Standalone GRPO (Group Relative Policy Optimization) loss functions.

Pure PyTorch implementation – no trainer dependencies.
Reference: DeepSeekMath paper (https://arxiv.org/abs/2402.03300)
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def compute_group_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute group-relative advantages from rewards.

    For each group of G completions for the same prompt, the advantage is:
        A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

    Args:
        rewards: (B * G,) flat tensor of rewards
        num_generations: G – number of completions per prompt
        eps: small constant for numerical stability

    Returns:
        advantages: (B * G,) tensor of normalized advantages
    """
    # Reshape to (B, G)
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)      # (B, 1)
    std = grouped.std(dim=1, keepdim=True)         # (B, 1)
    advantages = (grouped - mean) / (std + eps)    # (B, G)
    return advantages.view(-1)                     # (B * G,)


def grpo_loss(
    trainer_log_probs: torch.Tensor,
    inference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon: float = 0.2,
    beta: float = 0.04,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the full GRPO loss.

    Uses two sets of logprobs:
    - ``trainer_log_probs``:   from the current forward pass (with gradients)
    - ``inference_log_probs``: from vLLM at generation time (old policy
      baseline, also used as KL reference)

    J_GRPO = -1/|tokens| * sum [
        min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
    ] + beta * KL(pi_theta || pi_inference)

    Args:
        trainer_log_probs:   (B * G, seq_len) per-token log probs from current policy
        inference_log_probs: (B * G, seq_len) per-token log probs from vLLM
        advantages:          (B * G,) group-relative advantages
        completion_mask:     (B * G, seq_len) binary mask (1 for real tokens, 0 for padding)
        epsilon:             clipping range
        beta:                KL penalty coefficient

    Returns:
        loss: scalar loss tensor
        stats: dict with auxiliary metrics for logging
    """
    # ── Policy ratio ───────────────────────────────────────────────────
    ratio = torch.exp(trainer_log_probs - inference_log_probs)  # (B*G, seq_len)

    # ── Broadcast advantages to token level ────────────────────────────
    # advantages is (B*G,), we need (B*G, 1) to broadcast over seq_len
    adv = advantages.unsqueeze(1)  # (B*G, 1)

    # ── Clipped surrogate loss ─────────────────────────────────────────
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    pg_loss = torch.max(pg_loss1, pg_loss2)  # (B*G, seq_len)

    # ── KL penalty (vs inference / old policy) ─────────────────────────
    log_ratio = inference_log_probs - trainer_log_probs
    kl = torch.exp(log_ratio) - log_ratio - 1.0  # (B*G, seq_len)

    # ── Combine and mask ───────────────────────────────────────────────
    per_token_loss = pg_loss + beta * kl                   # (B*G, seq_len)
    per_token_loss = per_token_loss * completion_mask       # zero out padding

    # Average over valid tokens
    num_valid_tokens = completion_mask.sum()
    loss = per_token_loss.sum() / (num_valid_tokens + 1e-8)

    # ── Logging stats ──────────────────────────────────────────────────
    with torch.no_grad():
        masked_kl = (kl * completion_mask).sum() / (num_valid_tokens + 1e-8)
        masked_pg = (pg_loss * completion_mask).sum() / (num_valid_tokens + 1e-8)
        mean_ratio = (ratio * completion_mask).sum() / (num_valid_tokens + 1e-8)
        clip_frac = (
            ((ratio - 1.0).abs() > epsilon).float() * completion_mask
        ).sum() / (num_valid_tokens + 1e-8)

    stats = {
        "loss/total": loss.item(),
        "loss/pg_loss": masked_pg.item(),
        "loss/kl": masked_kl.item(),
        "train/importance_ratio": mean_ratio.item(),
        "train/clip_fraction": clip_frac.item(),
    }

    return loss, stats
