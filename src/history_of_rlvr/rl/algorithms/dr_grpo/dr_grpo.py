"""Dr. GRPO (Group Relative Policy Optimization Done Right).
Reference: Understanding R1-Zero-Like Training (https://arxiv.org/abs/2503.20783)
"""

import torch
from typing import Tuple


def dr_grpo_loss(
    trainer_log_probs: torch.Tensor,
    inference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon_lower: float = 0.2,
    epsilon_upper: float = 0.2,
    beta: float = 0.0,
    max_log_ratio: float = 10.0,
    max_tokens: int = 1024,
) -> Tuple[torch.Tensor, dict]:
    """
    Unbiased GRPO that removes length and std normalization biases.

    Normalizes by constant max_tokens instead of actual response length.
    Advantages should NOT be std-normalized (handled at config level).
    """
    log_ratio = trainer_log_probs - inference_log_probs
    log_ratio = torch.clamp(log_ratio, min=-max_log_ratio, max=max_log_ratio)
    ratio = torch.exp(log_ratio)

    adv = advantages.unsqueeze(1)

    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * torch.clamp(ratio, 1.0 - epsilon_lower, 1.0 + epsilon_upper)
    pg_loss = torch.max(pg_loss1, pg_loss2)

    kl_log_ratio = inference_log_probs - trainer_log_probs
    kl_log_ratio = torch.clamp(kl_log_ratio, min=-max_log_ratio, max=max_log_ratio)
    kl = torch.exp(kl_log_ratio) - kl_log_ratio - 1.0

    per_token_loss = (pg_loss + beta * kl) * completion_mask

    # Constant normalization: per-sample sum / max_tokens, then mean across samples
    per_sample_loss = per_token_loss.sum(dim=1) / max_tokens
    loss = per_sample_loss.mean()

    num_valid_tokens = completion_mask.sum()
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
