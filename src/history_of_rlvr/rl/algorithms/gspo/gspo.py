"""GSPO (Group Sequence Policy Optimization).
Reference: Qwen Team (https://arxiv.org/abs/2507.18071)
"""

import torch
from typing import Tuple


def gspo_loss(
    trainer_log_probs: torch.Tensor,
    inference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon_lower: float = 3e-4,
    epsilon_upper: float = 4e-4,
    max_log_ratio: float = 10.0,
) -> Tuple[torch.Tensor, dict]:
    """
    GSPO: sequence-level importance ratio with sequence-level clipping.

    Uses geometric mean of token ratios as the sequence-level ratio,
    then clips and optimizes at the sequence level.
    """
    log_ratio = trainer_log_probs - inference_log_probs
    log_ratio = torch.clamp(log_ratio, min=-max_log_ratio, max=max_log_ratio)

    # Sequence-level importance ratio (geometric mean of token ratios)
    num_tokens = completion_mask.sum(dim=1).clamp(min=1)
    seq_log_ratio = (log_ratio * completion_mask).sum(dim=1) / num_tokens
    seq_ratio = torch.exp(seq_log_ratio)

    adv = advantages

    # Sequence-level clipped surrogate
    pg_loss1 = -adv * seq_ratio
    pg_loss2 = -adv * torch.clamp(seq_ratio, 1.0 - epsilon_lower, 1.0 + epsilon_upper)
    pg_loss = torch.max(pg_loss1, pg_loss2)

    loss = pg_loss.mean()

    with torch.no_grad():
        mean_seq_ratio = seq_ratio.mean()
        clipped = ((seq_ratio < 1.0 - epsilon_lower) | (seq_ratio > 1.0 + epsilon_upper))
        clip_frac = clipped.float().mean()

    stats = {
        "loss/total": loss.item(),
        "loss/pg_loss": loss.item(),
        "train/seq_importance_ratio": mean_seq_ratio.item(),
        "train/seq_clip_fraction": clip_frac.item(),
    }

    return loss, stats
