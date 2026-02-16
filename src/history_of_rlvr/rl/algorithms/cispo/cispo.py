"""CISPO (Clipped IS-weight Policy Optimization).
Reference: MiniMax-M1 (https://arxiv.org/abs/2506.13585)
"""

import torch
from typing import Tuple


def _reduce_loss(per_token_loss, completion_mask, reduction):
    """Apply loss reduction: 'token' or 'sample'."""
    if reduction == "sample":
        per_sample_tokens = completion_mask.sum(dim=1).clamp(min=1)
        per_sample_loss = (per_token_loss * completion_mask).sum(dim=1) / per_sample_tokens
        return per_sample_loss.mean()
    else:
        num_valid = completion_mask.sum()
        return (per_token_loss * completion_mask).sum() / (num_valid + 1e-8)


def cispo_loss(
    trainer_log_probs: torch.Tensor,
    inference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon_lower: float = 10.0,
    epsilon_upper: float = 0.2,
    max_log_ratio: float = 10.0,
    loss_reduction: str = "token",
) -> Tuple[torch.Tensor, dict]:
    """
    CISPO loss: clips the IS weight directly with stop-gradient.

    loss = -sg(clip(r, 1-eps_low, 1+eps_high)) * A * log pi_theta
    Gradients flow only through log pi_theta, not through the ratio.
    """
    log_ratio = trainer_log_probs - inference_log_probs
    log_ratio = torch.clamp(log_ratio, min=-max_log_ratio, max=max_log_ratio)
    ratio = torch.exp(log_ratio)

    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_lower, 1.0 + epsilon_upper)

    adv = advantages.unsqueeze(1)

    per_token_loss = -(clipped_ratio.detach() * adv * trainer_log_probs)

    loss = _reduce_loss(per_token_loss, completion_mask, loss_reduction)

    num_valid_tokens = completion_mask.sum()
    with torch.no_grad():
        masked_pg = (per_token_loss * completion_mask).sum() / (num_valid_tokens + 1e-8)
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
