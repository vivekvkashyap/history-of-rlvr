"""Prime-RL style masked importance sampling with stop-gradient.
Reference: Prime Intellect (https://github.com/PrimeIntellect-ai/prime-rl)
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
    loss_reduction: str = "token",
) -> Tuple[torch.Tensor, dict]:
    """
    Prime-RL loss: tokens with ratio outside [low, high] are masked out.
    Remaining tokens get weighted MLE loss with stop-gradient on coefficient.
    """
    log_ratio = trainer_log_probs - inference_log_probs
    log_ratio = torch.clamp(log_ratio, min=-max_log_ratio, max=max_log_ratio)
    ratio = torch.exp(log_ratio)

    is_masked = (ratio < token_mask_low) | (ratio > token_mask_high)
    keep_mask = completion_mask.bool() & ~is_masked

    adv = advantages.unsqueeze(1)

    coeff = (ratio * adv).detach()
    per_token_loss = -(coeff * trainer_log_probs)

    # Apply keep_mask, then reduce
    masked_loss = per_token_loss * keep_mask.float()
    if loss_reduction == "sample":
        per_sample_tokens = completion_mask.sum(dim=1).clamp(min=1)
        per_sample_loss = masked_loss.sum(dim=1) / per_sample_tokens
        loss = per_sample_loss.mean()
    else:
        num_valid_tokens = completion_mask.sum()
        loss = masked_loss.sum() / (num_valid_tokens + 1e-8)

    with torch.no_grad():
        token_mismatch_kl = ratio - log_ratio - 1.0

    num_valid_tokens = completion_mask.sum()
    with torch.no_grad():
        masked_pg = masked_loss.sum() / (num_valid_tokens + 1e-8)
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
