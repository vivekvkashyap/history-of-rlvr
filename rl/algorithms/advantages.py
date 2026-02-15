"""Shared advantage computation for all RL algorithms."""

import torch


def compute_group_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    normalize_by_std: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Group-relative advantages: A_i = r_i - mean(r_group), optionally
    normalized by std.

    Args:
        rewards: (B * G,) flat tensor of rewards.
        num_generations: G, number of completions per prompt.
        normalize_by_std: if True, divide by group std.
        eps: numerical stability constant for std normalization.

    Returns:
        advantages: (B * G,) tensor.
    """
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    advantages = grouped - mean
    if normalize_by_std:
        std = grouped.std(dim=1, keepdim=True)
        advantages = advantages / (std + eps)
    return advantages.view(-1)
