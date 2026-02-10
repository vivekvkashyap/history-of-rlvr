"""
Rich-based evaluation display for async GRPO training.

Prints a colored table of sampled query/completion/reward examples
to the terminal at configurable intervals during training.

Based on the verifiers ``print_prompt_completions_sample`` pattern.
"""

from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _truncate(text: str, max_chars: int = 500) -> str:
    """Truncate long text with an ellipsis indicator."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " [...]"


def print_examples(
    prompts: List[str],
    completions: List[str],
    rewards: List[float],
    step: int,
    num_samples: int = 2,
    num_generations: int = 1,
) -> None:
    """
    Print a Rich table showing sampled training examples.

    Parameters
    ----------
    prompts : list[str]
        The query/prompt strings (one per unique prompt).
    completions : list[str]
        The model completions (may be G completions per prompt).
    rewards : list[float]
        Reward scores corresponding to each completion.
    step : int
        Current training step (shown in the panel title).
    num_samples : int
        How many examples to display (default 2).
    num_generations : int
        Number of completions per prompt (G). Used to pick the
        best completion per prompt for display.
    """
    if not prompts or not completions:
        return

    console = Console()

    table = Table(
        show_header=True,
        header_style="bold white",
        expand=True,
        show_lines=False,
    )

    table.add_column("Query", style="bright_yellow", ratio=2)
    table.add_column("Completion", style="bright_green", ratio=3)
    table.add_column("Reward", style="bold cyan", justify="right", width=8)

    samples_to_show = min(num_samples, len(prompts))

    for i in range(samples_to_show):
        prompt_text = prompts[i]

        # Pick the best completion for this prompt from its generation group
        if num_generations > 1 and len(completions) >= (i + 1) * num_generations:
            start = i * num_generations
            end = start + num_generations
            group_rewards = rewards[start:end]
            best_idx = start + group_rewards.index(max(group_rewards))
            completion_text = completions[best_idx]
            reward_val = rewards[best_idx]
        elif i < len(completions):
            completion_text = completions[i]
            reward_val = rewards[i] if i < len(rewards) else 0.0
        else:
            continue

        # Show full text without truncation
        prompt_display = prompt_text
        completion_display = completion_text

        # Color the reward based on value
        if reward_val >= 0.8:
            reward_style = "bold green"
        elif reward_val >= 0.3:
            reward_style = "bold yellow"
        else:
            reward_style = "bold red"

        table.add_row(
            Text(prompt_display),
            Text(completion_display),
            Text(f"{reward_val:.2f}", style=reward_style),
        )

        if i < samples_to_show - 1:
            table.add_section()

    panel = Panel(
        table,
        expand=True,
        title=f"[bold white]Step {step} — Evaluation Samples[/bold white]",
        border_style="bold white",
    )
    console.print(panel)
