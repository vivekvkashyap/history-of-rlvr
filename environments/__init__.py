"""
Environment-based training runner.

Each environment is a single file that defines a dataset, reward function,
and system prompt.  The ``run()`` helper wires it into the async GRPO
training backend automatically.

Usage:
    python -m environments.gsm8k --launch        # full tmux launch
    python -m environments.gsm8k [train args]    # training only
"""

from environments.base import Environment, EnvironmentDataset, run

__all__ = ["Environment", "EnvironmentDataset", "run"]
