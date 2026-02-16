"""GPQA: graduate-level science multiple-choice (RL + SFT variants)."""

from history_of_rlvr.environments.gpqa.gpqa_rl import GPQA
from history_of_rlvr.environments.gpqa.gpqa_sft import GPQA_SFT

__all__ = ["GPQA", "GPQA_SFT"]
