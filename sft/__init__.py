"""
Supervised Fine-Tuning (SFT) utilities.

Provides:
  - ``SFTConfig``: training configuration (extends ``TrainingArguments``).
  - ``SFTDataset``: dataset wrapper for environment data.
  - ``SFTDataCollator``: tokenises prompt + completion with masked labels.
  - ``SFTTrainer``: trainer with PEFT support and eval metrics.
  - ``run_sft``: one-call launcher for environment-based SFT.

Usage:
    python -m environments.gsm8k.gsm8k_sft --model_name Qwen/Qwen2.5-0.5B-Instruct
"""

from sft.config import SFTConfig
from sft.data import SFTDataCollator, SFTDataset
from sft.main import run_sft
from sft.trainer import SFTTrainer

__all__ = [
    "SFTConfig",
    "SFTDataCollator",
    "SFTDataset",
    "SFTTrainer",
    "run_sft",
]
