"""
GSM8K Environment – SFT variant.

Supervised fine-tuning on GSM8K grade-school math word problems.
Uses the full chain-of-thought answer (not just the numeric result)
as the training completion.

Usage:
    cd history_of_rlvr

    # Basic SFT
    python -m environments.gsm8k_sft --model_name Qwen/Qwen2.5-0.5B-Instruct

    # With LoRA
    python -m environments.gsm8k_sft \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --use_lora true --lora_rank 16

    # Multi-GPU with accelerate
    accelerate launch -m environments.gsm8k_sft \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --per_device_train_batch_size 8
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

# Ensure root is on path for cross-package imports
_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from environments.base import Environment


# ════════════════════════════════════════════════════════════════════════
#  GSM8K SFT Environment
# ════════════════════════════════════════════════════════════════════════


class GSM8K_SFT(Environment):
    """
    GSM8K SFT: supervised fine-tuning on grade-school math word problems.

    Unlike the RL variant which only extracts the numeric answer as the
    ground truth, this SFT variant uses the **full chain-of-thought
    solution** from the dataset as the training completion.

    Dataset: ``openai/gsm8k`` on HuggingFace Hub.
    """

    name = "gsm8k_sft"

    system_prompt = (
        "You are a helpful math assistant. Solve the following math "
        "problem step by step. At the end, provide the final numeric "
        "answer after '#### '."
    )

    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "openai/gsm8k",
        dataset_config: str = "main",
    ):
        super().__init__()
        self.split = split
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

    # ── Dataset ────────────────────────────────────────────────────────

    def get_dataset(self) -> List[Dict[str, str]]:
        """
        Load GSM8K and format into prompt + completion pairs.

        The completion is the **full** chain-of-thought answer from the
        dataset (e.g. "Janet's ducks lay 16 eggs per day...\\n#### 36"),
        not just the numeric ground truth.
        """
        ds = load_dataset(
            self.dataset_name, self.dataset_config, split=self.split,
        )
        data = []
        for item in ds:
            question = item["question"]
            answer = item["answer"]  # full chain-of-thought answer

            data.append({
                "prompt": self.format_prompt(question),
                "completion": answer,
                "ground_truth": self._extract_ground_truth(answer),
                "question": question,
            })
        return data

    # ── Reward function (not used for SFT, satisfies abstract base) ───

    def compute_rewards(
        self,
        completions: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        """Not used during SFT training.  Provided to satisfy the abstract base class."""
        return [0.0] * len(completions)

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_ground_truth(answer_str: str) -> str:
        """Extract the final numeric answer from a GSM8K answer string."""
        match = re.search(r"####\s*(.+)", answer_str)
        if match:
            return match.group(1).strip().replace(",", "")
        return ""


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sft.main import run_sft
    run_sft(GSM8K_SFT())
