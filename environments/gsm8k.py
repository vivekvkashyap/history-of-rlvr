"""
GSM8K Environment – Grade-school math word problems.

Single-file environment for GRPO training on GSM8K.  Defines the dataset
loader, prompt format, and reward function.

Usage:
    cd history_of_rlvr

    # Full tmux launch (starts vLLM server + training)
    python -m environments.gsm8k --launch

    # Training only (vLLM server must already be running)
    python -m environments.gsm8k --model_name Qwen/Qwen2.5-0.5B-Instruct

    # With LoRA
    python -m environments.gsm8k --launch --use_lora true --lora_rank 16
"""

import re
from typing import Dict, List

from datasets import load_dataset

from environments.base import Environment, run


# ════════════════════════════════════════════════════════════════════════
#  GSM8K Environment
# ════════════════════════════════════════════════════════════════════════


class GSM8K(Environment):
    """
    GSM8K: 8.5K grade-school math word problems.

    Dataset: ``openai/gsm8k`` on HuggingFace Hub.
    Reward: binary 1.0 if extracted numeric answer matches ground truth.
    """

    name = "gsm8k"

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
        self.split = split
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

    # ── Dataset ────────────────────────────────────────────────────────

    def get_dataset(self) -> List[Dict[str, str]]:
        """Load GSM8K and format into prompt + ground_truth pairs."""
        ds = load_dataset(
            self.dataset_name, self.dataset_config, split=self.split,
        )
        data = []
        for item in ds:
            question = item["question"]
            answer = item["answer"]
            data.append({
                "prompt": self.format_prompt(question),
                "ground_truth": self._extract_ground_truth(answer),
                "question": question,
            })
        return data

    # ── Reward function ────────────────────────────────────────────────

    def compute_rewards(
        self,
        completions: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        """Binary reward: 1.0 if extracted answer matches ground truth."""
        assert len(completions) == len(ground_truths), (
            f"Length mismatch: {len(completions)} completions vs "
            f"{len(ground_truths)} ground truths"
        )
        return [
            self._compute_single_reward(comp, gt)
            for comp, gt in zip(completions, ground_truths)
        ]

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_ground_truth(answer_str: str) -> str:
        """Extract the final numeric answer from a GSM8K answer string."""
        match = re.search(r"####\s*(.+)", answer_str)
        if match:
            return match.group(1).strip().replace(",", "")
        return ""

    @staticmethod
    def _extract_answer_from_completion(completion: str) -> str:
        """
        Extract the predicted answer from a model completion.

        Tries multiple heuristics:
          1. ``#### <number>``  (trained format)
          2. ``the answer is <number>``
          3. Last number in the text
        """
        # Strategy 1: explicit #### marker
        match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", completion)
        if match:
            return match.group(1).replace(",", "").strip()

        # Strategy 2: "the answer is ..."
        match = re.search(
            r"(?:the\s+)?answer\s+is\s*[:\s]*([+-]?[\d,]+\.?\d*)",
            completion,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).replace(",", "").strip()

        # Strategy 3: last number in text
        numbers = re.findall(r"[+-]?[\d,]+\.?\d*", completion)
        if numbers:
            return numbers[-1].replace(",", "").strip()

        return ""

    @staticmethod
    def _normalize_number(s: str) -> str:
        """Normalize a numeric string for comparison."""
        s = s.strip()
        if not s:
            return ""
        try:
            val = float(s)
            if val == int(val):
                return str(int(val))
            return str(val)
        except ValueError:
            return s

    @classmethod
    def _compute_single_reward(cls, completion: str, ground_truth: str) -> float:
        """Binary reward for a single completion."""
        predicted = cls._extract_answer_from_completion(completion)
        pred_norm = cls._normalize_number(predicted)
        gt_norm = cls._normalize_number(ground_truth)
        return 1.0 if pred_norm == gt_norm else 0.0


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run(GSM8K())
