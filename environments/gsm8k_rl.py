"""
GSM8K Environment – RL variant (Grade-school math word problems).

Single-file environment for GRPO training on GSM8K. Defines the dataset
loader, prompt format, and binary reward function.

Reward: Binary correctness only. Extract answer from \\boxed{}, compare to ground truth.
  1.0 if correct, 0.0 if wrong.

Usage:
    cd history_of_rlvr

    # Full tmux launch (starts vLLM server + training)
    python -m environments.gsm8k_rl --launch

    # Training only (vLLM server must already be running)
    python -m environments.gsm8k_rl --model_name Qwen/Qwen2.5-1.5B-Instruct

    # With LoRA
    python -m environments.gsm8k_rl --launch --use_lora true --lora_rank 16
"""

import re
import logging
from typing import Any, Dict, List

from datasets import load_dataset
from math_verify import verify, parse

from environments.base import Environment, run

logger = logging.getLogger(__name__)

# Fallback regex (only used if math_verify fails to parse)
_ANSWER_PATTERN = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)


# ════════════════════════════════════════════════════════════════════════
#  GSM8K Environment
# ════════════════════════════════════════════════════════════════════════


class GSM8K(Environment):
    """
    GSM8K: 8.5K grade-school math word problems.

    Dataset: ``openai/gsm8k`` on HuggingFace Hub.
    Reward: Binary correctness (1.0 or 0.0) for answer from \\boxed{}.
    """

    name = "gsm8k_rl"

    system_prompt = (
        "You are a careful math tutor. Think step-by-step. When you are ready, "
        "write the final numeric answer inside \\boxed{}, like \\boxed{42}."
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

    # ── Config overrides ──────────────────────────────────────────────

    def get_config_overrides(self) -> Dict[str, Any]:
        """
        GSM8K-specific training config defaults.

        These are applied automatically so you don't need to pass them
        via CLI every time.  CLI arguments still take precedence.
        """
        return {
            # Customize these defaults for GSM8K training:
            # "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            # "learning_rate": 1e-5,
            # "max_steps": 500,
            # "num_generations": 16,
            # "batch_size": 512,
            # "max_new_tokens": 1024,
            # "temperature": 0.7,
        }

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
        """
        Reward: Binary correctness (1.0 or 0.0) from \\boxed{...}.
        """
        assert len(completions) == len(ground_truths), (
            f"Length mismatch: {len(completions)} completions vs "
            f"{len(ground_truths)} ground truths"
        )
        return [
            self._compute_single_reward(comp, gt)
            for comp, gt in zip(completions, ground_truths)
        ]

    # ── Evaluation ──────────────────────────────────────────────────────

    def get_eval_dataset(self) -> List[Dict[str, str]]:
        """Load GSM8K test split for evaluation."""
        ds = load_dataset(
            self.dataset_name, self.dataset_config, split="test",
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

    @classmethod
    def compute_reward_details(
        cls, completion: str, ground_truth: str,
    ) -> Dict[str, Any]:
        """
        Binary correctness: 1.0 if correct, 0.0 if wrong.
        Uses math_verify for robust verification with regex fallback.
        """
        total = cls._compute_single_reward(completion, ground_truth)
        extracted = cls._extract_answer_from_boxed(completion)

        return {
            "correctness": total,
            "total": total,
            "extracted_answer": extracted,
        }

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_ground_truth(answer_str: str) -> str:
        """Extract the final numeric answer from a GSM8K answer string."""
        if "####" not in answer_str:
            return ""
        return answer_str.split("####")[1].strip().replace(",", "").replace("$", "")

    @staticmethod
    def _extract_answer_from_boxed(completion: str) -> str:
        """Extract answer from \\boxed{...} per prompt. No fallback."""
        match = _ANSWER_PATTERN.search(completion)
        if match:
            return match.group(1).strip().replace(",", "").replace("$", "")
        return ""

    # ── Reward computation ────────────────────────────────────────────

    @staticmethod
    def _normalize_number(s: str) -> str:
        """Normalize a numeric string for comparison (e.g. '72.0' → '72')."""
        s = s.strip().replace(",", "").replace("$", "")
        if not s:
            return ""
        try:
            val = float(s)
            if not (val == val) or abs(val) == float('inf'):
                return s
            if val == int(val):
                return str(int(val))
            return str(val)
        except (ValueError, OverflowError):
            return s

    @classmethod
    def _compute_single_reward(cls, completion: str, ground_truth: str) -> float:
        """
        Binary correctness reward: 1.0 if correct, 0.0 if wrong.

        Uses math_verify for robust answer verification (handles LaTeX,
        equivalent expressions, nested braces, etc.). Falls back to simple
        regex + string comparison if math_verify cannot parse.
        """
        try:
            # math_verify: parse the model completion and the ground truth,
            # then verify equivalence. This handles \boxed{}, LaTeX expressions,
            # nested braces, equivalent number formats, etc.
            parsed_completion = parse(completion)
            parsed_gt = parse(ground_truth)
            if verify(parsed_completion, parsed_gt):
                return 1.0
        except Exception:
            # math_verify failed to parse — fall back to regex extraction
            pass

        # Fallback: simple regex extraction + string comparison
        extracted = cls._extract_answer_from_boxed(completion)
        if extracted and cls._normalize_number(extracted) == cls._normalize_number(ground_truth):
            return 1.0

        return 0.0


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run(GSM8K())
