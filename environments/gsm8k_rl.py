"""
GSM8K Environment – RL variant (Grade-school math word problems).

Single-file environment for GRPO training on GSM8K. Defines the dataset
loader, prompt format, and binary reward function.

Reward: Matches system prompt – answer inside \\boxed{}.
  - Extract answer only from \\boxed{}. Binary correctness: 1.0 (correct) or 0.0 (wrong).
  - Format reward available via reward_coef (currently disabled with coef=0.0).

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
from typing import Any, Dict, List

from datasets import load_dataset

from environments.base import Environment, run

# Pattern to match \boxed{...} in model completions
# Model outputs contain literal backslash chars, e.g., the text: \boxed{42}
# The raw string r"\\boxed\{([^}]*)\}" matches: \ + boxed + { + content + }
# Note: In testing, use raw strings for test completions: r"\boxed{42}"
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
        Per-component: Binary correctness (1.0 or 0.0) from \\boxed{}.
        Format reward tracked but disabled via reward_coef=0.0.
        """
        extracted = cls._extract_answer_from_boxed(completion)
        is_correct = cls._normalize_number(extracted) == cls._normalize_number(ground_truth)
        has_format = cls._has_boxed_format(completion)

        correctness = 1.0 if is_correct else 0.0
        format_reward = 0.5 if has_format else -0.5
        reward_coef = 0.0
        total = correctness + (reward_coef * format_reward)

        return {
            "correctness": correctness,
            "format_reward": format_reward,
            "reward_coef": reward_coef,
            "total": total,
            "extracted_answer": extracted,
            "has_boxed_format": has_format,
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

    @staticmethod
    def _has_boxed_format(completion: str) -> bool:
        """True if completion has \\boxed{...} as instructed in prompt."""
        return _ANSWER_PATTERN.search(completion) is not None

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
        Answer extracted from \\boxed{...} as per system prompt.
        """
        extracted = cls._extract_answer_from_boxed(completion)
        is_correct = cls._normalize_number(extracted) == cls._normalize_number(ground_truth)
        has_format = cls._has_boxed_format(completion)

        correctness = 1.0 if is_correct else 0.0
        format_reward = 0.5 if has_format else -0.5
        reward_coef = 0.0
        return correctness + (reward_coef * format_reward)


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run(GSM8K())
