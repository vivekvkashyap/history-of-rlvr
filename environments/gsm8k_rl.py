"""
GSM8K Environment – RL variant (Grade-school math word problems).

Single-file environment for GRPO training on GSM8K.  Defines the dataset
loader, prompt format, and multi-component reward function inspired by
Will Brown's GRPO recipe.

Reward components (summed per completion):
  - correctness:   2.0 if extracted answer matches ground truth, 0.0 otherwise
  - int_check:     0.5 if extracted answer is a pure integer, 0.0 otherwise
  - strict_format: 0.5 if output strictly follows the XML template, 0.0 otherwise
  - soft_format:   0.5 if output loosely matches the XML template, 0.0 otherwise
  - xml_count:     up to ~0.5 for correct XML tag usage (partial credit)

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
    Reward: multi-component (correctness + int + strict/soft format + XML count).
    """

    name = "gsm8k"

    system_prompt = (
        "Respond in the following format:\n"
        "<reasoning>\n"
        "...\n"
        "</reasoning>\n"
        "<answer>\n"
        "...\n"
        "</answer>"
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
        """
        Multi-component reward (summed per completion):

          correctness:   2.0 if answer matches, 0.0 otherwise
          int_check:     0.5 if extracted answer is a pure integer
          strict_format: 0.5 if strict XML template match
          soft_format:   0.5 if loose XML template match
          xml_count:     up to ~0.5 for correct XML tag presence
        """
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
        if "####" not in answer_str:
            return ""
        return answer_str.split("####")[1].strip().replace(",", "").replace("$", "")

    @staticmethod
    def _extract_xml_answer(completion: str) -> str:
        """Extract the content between <answer>...</answer> tags."""
        if "<answer>" not in completion:
            return ""
        answer = completion.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    # ── Individual reward components ───────────────────────────────────

    @staticmethod
    def _correctness_reward(extracted: str, ground_truth: str) -> float:
        """2.0 if extracted answer matches ground truth, 0.0 otherwise."""
        return 2.0 if extracted == ground_truth else 0.0

    @staticmethod
    def _int_reward(extracted: str) -> float:
        """0.5 if the extracted answer is a pure integer string."""
        return 0.5 if extracted.isdigit() else 0.0

    @staticmethod
    def _strict_format_reward(completion: str) -> float:
        """0.5 if completion strictly follows the XML template with newlines."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        return 0.5 if re.match(pattern, completion, flags=re.DOTALL) else 0.0

    @staticmethod
    def _soft_format_reward(completion: str) -> float:
        """0.5 if completion loosely matches <reasoning>...<answer>... tags."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        return 0.5 if re.match(pattern, completion, flags=re.DOTALL) else 0.0

    @staticmethod
    def _xml_count_reward(completion: str) -> float:
        """
        Partial-credit reward for correct XML tag usage (up to ~0.5).

        Awards 0.125 per correct tag occurrence, with a small penalty for
        trailing content after </answer>.
        """
        count = 0.0
        if completion.count("<reasoning>\n") == 1:
            count += 0.125
        if completion.count("\n</reasoning>\n") == 1:
            count += 0.125
        if completion.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(completion.split("\n</answer>\n")[-1]) * 0.001
        if completion.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(completion.split("\n</answer>")[-1]) - 1) * 0.001
        return count

    @classmethod
    def _compute_single_reward(cls, completion: str, ground_truth: str) -> float:
        """
        Sum of all reward components for a single completion.

        Components:
          - correctness:   2.0 / 0.0
          - int_check:     0.5 / 0.0
          - strict_format: 0.5 / 0.0
          - soft_format:   0.5 / 0.0
          - xml_count:     up to ~0.5
        """
        extracted = cls._extract_xml_answer(completion)

        reward = 0.0
        reward += cls._correctness_reward(extracted, ground_truth)
        reward += cls._int_reward(extracted)
        reward += cls._strict_format_reward(completion)
        reward += cls._soft_format_reward(completion)
        reward += cls._xml_count_reward(completion)
        return reward


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run(GSM8K())
