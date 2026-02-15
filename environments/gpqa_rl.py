"""
GPQA RL Environment – Graduate-level science multiple-choice questions.

Single-file environment for GRPO training on GPQA (Diamond subset).
Defines the dataset loader, prompt format, and binary reward function.

Reward: Binary correctness only. Extract answer letter from \\boxed{}, compare to ground truth.
  1.0 if correct, 0.0 if wrong.

Dataset: ``Idavidrein/gpqa`` on HuggingFace Hub (requires accepting terms).
         Covers expert-level questions in biology, physics, and chemistry.

Usage:
    cd history_of_rlvr

    # Full tmux launch (starts vLLM server + training)
    python -m environments.gpqa_rl --launch

    # Training only (vLLM server must already be running)
    python -m environments.gpqa_rl --model_name Qwen/Qwen2.5-0.5B-Instruct

    # With LoRA
    python -m environments.gpqa_rl --launch --use_lora true --lora_rank 16
"""

import random
import re
from typing import Any, Dict, List

from datasets import load_dataset

from environments.base import Environment, run


# ════════════════════════════════════════════════════════════════════════
#  GPQA RL Environment
# ════════════════════════════════════════════════════════════════════════


ANSWER_LETTERS = ["A", "B", "C", "D"]

# Pattern to match \boxed{...} in model completions
# Model outputs contain literal backslash chars, e.g., the text: \boxed{A}
# The raw string r"\\boxed\{([^}]*)\}" matches: \ + boxed + { + content + }
_ANSWER_PATTERN = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)


class GPQA(Environment):
    """
    GPQA: Graduate-level Google-Proof Q&A Benchmark (Diamond subset).

    448 multiple-choice questions written by PhD-level domain experts
    in biology, physics, and chemistry.

    Dataset: ``Idavidrein/gpqa`` (config ``gpqa_diamond``, 198 questions).
    Reward: binary 1.0 if extracted answer letter matches correct answer, 0.0 otherwise.
    """

    name = "gpqa_rl"

    system_prompt = (
        "Please reason step by step, then ONLY give the letter of the correct "
        "answer within \\boxed{}."
    )

    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "Idavidrein/gpqa",
        dataset_config: str = "gpqa_diamond",
        shuffle_choices: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.split = split
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.shuffle_choices = shuffle_choices
        self.seed = seed

    # ── Config overrides ──────────────────────────────────────────────

    def get_config_overrides(self) -> Dict[str, Any]:
        """
        GPQA-specific training config defaults.

        These are applied automatically so you don't need to pass them
        via CLI every time.  CLI arguments still take precedence.
        """
        return {
            # Customize these defaults for GPQA training:
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
        """
        Load GPQA and format into prompt + ground_truth pairs.

        Each question has 1 correct + 3 incorrect answers.  The choices
        are optionally shuffled, assigned A/B/C/D labels, and the correct
        letter is recorded as the ground truth.
        """
        ds = load_dataset(
            self.dataset_name, self.dataset_config, split=self.split,
        )
        rng = random.Random(self.seed)

        data = []
        for item in ds:
            q = item["Question"]
            
            # Shuffle the letter assignments
            letters = ANSWER_LETTERS.copy()
            if self.shuffle_choices:
                rng.shuffle(letters)
            
            # Create mapping: index -> letter
            itos = {k: v for k, v in enumerate(letters)}
            
            # Create answer mapping: letter -> answer text
            ans = {
                itos[0]: item["Correct Answer"],
                itos[1]: item["Incorrect Answer 1"],
                itos[2]: item["Incorrect Answer 2"],
                itos[3]: item["Incorrect Answer 3"],
            }
            
            # Format question
            question = f"Question: {q}\n\n"
            question += f"A: {ans['A']}\n"
            question += f"B: {ans['B']}\n"
            question += f"C: {ans['C']}\n"
            question += f"D: {ans['D']}"

            data.append({
                "prompt": self.format_prompt(question),
                "ground_truth": itos[0],  # The correct answer is at index 0
                "question": q,
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
        """Load GPQA test split for evaluation."""
        ds = load_dataset(
            self.dataset_name, self.dataset_config, split="test",
        )
        rng = random.Random(self.seed)

        data = []
        for item in ds:
            q = item["Question"]
            
            # Shuffle the letter assignments
            letters = ANSWER_LETTERS.copy()
            if self.shuffle_choices:
                rng.shuffle(letters)
            
            # Create mapping: index -> letter
            itos = {k: v for k, v in enumerate(letters)}
            
            # Create answer mapping: letter -> answer text
            ans = {
                itos[0]: item["Correct Answer"],
                itos[1]: item["Incorrect Answer 1"],
                itos[2]: item["Incorrect Answer 2"],
                itos[3]: item["Incorrect Answer 3"],
            }
            
            # Format question
            question = f"Question: {q}\n\n"
            question += f"A: {ans['A']}\n"
            question += f"B: {ans['B']}\n"
            question += f"C: {ans['C']}\n"
            question += f"D: {ans['D']}"

            data.append({
                "prompt": self.format_prompt(question),
                "ground_truth": itos[0],  # The correct answer is at index 0
                "question": q,
            })

        return data

    @classmethod
    def compute_reward_details(
        cls, completion: str, ground_truth: str,
    ) -> Dict[str, Any]:
        """
        Binary correctness: 1.0 if correct, 0.0 if wrong.
        Answer extracted from \\boxed{}, compared to ground truth.
        """
        extracted = cls._extract_answer_letter(completion)
        is_correct = extracted == ground_truth.upper()
        total = 1.0 if is_correct else 0.0

        return {
            "correctness": total,
            "total": total,
            "extracted_answer": extracted,
        }

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_answer_letter(completion: str) -> str:
        """
        Extract the answer letter (A/B/C/D) from a model completion.

        Primary method: Extract from \\boxed{X} as per system prompt.
        Fallback: Last standalone letter A/B/C/D in the text.
        """
        # Strategy 1: \boxed{X} - primary method as per system prompt
        match = _ANSWER_PATTERN.search(completion)
        if match:
            content = match.group(1).strip().upper()
            # Extract just the letter if present
            letter_match = re.search(r"([A-D])", content)
            if letter_match:
                return letter_match.group(1)

        # Fallback: last standalone letter A-D
        matches = re.findall(r"\b([A-Da-d])\b", completion)
        # Filter to only valid answer letters
        valid = [m.upper() for m in matches if m.upper() in ANSWER_LETTERS]
        if valid:
            return valid[-1]

        return ""

    @classmethod
    def _compute_single_reward(cls, completion: str, ground_truth: str) -> float:
        """
        Binary correctness reward: 1.0 if correct, 0.0 if wrong.
        Answer extracted from \\boxed{...} as per system prompt.
        """
        predicted = cls._extract_answer_letter(completion)
        return 1.0 if predicted == ground_truth.upper() else 0.0


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run(GPQA())
