"""
GPQA Environment – Graduate-level science multiple-choice questions.

Single-file environment for GRPO training on GPQA (Diamond subset).
Defines the dataset loader, prompt format, and reward function.

Dataset: ``Idavidrein/gpqa`` on HuggingFace Hub (requires accepting terms).
         Covers expert-level questions in biology, physics, and chemistry.

Usage:
    cd history_of_rlvr

    # Full tmux launch (starts vLLM server + training)
    python -m environments.gpqa --launch

    # Training only (vLLM server must already be running)
    python -m environments.gpqa --model_name Qwen/Qwen2.5-0.5B-Instruct

    # With LoRA
    python -m environments.gpqa --launch --use_lora true --lora_rank 16
"""

import random
import re
from typing import Dict, List

from datasets import load_dataset

from environments.base import Environment, run


# ════════════════════════════════════════════════════════════════════════
#  GPQA Environment
# ════════════════════════════════════════════════════════════════════════


ANSWER_LETTERS = ["A", "B", "C", "D"]


class GPQA(Environment):
    """
    GPQA: Graduate-level Google-Proof Q&A Benchmark (Diamond subset).

    448 multiple-choice questions written by PhD-level domain experts
    in biology, physics, and chemistry.

    Dataset: ``Idavidrein/gpqa`` (config ``gpqa_diamond``, 198 questions).
    Reward: binary 1.0 if extracted answer letter matches correct answer.
    """

    name = "gpqa"

    system_prompt = (
        "You are an expert scientist. Answer the following multiple-choice "
        "question by reasoning step by step. At the end, state your final "
        "answer as a single letter (A, B, C, or D) after 'The answer is '."
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
            question = item["Question"]
            correct = item["Correct Answer"]
            distractors = [
                item["Incorrect Answer 1"],
                item["Incorrect Answer 2"],
                item["Incorrect Answer 3"],
            ]

            # Build choices list: [(text, is_correct), ...]
            choices = [(correct, True)] + [(d, False) for d in distractors]
            if self.shuffle_choices:
                rng.shuffle(choices)

            # Assign letters and find the correct one
            correct_letter = ""
            choice_lines = []
            for i, (text, is_correct) in enumerate(choices):
                letter = ANSWER_LETTERS[i]
                choice_lines.append(f"({letter}) {text}")
                if is_correct:
                    correct_letter = letter

            # Format question with choices
            question_with_choices = (
                f"{question}\n\n" + "\n".join(choice_lines)
            )

            data.append({
                "prompt": self.format_prompt(question_with_choices),
                "ground_truth": correct_letter,
                "question": question,
            })

        return data

    # ── Reward function ────────────────────────────────────────────────

    def compute_rewards(
        self,
        completions: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        """Binary reward: 1.0 if extracted answer letter matches ground truth."""
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
    def _extract_answer_letter(completion: str) -> str:
        """
        Extract the answer letter (A/B/C/D) from a model completion.

        Tries multiple heuristics:
          1. ``The answer is (X)`` or ``The answer is X``
          2. ``\\boxed{X}``
          3. Last standalone letter A/B/C/D in the text
        """
        # Strategy 1: "the answer is X" pattern
        match = re.search(
            r"(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?",
            completion,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()

        # Strategy 2: \boxed{X}
        match = re.search(r"\\boxed\{([A-Da-d])\}", completion)
        if match:
            return match.group(1).upper()

        # Strategy 3: last standalone letter A-D
        matches = re.findall(r"\b([A-Da-d])\b", completion)
        # Filter to only valid answer letters
        valid = [m.upper() for m in matches if m.upper() in ANSWER_LETTERS]
        if valid:
            return valid[-1]

        return ""

    @classmethod
    def _compute_single_reward(cls, completion: str, ground_truth: str) -> float:
        """Binary reward for a single completion."""
        predicted = cls._extract_answer_letter(completion)
        return 1.0 if predicted == ground_truth.upper() else 0.0


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run(GPQA())
