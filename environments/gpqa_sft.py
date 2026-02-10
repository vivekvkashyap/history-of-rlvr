"""
GPQA Environment – SFT variant.

Supervised fine-tuning on GPQA graduate-level science multiple-choice
questions.  Since the GPQA dataset only provides the correct answer
(not chain-of-thought reasoning), the completion is the correct answer
letter (A/B/C/D).

Usage:
    cd history_of_rlvr

    # Basic SFT
    python -m environments.gpqa_sft --model_name Qwen/Qwen2.5-0.5B-Instruct

    # With LoRA
    python -m environments.gpqa_sft \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --use_lora true --lora_rank 16

    # Multi-GPU with accelerate
    accelerate launch -m environments.gpqa_sft \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --per_device_train_batch_size 8
"""

import random
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
#  GPQA SFT Environment
# ════════════════════════════════════════════════════════════════════════

ANSWER_LETTERS = ["A", "B", "C", "D"]


class GPQA_SFT(Environment):
    """
    GPQA SFT: supervised fine-tuning on graduate-level science questions.

    The GPQA dataset provides only the correct answer, not chain-of-thought
    reasoning.  The completion is formatted as:
        "The answer is (X)"
    where X is the correct letter (A/B/C/D).

    Dataset: ``Idavidrein/gpqa`` (config ``gpqa_diamond``, 198 questions).
    """

    name = "gpqa_sft"

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
        self.split = split
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.shuffle_choices = shuffle_choices
        self.seed = seed

    # ── Dataset ────────────────────────────────────────────────────────

    def get_dataset(self) -> List[Dict[str, str]]:
        """
        Load GPQA and format into prompt + completion pairs.

        Each question has 1 correct + 3 incorrect answers.  The choices
        are optionally shuffled, assigned A/B/C/D labels, and the correct
        letter is recorded.  The completion is ``"The answer is (X)"``.
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
                "completion": f"The answer is ({correct_letter})",
                "ground_truth": correct_letter,
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


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sft.main import run_sft
    run_sft(GPQA_SFT())
