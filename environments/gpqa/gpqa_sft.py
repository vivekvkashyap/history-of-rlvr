import random
from typing import Dict, List

from datasets import load_dataset

from environments.base import Environment


ANSWER_LETTERS = ["A", "B", "C", "D"]


class GPQA_SFT(Environment):
    name = "gpqa.gpqa_sft"

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

    def get_dataset(self) -> List[Dict[str, str]]:
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
            choices = [(correct, True)] + [(d, False) for d in distractors]
            if self.shuffle_choices:
                rng.shuffle(choices)
            correct_letter = ""
            choice_lines = []
            for i, (text, is_correct) in enumerate(choices):
                letter = ANSWER_LETTERS[i]
                choice_lines.append(f"({letter}) {text}")
                if is_correct:
                    correct_letter = letter
            question_with_choices = f"{question}\n\n" + "\n".join(choice_lines)
            data.append({
                "prompt": self.format_prompt(question_with_choices),
                "completion": f"The answer is ({correct_letter})",
                "ground_truth": correct_letter,
                "question": question,
            })
        return data

    def compute_rewards(
        self,
        completions: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        return [0.0] * len(completions)


if __name__ == "__main__":
    from sft.main import run_sft
    run_sft(GPQA_SFT())
