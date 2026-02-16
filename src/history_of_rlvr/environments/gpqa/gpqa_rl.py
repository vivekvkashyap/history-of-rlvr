import random
import re
from typing import Any, Dict, List

from datasets import load_dataset

from history_of_rlvr.environments.base import Environment, run


ANSWER_LETTERS = ["A", "B", "C", "D"]
_ANSWER_PATTERN = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)


class GPQA(Environment):
    name = "gpqa.gpqa_rl"

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

    def get_config_overrides(self) -> Dict[str, Any]:
        return {}

    def get_dataset(self) -> List[Dict[str, str]]:
        ds = load_dataset(
            self.dataset_name, self.dataset_config, split=self.split,
        )
        rng = random.Random(self.seed)
        data = []
        for item in ds:
            q = item["Question"]
            letters = ANSWER_LETTERS.copy()
            if self.shuffle_choices:
                rng.shuffle(letters)
            itos = {k: v for k, v in enumerate(letters)}
            ans = {
                itos[0]: item["Correct Answer"],
                itos[1]: item["Incorrect Answer 1"],
                itos[2]: item["Incorrect Answer 2"],
                itos[3]: item["Incorrect Answer 3"],
            }
            question = f"Question: {q}\n\n"
            question += f"A: {ans['A']}\n"
            question += f"B: {ans['B']}\n"
            question += f"C: {ans['C']}\n"
            question += f"D: {ans['D']}"
            data.append({
                "prompt": self.format_prompt(question),
                "ground_truth": itos[0],
                "question": q,
            })
        return data

    def compute_rewards(
        self,
        completions: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        assert len(completions) == len(ground_truths), (
            f"Length mismatch: {len(completions)} completions vs "
            f"{len(ground_truths)} ground truths"
        )
        return [
            self._compute_single_reward(comp, gt)
            for comp, gt in zip(completions, ground_truths)
        ]

    def get_eval_dataset(self) -> List[Dict[str, str]]:
        ds = load_dataset(
            self.dataset_name, self.dataset_config, split="test",
        )
        rng = random.Random(self.seed)
        data = []
        for item in ds:
            q = item["Question"]
            letters = ANSWER_LETTERS.copy()
            if self.shuffle_choices:
                rng.shuffle(letters)
            itos = {k: v for k, v in enumerate(letters)}
            ans = {
                itos[0]: item["Correct Answer"],
                itos[1]: item["Incorrect Answer 1"],
                itos[2]: item["Incorrect Answer 2"],
                itos[3]: item["Incorrect Answer 3"],
            }
            question = f"Question: {q}\n\n"
            question += f"A: {ans['A']}\n"
            question += f"B: {ans['B']}\n"
            question += f"C: {ans['C']}\n"
            question += f"D: {ans['D']}"
            data.append({
                "prompt": self.format_prompt(question),
                "ground_truth": itos[0],
                "question": q,
            })
        return data

    @classmethod
    def compute_reward_details(
        cls, completion: str, ground_truth: str,
    ) -> Dict[str, Any]:
        extracted = cls._extract_answer_letter(completion)
        is_correct = extracted == ground_truth.upper()
        total = 1.0 if is_correct else 0.0
        return {
            "correctness": total,
            "total": total,
            "extracted_answer": extracted,
        }

    @staticmethod
    def _extract_answer_letter(completion: str) -> str:
        match = _ANSWER_PATTERN.search(completion)
        if match:
            content = match.group(1).strip().upper()
            letter_match = re.search(r"([A-D])", content)
            if letter_match:
                return letter_match.group(1)
        matches = re.findall(r"\b([A-Da-d])\b", completion)
        valid = [m.upper() for m in matches if m.upper() in ANSWER_LETTERS]
        if valid:
            return valid[-1]
        return ""

    @classmethod
    def _compute_single_reward(cls, completion: str, ground_truth: str) -> float:
        predicted = cls._extract_answer_letter(completion)
        return 1.0 if predicted == ground_truth.upper() else 0.0


if __name__ == "__main__":
    run(GPQA())
