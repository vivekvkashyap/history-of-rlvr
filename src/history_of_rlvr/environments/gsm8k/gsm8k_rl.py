import re
import logging
from typing import Any, Dict, List

from datasets import load_dataset
from math_verify import verify, parse

from history_of_rlvr.environments.base import Environment, run

logger = logging.getLogger(__name__)
_ANSWER_PATTERN = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)


class GSM8K(Environment):
    name = "gsm8k.gsm8k_rl"

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

    def get_config_overrides(self) -> Dict[str, Any]:
        return {}

    def get_dataset(self) -> List[Dict[str, str]]:
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
        total = cls._compute_single_reward(completion, ground_truth)
        extracted = cls._extract_answer_from_boxed(completion)
        return {
            "correctness": total,
            "total": total,
            "extracted_answer": extracted,
        }

    @staticmethod
    def _extract_ground_truth(answer_str: str) -> str:
        if "####" not in answer_str:
            return ""
        return answer_str.split("####")[1].strip().replace(",", "").replace("$", "")

    @staticmethod
    def _extract_answer_from_boxed(completion: str) -> str:
        match = _ANSWER_PATTERN.search(completion)
        if match:
            return match.group(1).strip().replace(",", "").replace("$", "")
        return ""

    @staticmethod
    def _normalize_number(s: str) -> str:
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
        try:
            parsed_completion = parse(completion)
            parsed_gt = parse(ground_truth)
            if verify(parsed_completion, parsed_gt):
                return 1.0
        except Exception:
            pass
        extracted = cls._extract_answer_from_boxed(completion)
        if extracted and cls._normalize_number(extracted) == cls._normalize_number(ground_truth):
            return 1.0
        return 0.0


if __name__ == "__main__":
    run(GSM8K())
