import re
import sys
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

from environments.base import Environment


class GSM8K_SFT(Environment):
    name = "gsm8k.gsm8k_sft"

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
        super().__init__()
        self.split = split
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

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
                "completion": answer,
                "ground_truth": self._extract_ground_truth(answer),
                "question": question,
            })
        return data

    def compute_rewards(
        self,
        completions: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        return [0.0] * len(completions)

    @staticmethod
    def _extract_ground_truth(answer_str: str) -> str:
        match = re.search(r"####\s*(.+)", answer_str)
        if match:
            return match.group(1).strip().replace(",", "")
        return ""


if __name__ == "__main__":
    from history_of_rlvr.sft.main import run_sft
    run_sft(GSM8K_SFT())
