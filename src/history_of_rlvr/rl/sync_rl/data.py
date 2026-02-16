"""
GSM8K dataset loading and prompt formatting for GRPO training.
"""

import re
from typing import List, Dict, Any

from datasets import load_dataset
from torch.utils.data import Dataset


SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following math problem step by step. "
    "At the end, provide the final numeric answer after '#### '."
)


def extract_ground_truth(answer_str: str) -> str:
    """
    Extract the final numeric answer from a GSM8K answer string.
    GSM8K answers end with '#### <number>'.
    """
    match = re.search(r"####\s*(.+)", answer_str)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


def format_prompt(question: str) -> str:
    """
    Format a GSM8K question into a chat-style prompt.
    """
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


class GSM8KDataset(Dataset):
    """
    Wraps the GSM8K HuggingFace dataset for GRPO training.

    Each item returns:
        - prompt: formatted chat prompt string
        - ground_truth: the expected numeric answer as a string
        - question: the raw question text
    """

    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "openai/gsm8k",
        dataset_config: str = "main",
    ):
        self.dataset = load_dataset(dataset_name, dataset_config, split=split)
        self.prompts: List[str] = []
        self.ground_truths: List[str] = []
        self.questions: List[str] = []

        for item in self.dataset:
            question = item["question"]
            answer = item["answer"]
            self.prompts.append(format_prompt(question))
            self.ground_truths.append(extract_ground_truth(answer))
            self.questions.append(question)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "prompt": self.prompts[idx],
            "ground_truth": self.ground_truths[idx],
            "question": self.questions[idx],
        }
