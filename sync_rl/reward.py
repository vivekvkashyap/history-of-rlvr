"""
Rule-based reward function for math (GSM8K) answer verification.
"""

import re
from typing import List


def extract_answer_from_completion(completion: str) -> str:
    """
    Extract the final numeric answer from a model completion.

    Tries multiple heuristics:
      1. Look for '#### <number>'  (trained format)
      2. Look for 'the answer is <number>'
      3. Fall back to the last number in the text
    """
    # Strategy 1: explicit #### marker
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", completion)
    if match:
        return match.group(1).replace(",", "").strip()

    # Strategy 2: "the answer is ..."
    match = re.search(
        r"(?:the\s+)?answer\s+is\s*[:\s]*([+-]?[\d,]+\.?\d*)",
        completion,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(",", "").strip()

    # Strategy 3: last number in text
    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", completion)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return ""


def compute_reward(completion: str, ground_truth: str) -> float:
    """
    Binary reward: 1.0 if the extracted answer matches ground truth, else 0.0.
    """
    predicted = extract_answer_from_completion(completion)

    # Normalize: strip leading zeros, trailing '.0', etc.
    def _normalize(s: str) -> str:
        s = s.strip()
        if not s:
            return ""
        try:
            val = float(s)
            # Return as int string if integer-valued
            if val == int(val):
                return str(int(val))
            return str(val)
        except ValueError:
            return s

    pred_norm = _normalize(predicted)
    gt_norm = _normalize(ground_truth)

    return 1.0 if pred_norm == gt_norm else 0.0


def compute_rewards_batch(
    completions: List[str],
    ground_truths: List[str],
) -> List[float]:
    """
    Compute rewards for a batch of completions.

    Args:
        completions: list of model-generated completions
        ground_truths: list of ground-truth answers (one per prompt,
                       repeated G times to match completions)

    Returns:
        list of float rewards (same length as completions)
    """
    assert len(completions) == len(ground_truths), (
        f"Length mismatch: {len(completions)} completions vs "
        f"{len(ground_truths)} ground truths"
    )
    return [
        compute_reward(comp, gt)
        for comp, gt in zip(completions, ground_truths)
    ]
