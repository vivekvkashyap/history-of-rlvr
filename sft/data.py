"""
Dataset and data collation utilities for SFT training.

Provides:
  - ``SFTDataset``: wraps environment ``get_dataset()`` output into a
    ``torch.utils.data.Dataset``.
  - ``SFTDataCollator``: tokenises prompt + completion pairs on the fly,
    masking prompt tokens in the labels so the loss is only computed on
    the completion portion.
"""

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# ════════════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════════════


class SFTDataset(Dataset):
    """
    Thin ``torch.utils.data.Dataset`` wrapper around the list returned by
    ``Environment.get_dataset()``.

    Each element is a dict with at least ``"prompt"`` and ``"completion"``
    keys (string values).
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        prompt_field: str = "prompt",
        completion_field: str = "completion",
    ):
        self.data = data
        self.prompt_field = prompt_field
        self.completion_field = completion_field

        # Validate that required fields are present
        if len(data) > 0:
            first = data[0]
            if self.prompt_field not in first:
                raise ValueError(
                    f"Dataset entries must contain '{self.prompt_field}' key. "
                    f"Found keys: {list(first.keys())}"
                )
            if self.completion_field not in first:
                raise ValueError(
                    f"Dataset entries must contain '{self.completion_field}' key. "
                    f"Found keys: {list(first.keys())}"
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]


# ════════════════════════════════════════════════════════════════════════
#  Data Collator
# ════════════════════════════════════════════════════════════════════════


class SFTDataCollator:
    """
    Collate a batch of ``{"prompt": str, "completion": str}`` dicts into
    padded tensors suitable for causal LM training.

    Tokenisation strategy:
      1. Tokenise the prompt (with special tokens).
      2. Tokenise the completion (without special tokens) + append EOS.
      3. Concatenate to form the full sequence.
      4. Build labels: ``-100`` for prompt tokens, real token ids for
         completion tokens.  This ensures the loss is only computed on the
         completion portion.
      5. Right-pad all sequences to the longest in the batch.
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 1024,
        prompt_field: str = "prompt",
        completion_field: str = "completion",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.prompt_field = prompt_field
        self.completion_field = completion_field

        # Ensure pad token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ── Main collation ─────────────────────────────────────────────────

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Tokenise and collate a batch of prompt/completion pairs."""
        batch_input_ids: List[List[int]] = []
        batch_labels: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []

        for example in features:
            prompt_text = example[self.prompt_field]
            completion_text = example[self.completion_field]

            input_ids, labels = self._tokenize_pair(prompt_text, completion_text)
            attention_mask = [1] * len(input_ids)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)

        # Pad to the longest sequence in this batch (capped at max_seq_length)
        return self._pad_batch(batch_input_ids, batch_labels, batch_attention_mask)

    # ── Tokenise a single pair ─────────────────────────────────────────

    def _tokenize_pair(
        self,
        prompt: str,
        completion: str,
    ) -> tuple:
        """
        Tokenise a prompt + completion pair.

        Returns:
            (input_ids, labels) where labels has -100 for prompt tokens.
        """
        # Tokenise prompt – keep special tokens (e.g. BOS if the model uses one)
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=False,
        )

        # Tokenise completion – no special tokens, we append EOS manually
        completion_ids = self.tokenizer.encode(
            completion,
            add_special_tokens=False,
            truncation=False,
        )

        # Append EOS to completion
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            completion_ids = completion_ids + [eos_id]

        # Concatenate
        input_ids = prompt_ids + completion_ids

        # Truncate to max_seq_length
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]
            # Recompute split point
            prompt_len = min(len(prompt_ids), self.max_seq_length)
        else:
            prompt_len = len(prompt_ids)

        # Build labels: -100 for prompt, real ids for completion
        labels = [self.IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]

        assert len(input_ids) == len(labels), (
            f"Length mismatch: input_ids={len(input_ids)}, labels={len(labels)}"
        )

        return input_ids, labels

    # ── Pad a batch ────────────────────────────────────────────────────

    def _pad_batch(
        self,
        batch_input_ids: List[List[int]],
        batch_labels: List[List[int]],
        batch_attention_mask: List[List[int]],
    ) -> Dict[str, torch.Tensor]:
        """Right-pad all sequences to the longest in the batch."""
        max_len = min(
            max(len(ids) for ids in batch_input_ids),
            self.max_seq_length,
        )

        pad_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, lbls, mask in zip(batch_input_ids, batch_labels, batch_attention_mask):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(lbls + [self.IGNORE_INDEX] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }
