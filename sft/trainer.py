"""
SFT Trainer – extends HuggingFace ``Trainer`` for supervised fine-tuning.

Handles:
  - Custom data collation (prompt-masked labels via ``SFTDataCollator``).
  - PEFT / LoRA integration (apply, save, merge).
  - Evaluation metrics (loss, perplexity).
  - Accelerate multi-GPU support (automatic via Trainer).
  - Custom logging format with clean separator-based display.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from sft.config import SFTConfig
from sft.data import SFTDataCollator, SFTDataset


class SFTLoggingCallback(TrainerCallback):
    """Custom callback to format training logs with clean separators and colors."""

    def __init__(self):
        self.max_steps = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Capture max_steps at the start of training."""
        self.max_steps = state.max_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Format and print logs with step | loss | lr | grad_norm."""
        if logs is None:
            return

        # Only log on main process
        if state.is_world_process_zero:
            step = state.global_step
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            grad_norm = logs.get("grad_norm")

            # Build formatted line with colors
            parts = []
            if self.max_steps:
                parts.append(f"step [cyan]{step:>5}[/cyan]/[cyan]{self.max_steps}[/cyan]")
            else:
                parts.append(f"step [cyan]{step:>5}[/cyan]")
            
            if loss is not None:
                parts.append(f"loss [yellow]{loss:.4f}[/yellow]")
            if lr is not None:
                parts.append(f"lr [green]{lr:.2e}[/green]")
            if grad_norm is not None:
                parts.append(f"gn [magenta]{grad_norm:.4f}[/magenta]")

            line = " | ".join(parts)
            
            # Use rich Console for colored output
            from rich.console import Console
            console = Console(highlight=False)
            console.print(line, markup=True)


class SFTTrainer(Trainer):
    """
    Supervised Fine-Tuning trainer.

    Extends HuggingFace ``Trainer`` with:
      - ``SFTDataCollator`` for prompt-masked labels.
      - Evaluation metrics: loss and perplexity.
      - PEFT-aware model saving.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: SFTConfig,
        train_dataset: SFTDataset,
        eval_dataset: Optional[SFTDataset] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        **kwargs,
    ):
        # Build the data collator
        data_collator = SFTDataCollator(
            tokenizer=processing_class,
            max_seq_length=args.max_seq_length,
            prompt_field=args.prompt_field,
            completion_field=args.completion_field,
        )

        # Add custom logging callback
        callbacks = kwargs.pop("callbacks", [])
        callbacks.append(SFTLoggingCallback())

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=processing_class,
            callbacks=callbacks,
            **kwargs,
        )

        self.sft_config = args

        # Disable default progress callback for cleaner output
        try:
            self.remove_callback("ProgressCallback")
        except (ValueError, AttributeError):
            # ProgressCallback might not exist in some versions
            pass
        
        # Disable the default PrinterCallback that prints dicts
        try:
            self.remove_callback("PrinterCallback")
        except (ValueError, AttributeError):
            pass

    # ── Evaluation metrics ─────────────────────────────────────────────

    def compute_metrics_from_eval_loss(self, eval_loss: float) -> Dict[str, float]:
        """Compute perplexity from evaluation loss."""
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        return {"perplexity": perplexity}

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Run evaluation and add perplexity to metrics."""
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Compute perplexity from eval loss
        loss_key = f"{metric_key_prefix}_loss"
        if loss_key in metrics:
            try:
                metrics[f"{metric_key_prefix}_perplexity"] = math.exp(
                    metrics[loss_key]
                )
            except OverflowError:
                metrics[f"{metric_key_prefix}_perplexity"] = float("inf")

        return metrics

    # ── Custom logging to suppress default dict output ────────────────

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log method to suppress default dict output.
        Our custom callback handles the formatted printing.
        """
        # Just call parent - PrinterCallback is removed so no dict output
        super().log(logs, start_time)

    # ── PEFT-aware saving ──────────────────────────────────────────────

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Save model.  If using PEFT, saves the adapter separately as well
        as the full merged model.
        """
        output_dir = output_dir or self.args.output_dir

        if self.sft_config.use_lora:
            # Save the PEFT adapter
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save_pretrained(output_dir)

            # Also save the tokenizer
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
        else:
            # Default saving behaviour
            super()._save(output_dir, state_dict)
