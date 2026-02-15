"""
SFT training entry point.

Provides ``run_sft(env)`` which wires an ``Environment`` into the SFT
training pipeline: loads model + tokenizer, builds the dataset from the
environment, applies optional LoRA, and runs training via ``SFTTrainer``.

Usage (from an environment file):
    from sft.main import run_sft
    run_sft(MyEnvironment())

Or directly:
    python -m environments.gsm8k.gsm8k_sft --model_name Qwen/Qwen2.5-0.5B-Instruct
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import torch
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from sft.config import SFTConfig
from sft.data import SFTDataset
from sft.trainer import SFTTrainer


def run_sft(env, cli_args: Optional[List[str]] = None) -> None:
    """
    Main entry point for SFT training with an environment.

    Args:
        env: An ``Environment`` instance whose ``get_dataset()`` returns
             dicts with ``"prompt"`` and ``"completion"`` keys.
        cli_args: Optional list of CLI arguments.  If ``None``, uses
                  ``sys.argv[1:]``.
    """
    # ── Ensure parent directory is on sys.path ────────────────────────
    root = str(Path(__file__).parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    # ── Suppress noisy logs ───────────────────────────────────────────
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for name in ("transformers", "accelerate", "peft", "datasets"):
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*use_cache=True.*")
    warnings.filterwarnings("ignore", message=".*PAD/BOS/EOS.*")

    console = Console(force_terminal=True, highlight=False)

    # ── Parse config ──────────────────────────────────────────────────
    args = cli_args if cli_args is not None else sys.argv[1:]
    old_argv = sys.argv
    sys.argv = ["sft_runner"] + args
    parser = HfArgumentParser(SFTConfig)
    (config,) = parser.parse_args_into_dataclasses()
    sys.argv = old_argv

    # Default output_dir to include environment name
    if config.output_dir == "outputs/sft":
        config.output_dir = f"outputs/{env.name}"

    console.print()
    console.rule(f"{env.name} – Supervised Fine-Tuning")
    console.print()
    console.print(f"  Environment     {env.name}")
    console.print(f"  Model           {config.model_name}")
    console.print(f"  Max seq length  {config.max_seq_length}")
    console.print(f"  Train           lr={config.learning_rate}  epochs={config.num_train_epochs}  batch/gpu={config.per_device_train_batch_size}")
    console.print(f"  Grad accum      {config.gradient_accumulation_steps}")
    console.print(f"  Output          {config.output_dir}")
    if config.use_lora:
        console.print(f"  LoRA            rank={config.lora_rank}  alpha={config.lora_alpha}  dropout={config.lora_dropout}")
    console.print()

    # ── Build dataset ─────────────────────────────────────────────────
    console.print(f"Loading {env.name} dataset...")
    raw_data = env.get_dataset()
    console.print(f"{len(raw_data)} examples loaded.")

    # Split into train / eval
    eval_size = int(len(raw_data) * config.eval_split_ratio)
    if eval_size > 0:
        train_data = raw_data[:-eval_size]
        eval_data = raw_data[-eval_size:]
    else:
        train_data = raw_data
        eval_data = None

    train_dataset = SFTDataset(
        train_data,
        prompt_field=config.prompt_field,
        completion_field=config.completion_field,
    )
    eval_dataset = (
        SFTDataset(
            eval_data,
            prompt_field=config.prompt_field,
            completion_field=config.completion_field,
        )
        if eval_data
        else None
    )

    console.print(f"  Train: {len(train_dataset)}  Eval: {len(eval_dataset) if eval_dataset else 0}")

    # ── Model + Tokenizer ─────────────────────────────────────────────
    console.print("Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Align model config with tokenizer so Trainer doesn't emit
    # "PAD/BOS/EOS tokens differ" warnings
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    # Enable gradient checkpointing if configured
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=config.gradient_checkpointing_kwargs
        )

    console.print("Model loaded.")

    # ── Apply PEFT / LoRA ─────────────────────────────────────────────
    if config.use_lora and config.lora_config is not None:
        from peft import get_peft_model, prepare_model_for_kbit_training

        console.print(
            f"Applying LoRA (rank={config.lora_rank}, "
            f"alpha={config.lora_alpha})..."
        )
        model = get_peft_model(model, config.lora_config)
        model.print_trainable_parameters()

    # ── Trainer ────────────────────────────────────────────────────────
    console.print("Initialising SFT trainer...")

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    console.print("Trainer ready. Starting training...")
    console.rule("Training")
    console.print()

    # ── Train ─────────────────────────────────────────────────────────
    trainer.train()

    # ── Save final model ──────────────────────────────────────────────
    console.print()
    console.print("Saving final model...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # If using LoRA, also save a merged version
    if config.use_lora:
        merged_dir = os.path.join(config.output_dir, "merged")
        console.print(f"Saving merged model to {merged_dir}...")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

    console.print()
    console.rule("Training Complete")
    console.print(f"  Model saved to: {config.output_dir}")
    if config.use_lora:
        console.print(f"  Merged model:   {merged_dir}")
    console.print()
