#!/usr/bin/env python3
"""
Merge a LoRA checkpoint into the base model and save for inference.

Use this if vLLM LoRA serving has issues, or to serve without --enable-lora.

Usage:
    python scripts/merge_lora_checkpoint.py outputs/gsm8k_rl_grpo/checkpoint-50
    # Creates outputs/gsm8k_rl_grpo/merged-50/

    # Then serve the merged model (no LoRA flags):
    vllm serve outputs/gsm8k_rl_grpo/merged-50 --port 8000 ...
    python scripts/eval_gsm8k.py --model_name outputs/gsm8k_rl_grpo/merged-50
"""

import argparse
import os
import sys
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to LoRA checkpoint (e.g. outputs/gsm8k_rl_grpo/checkpoint-50)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <checkpoint_parent>/merged-<checkpoint_name>)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model (default: read from adapter_config.json)",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        print(f"Error: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    # Load adapter config to get base model
    import json
    adapter_config_path = checkpoint / "adapter_config.json"
    if not adapter_config_path.exists():
        print(f"Error: {adapter_config_path} not found")
        sys.exit(1)

    with open(adapter_config_path) as f:
        config = json.load(f)
    base_model_name = args.base_model or config.get("base_model_name_or_path")
    if not base_model_name:
        print("Error: base_model_name_or_path not in adapter_config.json")
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        parent = checkpoint.parent
        ckpt_name = checkpoint.name
        output_dir = parent / f"merged-{ckpt_name.replace('checkpoint-', '')}"
    output_dir = Path(output_dir).resolve()

    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print(f"Loading LoRA from: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint)
    print("Merging adapter...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print()
    print("Done. To serve and evaluate:")
    print(f"  vllm serve {output_dir} --port 8000 --host 0.0.0.0 ...")
    print(f"  python scripts/eval_gsm8k.py --model_name {output_dir} --num_problems 1319")


if __name__ == "__main__":
    main()
