"""
Async GRPO RL Training – Main Entry Point (backward-compatible, defaults to GSM8K).

Thin script: create config, load model + tokenizer, connect to the
vLLM inference server, create the async trainer, call trainer.train().

All training logic lives in AsyncGRPOTrainer which extends HuggingFace
Trainer.  Inference runs on a separate vLLM server process that must
be started before this script.

GPU layout (typical 2-GPU setup):
    cuda:0 → training model                  (trainer_gpu_id)
    cuda:1 → vLLM server (started separately)

For the full tmux experience, launch via:
    python -m async_rl.launch [args...]
instead of calling this script directly.

For environment-based training (recommended), use:
    python -m environments.gsm8k --launch
    python -m environments.gpqa --launch
"""

import os
import sys
import logging
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from rich.console import Console

# Parent dir for cross-package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from async_rl.config import AsyncGRPOConfig
from async_rl.trainer import AsyncGRPOTrainer
from sync_rl.log_router import LogRouter

# Suppress noisy logs in the Main pane
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)
logging.getLogger("peft").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*use_cache=True.*")
warnings.filterwarnings("ignore", message=".*PAD/BOS/EOS.*")

console = Console(force_terminal=True, highlight=False)


def log(msg: str) -> None:
    """Print a Rich-formatted message to the Main pane."""
    console.print(msg)


def main():
    # ── Config (parse CLI arguments) ──────────────────────────────────
    parser = HfArgumentParser(AsyncGRPOConfig)
    (config,) = parser.parse_args_into_dataclasses()

    console.print()
    console.rule("Async GRPO Training")
    console.print()
    log(f"  Model    {config.model_name}")
    log(f"  GRPO     G={config.num_generations}  eps={config.epsilon}  beta={config.beta}")
    log(f"  Train    lr={config.learning_rate}  steps={config.max_steps}  batch={config.batch_size}  micro={config.micro_batch_size}")
    log(f"  Server   {config.vllm_server_host}:{config.vllm_server_port}")
    log(f"  GPU      trainer=cuda:{config.trainer_gpu_id}")
    if config.use_lora:
        log(f"  LoRA     rank={config.lora_rank}  alpha={config.lora_alpha}")
    if config.inflight_weight_updates:
        log(f"  Inflight ON  max_off_policy_steps={config.max_off_policy_steps}")
    else:
        log(f"  Inflight OFF (legacy blocking sync)")
    if config.continuous_batching:
        log(f"  ContBatch ON  pool_size={config.pool_size}")
    else:
        log(f"  ContBatch OFF (batch-at-a-time)")
    console.print()

    # ── Model + Tokenizer ──────────────────────────────────────────────
    log("[green]Loading model and tokenizer...[/green]")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(f"cuda:{config.trainer_gpu_id}")

    # Align model config with tokenizer so Trainer doesn't emit
    # "PAD/BOS/EOS tokens differ" warnings
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    log("[green]Model loaded.[/green]")

    # ── Log router (writes to files for tmux panes) ────────────────────
    log_dir = os.path.join(config.output_dir, "logs")
    log_router = LogRouter(log_dir=log_dir)
    log_router.start()

    # ── Trainer ────────────────────────────────────────────────────────
    log("[green]Initialising trainer + NCCL communicator...[/green]")
    trainer = AsyncGRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        log_router=log_router,
    )
    log("[green]Trainer ready. Starting training...[/green]")
    console.rule("[green]Training[/green]")
    console.print()

    # ── Train ──────────────────────────────────────────────────────────
    try:
        trainer.train()
    finally:
        log_router.stop()
    console.print()
    console.rule("[bold green]Training Complete[/bold green]")


if __name__ == "__main__":
    main()
