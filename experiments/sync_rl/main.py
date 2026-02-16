"""
GRPO Synchronous RL Training – Main Entry Point.

Thin script: create config, load model + tokenizer, create trainer,
call trainer.train().  All training logic lives in GRPOTrainer which
extends HuggingFace Trainer.

GPU layout (2-GPU default):
    cuda:0 → training model                      (trainer_gpu_id)
    cuda:1 → vLLM inference engine               (vllm_gpu_id)

For the split-pane tmux experience, launch via:
    python -m experiments.sync_rl.launch [args...]
instead of calling this script directly.
"""

import os
import sys
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from history_of_rlvr.rl.log_router import LogRouter

from .config import GRPOConfig
from .trainer import GRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    # ── Config (parse CLI arguments) ──────────────────────────────────
    parser = HfArgumentParser(GRPOConfig)
    (config,) = parser.parse_args_into_dataclasses()
    logger.info(f"Model:  {config.model_name}")
    logger.info(f"GRPO:   G={config.num_generations}, eps_lo={config.epsilon_lower}, eps_hi={config.epsilon_upper}, "
                f"beta={config.beta}")
    logger.info(f"Train:  lr={config.learning_rate}, max_steps={config.max_steps}, "
                f"batch_size={config.batch_size}")
    logger.info(f"GPUs:   trainer=cuda:{config.trainer_gpu_id}, "
                f"vllm=cuda:{config.vllm_gpu_id}")

    # ── Model + Tokenizer ──────────────────────────────────────────────
    logger.info(f"Loading model and tokenizer on cuda:{config.trainer_gpu_id}...")
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

    # ── Log router (writes to files for tmux panes) ────────────────────
    log_dir = os.path.join(config.output_dir, "logs")
    log_router = LogRouter(log_dir=log_dir)
    log_router.start()

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        log_router=log_router,
    )

    # ── Train ──────────────────────────────────────────────────────────
    logger.info("Starting GRPO training...")
    try:
        trainer.train()
    finally:
        log_router.stop()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
