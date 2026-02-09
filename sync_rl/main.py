"""
GRPO Synchronous RL Training – Main Entry Point.

Thin script: create config, load model + tokenizer, create trainer,
call trainer.train().  All training logic lives in GRPOTrainer which
extends HuggingFace Trainer.

GPU layout (2-GPU default):
    cuda:0 → training model                      (trainer_gpu_id)
    cuda:1 → vLLM inference engine               (vllm_gpu_id)
"""

import os
import sys
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

# Parent dir for cross-package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sync_rl.config import GRPOConfig
from sync_rl.trainer import GRPOTrainer

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
    logger.info(f"GRPO:   G={config.num_generations}, eps={config.epsilon}, "
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
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(f"cuda:{config.trainer_gpu_id}")

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
    )

    # ── Train ──────────────────────────────────────────────────────────
    logger.info("Starting GRPO training...")
    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
