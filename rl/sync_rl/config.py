"""
Configuration for GRPO synchronous RL training.

Extends HuggingFace TrainingArguments so we inherit all standard training
infrastructure: optimizer, LR scheduler, gradient accumulation, mixed
precision, checkpointing, logging, distributed training, wandb, etc.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from peft import LoraConfig
from transformers import TrainingArguments


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "down_proj", "up_proj",
]


@dataclass
class GRPOConfig(TrainingArguments):
    """
    GRPO training configuration.

    Inherits from TrainingArguments -- standard fields like learning_rate,
    weight_decay, max_grad_norm, warmup_ratio, output_dir, seed,
    logging_steps, save_steps, bf16, gradient_accumulation_steps, etc.
    are all available and handled by the Trainer automatically.

    Only GRPO-specific fields are declared here.
    """

    # ── Model ──────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # ── Generation / Inference ─────────────────────────────────────────
    num_generations: int = 8          # G – group size per prompt
    max_new_tokens: int = 512         # max tokens per completion
    temperature: float = 0.7
    top_p: float = 0.95

    # ── GRPO objective ─────────────────────────────────────────────────
    epsilon_lower: float = 0.2        # lower clip bound: ratio >= 1 - epsilon_lower
    epsilon_upper: float = 0.2        # upper clip bound: ratio <= 1 + epsilon_upper
    beta: float = 0.04                # KL penalty coefficient

    # ── Batch (number of *prompts* sampled per training step) ──────────
    batch_size: int = 4

    # ── LoRA / PEFT ────────────────────────────────────────────────────
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None   # None → DEFAULT_LORA_TARGET_MODULES
    lora_modules_to_save: Optional[List[str]] = None
    lora_use_rslora: bool = False
    lora_config: Optional[LoraConfig] = field(default=None, repr=False)

    # ── Data ───────────────────────────────────────────────────────────
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    dataset_split: str = "train"
    max_prompt_length: int = 512

    # ── vLLM ───────────────────────────────────────────────────────────
    vllm_gpu_memory_utilization: float = 0.5
    vllm_dtype: str = "bfloat16"
    vllm_max_model_len: int = 1536    # prompt + completion budget
    vllm_gpu_id: int = 1              # GPU for vLLM inference (0-indexed)

    # ── Device placement ───────────────────────────────────────────────
    trainer_gpu_id: int = 0            # GPU for training + ref model

    # ── Override TrainingArguments defaults ─────────────────────────────
    output_dir: str = "outputs/grpo"
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    max_steps: int = 500
    logging_steps: int = 1
    save_steps: int = 50
    save_strategy: str = "steps"
    bf16: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 1   # dummy – real batch comes from vLLM
    gradient_accumulation_steps: int = 1
    remove_unused_columns: bool = False
    gradient_checkpointing: bool = True    # trade compute for memory
    gradient_checkpointing_kwargs: Optional[Dict] = field(
        default_factory=lambda: {"use_reentrant": False}
    )  # use_reentrant=False required for LoRA (frozen params)
    report_to: str = "none"                # set to "wandb" to enable W&B

    def __post_init__(self):
        # ── Build LoRA config from individual fields ───────────────────
        if not self.use_lora:
            self.lora_config = None
        elif self.lora_config is None:
            target_modules = self.lora_target_modules or DEFAULT_LORA_TARGET_MODULES
            self.lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=list(target_modules),
                modules_to_save=self.lora_modules_to_save,
                use_rslora=self.lora_use_rslora,
                task_type="CAUSAL_LM",
            )

        super().__post_init__()
