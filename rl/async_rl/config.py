"""
Configuration for GRPO asynchronous RL training.

Extends HuggingFace TrainingArguments so we inherit all standard training
infrastructure.  Compared to the sync_rl config, the in-process vLLM
parameters are replaced with connection parameters for a remote vLLM
inference server.
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
class AsyncGRPOConfig(TrainingArguments):
    """
    Async GRPO training configuration.

    Inherits from TrainingArguments -- standard fields like learning_rate,
    weight_decay, max_grad_norm, warmup_ratio, output_dir, seed,
    logging_steps, save_steps, bf16, gradient_accumulation_steps, etc.
    are all available and handled by the Trainer automatically.

    Only GRPO-specific and async-specific fields are declared here.
    """

    # ── Model ──────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # ── Generation / Inference ─────────────────────────────────────────
    num_generations: int = 16         # G – group size per prompt
    max_new_tokens: int = 1024        # max tokens per completion
    temperature: float = 1.0
    top_p: float = 1.0

    # ── GRPO objective ─────────────────────────────────────────────────
    epsilon: float = 0.2              # clipping range for policy ratio
    beta: float = 0.01                # KL penalty coefficient (lower = less restrictive)
    max_log_ratio: float = 10.0       # clamp log ratios to prevent exp overflow (increased for stability)

    # ── Batch ───────────────────────────────────────────────────────────
    batch_size: int = 512             # total rollouts (completions) per training step
    micro_batch_size: int = 8         # sequences per forward+backward pass

    # ── LoRA / PEFT ────────────────────────────────────────────────────
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 32              # 4x rank for proper scaling
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

    # ── vLLM server connection ─────────────────────────────────────────
    vllm_server_host: str = "0.0.0.0"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 300.0   # seconds to wait for server to come up
    group_port: int = 51216               # NCCL communicator port

    # ── Async orchestration ────────────────────────────────────────────
    generation_timeout: float = 600.0     # max seconds to wait for a batch
    max_concurrent: int = 1024            # max concurrent HTTP connections to vLLM

    # ── Continuous batching ─────────────────────────────────────────────
    continuous_batching: bool = True      # keep a saturated pool of rollout tasks
    pool_size: int = 16                   # number of concurrent prompt-generation slots

    # ── In-flight weight updates (PipelineRL-style) ───────────────────
    inflight_weight_updates: bool = True  # disable for on-policy learning
    max_off_policy_steps: int = 4          # strict on-policy (only used if inflight=True)

    # ── Evaluation / Display ───────────────────────────────────────────
    eval_display_steps: int = 10          # print examples every N steps
    eval_num_samples: int = 2             # number of examples to print

    # ── Device placement ───────────────────────────────────────────────
    trainer_gpu_id: int = 0               # GPU for training model

    # ── Override TrainingArguments defaults ─────────────────────────────
    output_dir: str = "outputs/async_grpo"
    learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    max_steps: int = 500
    logging_steps: int = 1
    save_steps: int = 50
    save_strategy: str = "steps"
    bf16: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 1   # dummy – real batch comes from orchestrator
    gradient_accumulation_steps: int = 1   # grad accum handled by micro-batch loop
    remove_unused_columns: bool = False
    gradient_checkpointing: bool = False
    log_on_each_node: bool = False
    report_to: str = "wandb"

    def __post_init__(self):
        # ── Validate batch / micro-batch divisibility ──────────────────
        assert self.batch_size % self.num_generations == 0, (
            f"batch_size ({self.batch_size}) must be divisible by "
            f"num_generations ({self.num_generations})."
        )
        assert self.batch_size % self.micro_batch_size == 0, (
            f"batch_size ({self.batch_size}) must be divisible by "
            f"micro_batch_size ({self.micro_batch_size})."
        )

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
