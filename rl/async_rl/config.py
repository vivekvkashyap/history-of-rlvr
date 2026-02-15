"""
Configuration for asynchronous RL training (GRPO / CISPO / DAPO).

Extends HuggingFace TrainingArguments so we inherit all standard training
infrastructure.  Compared to the sync_rl config, the in-process vLLM
parameters are replaced with connection parameters for a remote vLLM
inference server.

Supported algorithms:
  - grpo: GRPO (Group Relative Policy Optimization)
          DeepSeek-Math (https://arxiv.org/abs/2402.03300)
  - cispo: CISPO (Clipped IS-weight Policy Optimization)
           MiniMax-M1 (https://arxiv.org/abs/2506.13585)
  - dapo: DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
          ByteDance (https://arxiv.org/abs/2503.14476)
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
    Async RL training configuration (supports GRPO, CISPO, and DAPO).

    Inherits from TrainingArguments -- standard fields like learning_rate,
    weight_decay, max_grad_norm, warmup_ratio, output_dir, seed,
    logging_steps, save_steps, bf16, gradient_accumulation_steps, etc.
    are all available and handled by the Trainer automatically.

    Only RL-specific and async-specific fields are declared here.
    """

    # ── Model ──────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # ── Algorithm selection ────────────────────────────────────────────
    algorithm: str = "cispo"           # "grpo", "cispo", "dapo", or "prime"

    # ── Generation / Inference ─────────────────────────────────────────
    num_generations: int = 16         # G – group size per prompt
    max_new_tokens: int = 1024        # max tokens per completion
    temperature: float = 1.0
    top_p: float = 1.0

    # ── GRPO objective ─────────────────────────────────────────────────
    # Used when algorithm="grpo".
    # J_GRPO = min(r·A, clip(r, 1-ε_lo, 1+ε_hi)·A) - β·D_KL
    epsilon_lower: float = 0.2        # lower clip bound: ratio >= 1 - epsilon_lower
    epsilon_upper: float = 0.5       # upper clip bound: ratio <= 1 + epsilon_upper
    beta: float = 0.0                 # KL penalty coefficient (0 = no KL penalty)
    max_log_ratio: float = 10.0       # clamp log ratios to prevent exp overflow

    # ── CISPO objective (MiniMax-M1, arXiv:2506.13585) ─────────────────
    # Used when algorithm="cispo".
    # J_CISPO = sg(clip(r, 1-ε_low, 1+ε_high)) · A · log π_θ
    # Paper: "we did not impose a lower bound on the IS weight by setting
    #         ε_IS_low to a large value; instead, we only tuned ε_IS_high."
    # No KL penalty term in CISPO.
    cispo_epsilon_high: float = 8.0   # upper IS weight clip: r <= 1 + ε_high
                                      # (main tuning parameter; ScaleRL arXiv:2510.13786)
    cispo_epsilon_low: float = 0.0  # lower IS weight clip: r >= 1 - ε_low
                                      # (large value = effectively no lower bound, per paper)

    # ── Prime-RL objective (masked IS with stop-gradient) ─────────────
    # Used when algorithm="prime".
    # Masks out tokens with ratio outside [low, high] instead of clipping.
    # Gradients flow only through log π_θ (stop-gradient on coefficient).
    # Matches Prime Intellect's Prime-RL implementation.
    prime_token_mask_low: float = 0.125   # mask tokens with ratio < this
    prime_token_mask_high: float = 8.0    # mask tokens with ratio > this

    # ── DAPO objective (ByteDance, arXiv:2503.14476) ─────────────────
    # Used when algorithm="dapo".
    # J_DAPO = 1/(Σ|o_i|) Σ_i Σ_t min(r·A, clip(r, 1-ε_low, 1+ε_high)·A)
    # Uses epsilon_lower/epsilon_upper for Clip-Higher.
    # Token-level loss and Dynamic Sampling are automatic when algorithm="dapo".
    # Overlong Reward Shaping (Eq. 13): soft punishment for truncated responses.
    dapo_overlong_max_length: int = 16384  # expected max response length (tokens)
    dapo_overlong_cache: int = 4096        # soft punishment cache (tokens)

    # ── Batch ───────────────────────────────────────────────────────────
    batch_size: int = 512             # total rollouts (completions) per training step
    micro_batch_size: int = 8         # sequences per forward+backward pass

    # ── LoRA / PEFT ────────────────────────────────────────────────────
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16              # scaling factor = alpha/rank = 1.0
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None   # None → DEFAULT_LORA_TARGET_MODULES
    lora_modules_to_save: Optional[List[str]] = None
    lora_use_rslora: bool = False
    lora_config: Optional[LoraConfig] = field(default=None, repr=False)

    # ── Data ───────────────────────────────────────────────────────────
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    dataset_split: str = "train"
    max_prompt_length: int = 2048  # Increased for longer prompts (e.g., GPQA)

    # ── vLLM server connection ─────────────────────────────────────────
    vllm_server_host: str = "0.0.0.0"
    vllm_server_port: int = 8000
    vllm_gpu_memory_utilization: float = 0.7  # 0.5 if vLLM and trainer share a GPU
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
    max_off_policy_steps: int = 9          # strict on-policy (only used if inflight=True)

    # ── Evaluation / Display ───────────────────────────────────────────
    eval_display_steps: int = 10          # print examples every N steps
    eval_num_samples: int = 2             # number of examples to print

    # ── Periodic evaluation on held-out test set ─────────────────────
    eval_steps: int = 50                  # run full eval every N training steps (0 = disabled)
    eval_num_problems: int = 50           # number of test problems to evaluate on
    eval_temperature: float = 0.0         # temperature for eval generation (lower = more deterministic)
    eval_max_new_tokens: int = 1024       # max tokens for eval completions
    eval_split: str = "test"              # dataset split for evaluation

    # ── Device placement ───────────────────────────────────────────────
    trainer_gpu_id: int = 0               # GPU for training model

    # ── Override TrainingArguments defaults ─────────────────────────────
    output_dir: str = "outputs/async_rl"
    learning_rate: float = 5e-6           # GRPO/CISPO: 1e-6 to 5e-6 typical for Qwen2.5
    warmup_ratio: float = 0.05            # fraction of steps for linear warmup
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0            # Prime-RL default (was 0.1)
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
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Optional[Dict] = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    log_on_each_node: bool = False
    report_to: str = "wandb"

    def __post_init__(self):
        # ── Validate algorithm selection ────────────────────────────────
        assert self.algorithm in ("grpo", "cispo", "dapo", "prime"), (
            f"algorithm must be 'grpo', 'cispo', 'dapo', or 'prime', got '{self.algorithm}'."
        )

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
