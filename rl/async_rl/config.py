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
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    algorithm: str = "cispo"
    num_generations: int = 16
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    normalize_advantages_by_std: bool = False
    epsilon_lower: float = 0.2
    epsilon_upper: float = 0.5
    max_log_ratio: float = 10.0
    beta: float = 0.0
    prime_token_mask_low: float = 0.125
    prime_token_mask_high: float = 8.0
    loss_reduction: str = "token"
    use_overlong_penalty: Optional[bool] = None
    use_dynamic_sampling: Optional[bool] = None
    overlong_max_length: int = 16384
    overlong_cache: int = 4096
    batch_size: int = 512
    micro_batch_size: int = 8
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None
    lora_modules_to_save: Optional[List[str]] = None
    lora_use_rslora: bool = False
    lora_config: Optional[LoraConfig] = field(default=None, repr=False)
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    dataset_split: str = "train"
    max_prompt_length: int = 2048
    vllm_server_host: str = "0.0.0.0"
    vllm_server_port: int = 8000
    vllm_gpu_memory_utilization: float = 0.7
    vllm_server_timeout: float = 300.0
    group_port: int = 51216
    generation_timeout: float = 600.0
    max_concurrent: int = 1024
    continuous_batching: bool = True
    pool_size: int = 16
    inflight_weight_updates: bool = True
    max_off_policy_steps: int = 9
    eval_display_steps: int = 10
    eval_num_samples: int = 2
    eval_steps: int = 50
    eval_num_problems: int = 50
    eval_temperature: float = 0.0
    eval_max_new_tokens: int = 1024
    eval_split: str = "test"
    trainer_gpu_id: int = 0
    output_dir: str = "outputs/async_rl"
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.05
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
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    remove_unused_columns: bool = False
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Optional[Dict] = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    log_on_each_node: bool = False
    report_to: str = "wandb"

    def __post_init__(self):
        valid_algorithms = ("grpo", "dr_grpo", "cispo", "dapo", "prime", "gspo")
        if self.algorithm not in valid_algorithms:
            raise ValueError(
                f"algorithm must be one of {valid_algorithms}, "
                f"got '{self.algorithm}'"
            )

        if self.algorithm in ("grpo", "dr_grpo"):
            if self.beta is None:
                raise ValueError(
                    f"{self.algorithm.upper()} requires beta (KL penalty coefficient) to be set."
                )

        if self.algorithm == "prime":
            if self.prime_token_mask_low is None or self.prime_token_mask_high is None:
                raise ValueError(
                    "Prime requires prime_token_mask_low and "
                    "prime_token_mask_high to be set."
                )

        valid_reductions = ("token", "sample")
        if self.loss_reduction not in valid_reductions:
            raise ValueError(
                f"loss_reduction must be one of {valid_reductions}, "
                f"got '{self.loss_reduction}'"
            )

        if self.use_overlong_penalty is None:
            self.use_overlong_penalty = (self.algorithm == "dapo")
        if self.use_dynamic_sampling is None:
            self.use_dynamic_sampling = (self.algorithm == "dapo")

        assert self.batch_size % self.num_generations == 0, (
            f"batch_size ({self.batch_size}) must be divisible by "
            f"num_generations ({self.num_generations})."
        )
        assert self.batch_size % self.micro_batch_size == 0, (
            f"batch_size ({self.batch_size}) must be divisible by "
            f"micro_batch_size ({self.micro_batch_size})."
        )

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
