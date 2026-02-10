"""
Configuration for Supervised Fine-Tuning (SFT).

Extends HuggingFace TrainingArguments so we inherit all standard training
infrastructure.  Only SFT-specific fields are declared here.
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
class SFTConfig(TrainingArguments):
    """
    SFT training configuration.

    Inherits from TrainingArguments -- standard fields like learning_rate,
    weight_decay, max_grad_norm, warmup_ratio, output_dir, seed,
    logging_steps, save_steps, bf16, gradient_accumulation_steps, etc.
    are all available and handled by the Trainer automatically.

    Only SFT-specific fields are declared here.
    """

    # ── Model ──────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # ── Data ─────────────────────────────────────────────────────────
    max_seq_length: int = 1024             # max total sequence length (prompt + completion)
    prompt_field: str = "prompt"           # key for the prompt in dataset dicts
    completion_field: str = "completion"   # key for the completion in dataset dicts

    # ── LoRA / PEFT ────────────────────────────────────────────────────
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None   # None → DEFAULT_LORA_TARGET_MODULES
    lora_modules_to_save: Optional[List[str]] = None
    lora_use_rslora: bool = False
    lora_config: Optional[LoraConfig] = field(default=None, repr=False)

    # ── Eval ─────────────────────────────────────────────────────────
    eval_split_ratio: float = 0.05         # fraction of data to use for eval
    eval_display_samples: int = 3          # number of samples to display during eval

    # ── Override TrainingArguments defaults ─────────────────────────────
    output_dir: str = "outputs/sft"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    save_steps: int = 500
    save_strategy: str = "steps"
    eval_strategy: str = "steps"
    eval_steps: int = 500
    bf16: bool = True
    seed: int = 42
    remove_unused_columns: bool = False
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Optional[Dict] = field(
        default_factory=lambda: {"use_reentrant": False}
    )  # use_reentrant=False required for LoRA (frozen params)
    report_to: str = "none"                # set to "wandb" to enable W&B
    log_level: str = "error"               # suppress HF trainer logs
    disable_tqdm: bool = True              # disable progress bars

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
