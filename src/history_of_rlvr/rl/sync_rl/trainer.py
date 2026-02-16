"""
GRPO Trainer – extends HuggingFace Trainer.

All standard training infrastructure (optimizer, LR scheduler, gradient
accumulation, mixed precision, checkpointing, logging, distributed
training) is inherited from Trainer.  Only the GRPO-specific logic lives
here: vLLM generation, reward computation, advantage calculation, and
the GRPO loss.
"""

import os
import sys
import time
import random
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from accelerate.utils import is_peft_model
from peft import PeftConfig, get_peft_model
from transformers.trainer import Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from history_of_rlvr.rl.algorithms.grpo.grpo import compute_group_advantages, grpo_loss

from history_of_rlvr.rl.sync_rl.config import GRPOConfig
from history_of_rlvr.rl.sync_rl.data import GSM8KDataset
from history_of_rlvr.rl.sync_rl.reward import compute_rewards_batch
from history_of_rlvr.rl.sync_rl.inference import VLLMInferenceEngine, InferenceConfig

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────


def selective_log_softmax(
    logits: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """
    Memory-efficient log_softmax + gather.

    Equivalent to:
        logits.log_softmax(-1).gather(-1, index.unsqueeze(-1)).squeeze(-1)
    but avoids materialising the full (B, L, V) log-probability tensor.

    Always processes row-by-row so that only one (L, V) slice is live at
    a time.  This is critical because the full (B, L, V) tensor for
    Qwen-0.5B is ~1.6 GiB per chunk and causes OOM.
    (Same strategy as the verifiers package.)
    """
    per_token_logps = []
    for row_logits, row_labels in zip(logits, index):
        row_logps = torch.nn.functional.log_softmax(row_logits, dim=-1)
        row_per_token = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
        per_token_logps.append(row_per_token)
    return torch.stack(per_token_logps)


# ── Trainer ────────────────────────────────────────────────────────────


class GRPOTrainer(Trainer):
    """
    GRPO trainer that extends HuggingFace ``Trainer``.

    Key overrides
    -------------
    * ``training_step`` – runs the full GRPO pipeline: sample prompts,
      generate completions via vLLM, compute rewards/advantages, compute
      the GRPO loss, call ``accelerator.backward``.
    * ``compute_loss`` – GRPO clipped surrogate + KL penalty.
    * ``get_train_dataloader`` – dummy dataloader (data comes from vLLM).
    * ``log`` – injects GRPO metrics into the standard logging output.
    """

    def __init__(
        self,
        model: nn.Module,
        args: GRPOConfig,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        log_router=None,
        **kwargs,
    ):
        # Suppress "estimate_tokens" warning from Trainer
        warnings_issued = getattr(model, "warnings_issued", None)
        if isinstance(warnings_issued, dict):
            warnings_issued["estimate_tokens"] = True

        self.log_router = log_router

        # ── LoRA: wrap model with PEFT *before* Trainer.__init__ ───────
        # This ensures the Trainer creates the optimizer over only the
        # LoRA parameters (the rest are frozen by PEFT automatically).
        if args.use_lora and isinstance(args.lora_config, PeftConfig):
            logger.info(
                f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}, "
                f"target_modules={args.lora_config.target_modules})"
            )
            model = get_peft_model(model, args.lora_config)
            model.print_trainable_parameters()

        super().__init__(
            model=model,
            args=args,
            processing_class=processing_class,
            **kwargs,
        )

        # Ensure tokenizer has a pad token
        assert isinstance(self.processing_class, PreTrainedTokenizerBase)
        if self.processing_class.pad_token is None:
            self.processing_class.pad_token = self.processing_class.eos_token
        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token_id = self.processing_class.eos_token_id

        self._trainer_device = torch.device(f"cuda:{args.trainer_gpu_id}")

        # ── GSM8K dataset ──────────────────────────────────────────────
        logger.info("Loading GSM8K dataset...")
        self.gsm8k_dataset = GSM8KDataset(
            split=args.dataset_split,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
        )
        logger.info(f"  {len(self.gsm8k_dataset)} training examples loaded")

        # ── vLLM inference engine ──────────────────────────────────────
        logger.info(f"Initialising vLLM inference engine on cuda:{args.vllm_gpu_id}...")
        self.inference_engine = VLLMInferenceEngine(
            InferenceConfig(
                model_name=args.model_name,
                num_generations=args.num_generations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                dtype=args.vllm_dtype,
                max_model_len=args.vllm_max_model_len,
                gpu_id=args.vllm_gpu_id,
            )
        )

        # ── Metric accumulator (merged into Trainer logs) ──────────────
        self._grpo_metrics: Dict[str, list] = defaultdict(list)

        # ── Rollout buffer for wandb completions table ────────────────
        self._grpo_rollouts: Optional[Dict[str, Any]] = None

        # ── Remove noisy HF Trainer callbacks when log_router is active ─
        # (output goes to log files, not stdout)
        if self.log_router:
            from transformers.trainer_callback import PrinterCallback, ProgressCallback
            self.remove_callback(PrinterCallback)
            self.remove_callback(ProgressCallback)

    # ================================================================
    #  Prevent DataParallel (GPU 1 is reserved for vLLM)
    # ================================================================

    def _wrap_model(self, model, training=True, dataloader=None):
        """Skip DataParallel wrapping – we manage GPU placement manually."""
        return model

    # ================================================================
    #  Dataloader – dummy (real data comes from vLLM generation)
    # ================================================================

    def get_train_dataloader(self) -> DataLoader:
        """
        Return a dummy dataloader whose length equals ``max_steps``.

        The Trainer iterates over this to drive the training loop; actual
        data is generated inside ``training_step`` via vLLM.
        """

        class _StepsDataset(Dataset):
            def __init__(self, n: int):
                self.n = n

            def __len__(self) -> int:
                return self.n

            def __getitem__(self, idx: int) -> dict:
                return {"labels": 0}

        return DataLoader(_StepsDataset(self.args.max_steps))

    # ================================================================
    #  Core: training_step  (called once per step by the Trainer)
    # ================================================================

    def training_step(
        self, model: nn.Module, inputs: Any = None, num_items_in_batch: Any = None, **kwargs
    ) -> torch.Tensor:
        """
        Full GRPO training step:

        1. Sync updated weights to vLLM (skip first step)
        2. Sample a batch of prompts from GSM8K
        3. Generate G completions per prompt via vLLM (+ get inference logprobs)
        4. Compute binary rewards (math answer verification)
        5. Compute group-relative advantages
        6. Single forward pass → trainer log probs (with gradients)
        7. GRPO loss (clipped surrogate + KL penalty vs inference logprobs)
        8. ``accelerator.backward(loss)``
        9. Return detached loss
        """
        # ── 1. Weight sync ─────────────────────────────────────────────
        t0 = time.time()
        if self.state.global_step > 0:
            self._sync_vllm_weights()
        t_sync = time.time() - t0

        model.train()
        device = self.accelerator.device
        G = self.args.num_generations

        # ── 2. Sample batch ────────────────────────────────────────────
        batch = self._sample_batch()
        prompts = batch["prompts"]
        ground_truths = batch["ground_truths"]
        B = len(prompts)

        # ── 3. Generate completions (+ inference logprobs from vLLM) ──
        if self.log_router:
            self.log_router.log_inference(
                f"Generating [bold]{B}[/bold]x[bold]{G}[/bold] completions "
                f"for step [bold]{self.state.global_step}[/bold]..."
            )
        t0 = time.time()
        completions, completion_token_ids, completion_logprobs = (
            self.inference_engine.generate(prompts)
        )
        t_gen = time.time() - t0

        # ── Post-generation stats ───────────────────────────────────────
        comp_lengths = [len(ids) for ids in completion_token_ids]
        avg_comp_len = sum(comp_lengths) / len(comp_lengths)
        max_comp_len = float(max(comp_lengths))
        total_tokens = sum(comp_lengths)
        tok_per_s = total_tokens / t_gen if t_gen > 0 else 0.0

        if self.log_router:
            self.log_router.log_inference(
                f"Generated [bold]{B * G}[/bold] completions in "
                f"[yellow]{t_gen:.1f}s[/yellow]  "
                f"([dim]{tok_per_s:.0f} tok/s, avg {avg_comp_len:.0f} tok, "
                f"max {max_comp_len:.0f} tok[/dim])"
            )

        # ── 4. Compute rewards ─────────────────────────────────────────
        expanded_gts = [gt for gt in ground_truths for _ in range(G)]
        rewards = compute_rewards_batch(completions, expanded_gts)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

        # ── 5. Group-relative advantages ───────────────────────────────
        advantages = compute_group_advantages(rewards_tensor, G)

        # ── 6. Tokenize & single forward pass ──────────────────────────
        t0 = time.time()
        expanded_prompts = [p for p in prompts for _ in range(G)]
        grpo_inputs = self._prepare_grpo_inputs(
            expanded_prompts, completion_token_ids, completion_logprobs,
        )
        input_ids = grpo_inputs["input_ids"]
        attention_mask = grpo_inputs["attention_mask"]
        loss_mask = grpo_inputs["loss_mask"]
        inference_logprobs = grpo_inputs["inference_logprobs"]

        # Only ONE forward pass – with gradients
        torch.cuda.empty_cache()
        trainer_logprobs = self._get_logprobs(model, input_ids, attention_mask)

        # Shift loss_mask to match the shifted logprobs (logits[:, :-1])
        loss_mask_shifted = loss_mask[:, 1:]

        # ── 7 + 8. GRPO loss + backward ───────────────────────────────
        with self.compute_loss_context_manager():
            loss, stats = self.compute_loss(
                model,
                {
                    "trainer_logprobs": trainer_logprobs,
                    "inference_logprobs": inference_logprobs,
                    "advantages": advantages,
                    "loss_mask": loss_mask_shifted,
                },
                return_outputs=True,
            )

        self.accelerator.backward(loss)
        t_train = time.time() - t0

        # ── Accumulate GRPO metrics ────────────────────────────────────
        # Reward metrics
        stats["rewards/mean"] = rewards_tensor.mean().item()
        stats["rewards/std"] = rewards_tensor.std().item()
        stats["rewards/advantage_mean"] = advantages.mean().item()

        # Training / generation metrics
        stats["train/completion_len_mean"] = avg_comp_len
        stats["train/completion_len_max"] = max_comp_len
        stats["train/masked_fraction"] = 1.0 - (
            loss_mask_shifted.sum() / loss_mask_shifted.numel()
        ).item()
        stats["train/sync_s"] = t_sync
        stats["train/generate_s"] = t_gen
        stats["train/train_s"] = t_train

        for key, value in stats.items():
            self._grpo_metrics[key].append(value)

        # ── Store rollout for wandb completions table ─────────────────
        self._grpo_rollouts = {
            "prompts": prompts,
            "completions": completions,
            "rewards": rewards,
        }

        # ── Log router: trainer update ─────────────────────────────────
        if self.log_router:
            self.log_router.log_trainer(
                f"step [bold]{self.state.global_step}[/bold] | "
                f"loss=[red]{loss.item():.4f}[/red] | "
                f"reward=[yellow]{stats['rewards/mean']:.3f}[/yellow] | "
                f"kl={stats['loss/kl']:.4f} | "
                f"gen={t_gen:.1f}s train={t_train:.1f}s | "
                f"comp_len={avg_comp_len:.0f}"
            )

        # ── Console summary (fallback when no log_router) ─────────────
        if not self.log_router:
            logger.info(
                f"step {self.state.global_step} | "
                f"loss={loss.item():.4f} | "
                f"reward={stats['rewards/mean']:.3f} | "
                f"kl={stats['loss/kl']:.4f} | "
                f"gen={t_gen:.1f}s | train={t_train:.1f}s | "
                f"comp_len={stats['train/completion_len_mean']:.0f}"
            )

        # ── 9. Return ─────────────────────────────────────────────────
        return loss.detach()

    # ================================================================
    #  Loss computation
    # ================================================================

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ):
        """
        Compute the GRPO loss (clipped surrogate + KL penalty).

        Uses ``trainer_logprobs`` (from the current forward pass with
        gradients) and ``inference_logprobs`` (from vLLM, serves as
        both the old-policy baseline and KL reference).

        Delegates to the standalone ``grpo_loss`` from ``rl/grpo.py``.
        """
        loss, stats = grpo_loss(
            trainer_log_probs=inputs["trainer_logprobs"],
            inference_log_probs=inputs["inference_logprobs"],
            advantages=inputs["advantages"],
            completion_mask=inputs["loss_mask"],
            epsilon_lower=self.args.epsilon_lower,
            epsilon_upper=self.args.epsilon_upper,
            beta=self.args.beta,
        )

        if return_outputs:
            return loss, stats
        return loss

    # ================================================================
    #  Log-probability computation
    # ================================================================

    def _get_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int = 4,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities for a full sequence.

        Processes in mini-batches of ``chunk_size`` to avoid OOM from the
        large (B, L, V) logits tensor.

        Returns:
            logprobs: (B, L-1) – shifted so that logprobs[:, t]
                      corresponds to log P(input_ids[:, t+1] | context).
        """
        all_logprobs = []
        B = input_ids.shape[0]

        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk_ids = input_ids[start:end]
            chunk_mask = attention_mask[start:end]

            with torch.set_grad_enabled(model.training):
                logits = model(
                    input_ids=chunk_ids,
                    attention_mask=chunk_mask,
                ).logits

            # Shift: logits at position t predict token at t+1
            logits = logits[:, :-1, :]          # (chunk, L-1, V)
            targets = chunk_ids[:, 1:]          # (chunk, L-1)

            chunk_logprobs = selective_log_softmax(logits, targets)
            all_logprobs.append(chunk_logprobs)

            del logits  # free the large tensor immediately

        return torch.cat(all_logprobs, dim=0)  # (B, L-1)

    # ================================================================
    #  Input preparation
    # ================================================================

    def _prepare_grpo_inputs(
        self,
        prompts: List[str],
        completion_token_ids: List[List[int]],
        completion_logprobs: List[List[float]],
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize prompt + completion pairs, right-pad, and build masks.

        Also aligns the per-completion-token vLLM logprobs into a tensor
        that matches the shape of the shifted logprobs from ``_get_logprobs``
        (i.e. shape ``(B*G, L-1)``).

        Returns:
            input_ids:          (B*G, L)    padded token ids
            attention_mask:     (B*G, L)    1 for real tokens, 0 for padding
            loss_mask:          (B*G, L)    1 for completion tokens only
            inference_logprobs: (B*G, L-1)  vLLM logprobs aligned with shifted seq
        """
        tokenizer = self.processing_class
        device = self.accelerator.device

        # Tokenize prompts (returns plain lists, not tensors)
        prompt_encodings = tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.args.max_prompt_length,
        )

        # Build full sequences and loss masks
        all_input_ids: List[List[int]] = []
        all_loss_masks: List[List[int]] = []
        prompt_lengths: List[int] = []

        for i in range(len(prompts)):
            p_ids = prompt_encodings["input_ids"][i]
            c_ids = completion_token_ids[i]
            all_input_ids.append(p_ids + c_ids)
            all_loss_masks.append([0] * len(p_ids) + [1] * len(c_ids))
            prompt_lengths.append(len(p_ids))

        # Right-pad to max length
        max_len = max(len(seq) for seq in all_input_ids)
        pad_id = tokenizer.pad_token_id or 0

        padded_ids = []
        padded_attn = []
        padded_loss = []

        for seq, mask in zip(all_input_ids, all_loss_masks):
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [pad_id] * pad_len)
            padded_attn.append([1] * len(seq) + [0] * pad_len)
            padded_loss.append(mask + [0] * pad_len)

        # Build inference logprobs tensor aligned with shifted logprobs
        # _get_logprobs returns (B*G, L-1) where position t gives
        # log P(input_ids[:, t+1] | ...).  Completion token c_i at
        # sequence position (prompt_len + i) is predicted at shifted
        # position (prompt_len + i - 1).
        inference_lps = torch.zeros(
            len(prompts), max_len - 1, dtype=torch.float32, device=device,
        )
        for i in range(len(prompts)):
            c_lps = completion_logprobs[i]
            start_pos = prompt_lengths[i] - 1
            end_pos = start_pos + len(c_lps)
            fit = min(end_pos, max_len - 1) - start_pos
            if fit > 0:
                inference_lps[i, start_pos:start_pos + fit] = torch.tensor(
                    c_lps[:fit], dtype=torch.float32, device=device,
                )

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(padded_attn, dtype=torch.long, device=device),
            "loss_mask": torch.tensor(padded_loss, dtype=torch.float, device=device),
            "inference_logprobs": inference_lps,
        }

    # ================================================================
    #  Batch sampling
    # ================================================================

    def _sample_batch(self) -> Dict[str, List[str]]:
        """Sample ``batch_size`` prompts from the GSM8K dataset."""
        n = len(self.gsm8k_dataset)
        indices = random.sample(range(n), min(self.args.batch_size, n))
        items = [self.gsm8k_dataset[i] for i in indices]
        return {
            "prompts": [item["prompt"] for item in items],
            "ground_truths": [item["ground_truth"] for item in items],
            "questions": [item["question"] for item in items],
        }

    # ================================================================
    #  vLLM weight sync
    # ================================================================

    def _sync_vllm_weights(self) -> None:
        """
        Push updated policy weights from the HF model into vLLM.

        For PEFT/LoRA models the adapters are temporarily merged into
        the base weights so vLLM (which knows nothing about adapters)
        receives a standard state dict.  After syncing, the adapters
        are unmerged so training can continue on the low-rank matrices.

        Uses ``VLLMInferenceEngine.update_weights(state_dict)`` which
        saves to a temp file and loads via ``llm.apply_model()``
        (compatible with vLLM V1 multi-process engine).
        """
        unwrapped = self.accelerator.unwrap_model(self.model)

        if is_peft_model(unwrapped):
            # Merge LoRA adapters → base weights
            unwrapped.merge_adapter()

            # Build a clean state dict with original parameter names
            merged_state = {}
            for name, param in unwrapped.named_parameters():
                # Strip PEFT name prefixes:
                #   "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
                #   → "model.layers.0.self_attn.q_proj.weight"
                name = name.removeprefix("base_model.model.").replace(
                    ".base_layer", ""
                )
                # Skip PEFT-internal bookkeeping params
                if unwrapped.prefix in name:
                    continue
                if "original_module" in name:
                    continue
                name = name.replace("modules_to_save.default.", "")
                merged_state[name] = param.data

            # Push merged weights into vLLM
            self.inference_engine.update_weights(merged_state)

            # Unmerge so LoRA training can continue
            unwrapped.unmerge_adapter()
            if self.log_router:
                self.log_router.log_inference("[cyan]Synced merged LoRA weights to vLLM[/cyan]")
            else:
                logger.info("Synced merged LoRA weights → vLLM")
        else:
            self.inference_engine.update_weights(unwrapped.state_dict())
            if self.log_router:
                self.log_router.log_inference("[cyan]Synced policy weights to vLLM[/cyan]")
            else:
                logger.info("Synced policy weights → vLLM")

    # ================================================================
    #  Logging (merge GRPO metrics into Trainer logs)
    # ================================================================

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        """
        Override to inject accumulated GRPO metrics into every log call
        and optionally log a wandb completions table.
        """
        # Average the accumulated GRPO metrics
        grpo_avg = {
            key: sum(vals) / len(vals)
            for key, vals in self._grpo_metrics.items()
            if vals
        }
        logs = {**logs, **grpo_avg}

        # ── wandb completions table ───────────────────────────────────
        if (
            self._grpo_rollouts is not None
            and self.args.report_to
            and "wandb" in self.args.report_to
        ):
            try:
                import wandb

                if wandb.run is not None:
                    rollouts = self._grpo_rollouts
                    G = self.args.num_generations
                    table_data = []
                    for i, (comp, rew) in enumerate(
                        zip(rollouts["completions"], rollouts["rewards"])
                    ):
                        prompt_idx = i // G
                        table_data.append([
                            self.state.global_step,
                            rollouts["prompts"][prompt_idx],
                            comp,
                            rew,
                        ])
                    wandb.log(
                        {
                            "completions": wandb.Table(
                                columns=["step", "prompt", "completion", "reward"],
                                data=table_data,
                            )
                        },
                        commit=False,
                    )
            except ImportError:
                pass

        # ── Log router: emit HF Trainer metrics (grad_norm, lr) ────────
        if self.log_router:
            lr = logs.get("learning_rate")
            gn = logs.get("grad_norm")
            if lr is not None or gn is not None:
                parts = []
                if lr is not None:
                    parts.append(f"lr=[magenta]{lr:.2e}[/magenta]")
                if gn is not None:
                    parts.append(f"grad_norm=[magenta]{gn:.4f}[/magenta]")
                self.log_router.log_trainer("  " + "  ".join(parts))

        super().log(logs, start_time)
        self._grpo_metrics.clear()
