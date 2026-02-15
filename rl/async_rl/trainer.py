"""
Async GRPO Trainer – extends HuggingFace Trainer.

All standard training infrastructure (optimizer, LR scheduler, gradient
accumulation, mixed precision, checkpointing, logging, distributed
training) is inherited from Trainer.  Only the GRPO-specific logic lives
here.

Key difference from sync_rl.trainer:
  - Inference runs on a remote vLLM server (always on).
  - Weight sync uses NCCL broadcast (via VLLMClient) instead of file I/O.
  - An Orchestrator thread pipelines batch generation: while training
    on batch N, batch N+1 is already being generated.
"""

import os
import sys
import time
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from accelerate.utils import is_peft_model
from peft import PeftConfig, get_peft_model
from transformers.trainer import Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Parent dir for cross-package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from rl.algorithms.grpo.grpo import grpo_loss
from rl.algorithms.cispo.cispo import cispo_loss
from rl.algorithms.dapo.dapo import dapo_loss
from rl.algorithms.prime.prime import prime_loss

from rl.async_rl.config import AsyncGRPOConfig
from rl.async_rl.client import VLLMClient
from rl.async_rl.display import print_examples
from rl.async_rl.evaluator import Evaluator
from rl.async_rl.orchestrator import Orchestrator, Batch

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────


def selective_log_softmax(
    logits: torch.Tensor, index: torch.Tensor,
) -> torch.Tensor:
    """
    Memory-efficient log_softmax + gather.

    Processes row-by-row so that only one (L, V) slice is live at a time,
    avoiding the full (B, L, V) materialisation that causes OOM.
    """
    per_token_logps = []
    for row_logits, row_labels in zip(logits, index):
        row_logps = torch.nn.functional.log_softmax(row_logits, dim=-1)
        row_per_token = row_logps.gather(
            dim=-1, index=row_labels.unsqueeze(-1),
        ).squeeze(-1)
        per_token_logps.append(row_per_token)
    return torch.stack(per_token_logps)


def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """
    Compute the Shannon entropy (in nats) for each row of *logits* in a
    memory-efficient way.

    Instead of materializing the full softmax for all rows at once, the
    logits are flattened to shape (N, num_classes), where N is the product
    of all leading dimensions. Computation is then performed in chunks of
    size ``chunk_size`` along this flattened dimension, reducing peak
    memory usage. The result is reshaped back to match the input's leading
    dimensions.

    Args:
        logits: Logits tensor of shape ``(..., num_classes)``. Entropy is
            taken along the last axis; all leading dimensions are preserved
            in the output.
        chunk_size: Number of rows from the flattened logits to process per
            iteration. Smaller values reduce memory usage at the cost of
            more iterations.

    Returns:
        Entropy values with shape ``logits.shape[:-1]``.
    """
    original_shape = logits.shape[:-1]  # all dims except num_classes
    num_classes = logits.shape[-1]

    # Flatten all leading dimensions into one
    flat_logits = logits.reshape(-1, num_classes)

    entropies = []
    for chunk in flat_logits.split(chunk_size, dim=0):
        logps = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        entropies.append(chunk_entropy)

    entropies = torch.cat(entropies, dim=0)
    return entropies.reshape(original_shape)


# ── Trainer ────────────────────────────────────────────────────────────


class AsyncGRPOTrainer(Trainer):
    """
    Async GRPO trainer that extends HuggingFace ``Trainer``.

    Key overrides
    -------------
    * ``training_step`` – pipeline pattern: sync weights, submit next
      batch, get current batch, forward/loss/backward.
    * ``compute_loss`` – GRPO clipped surrogate + KL penalty.
    * ``get_train_dataloader`` – dummy dataloader (data from orchestrator).
    * ``log`` – injects GRPO metrics into the standard logging output.
    """

    def __init__(
        self,
        model: nn.Module,
        args: AsyncGRPOConfig,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        dataset: Any = None,
        reward_fn: Optional[Callable[[List[str], List[str]], List[float]]] = None,
        eval_dataset: Any = None,
        reward_details_fn: Optional[Callable] = None,
        log_router=None,
        **kwargs,
    ):
        # Suppress "estimate_tokens" warning from Trainer
        warnings_issued = getattr(model, "warnings_issued", None)
        if isinstance(warnings_issued, dict):
            warnings_issued["estimate_tokens"] = True

        self.log_router = log_router

        # ── LoRA: wrap model with PEFT *before* Trainer.__init__ ───────
        if args.use_lora and isinstance(args.lora_config, PeftConfig):
            model = get_peft_model(model, args.lora_config)

        # Gradient checkpointing with use_reentrant=True (the default in
        # some transformers versions) requires at least one forward-pass
        # input to have requires_grad=True.  Since input_ids and
        # attention_mask are integer tensors, we register a hook on the
        # embedding layer so its output carries requires_grad=True.
        if args.gradient_checkpointing:
            model.enable_input_require_grads()

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
            self.processing_class.pad_token_id = (
                self.processing_class.eos_token_id
            )

        self._trainer_device = torch.device(f"cuda:{args.trainer_gpu_id}")

        # ── Dataset (injected from environment or fallback to GSM8K) ────
        if dataset is not None:
            self.env_dataset = dataset
        else:
            # Backward compat: default to GSM8K if no dataset provided
            from rl.sync_rl.data import GSM8KDataset
            logger.info("No dataset provided, falling back to GSM8K...")
            self.env_dataset = GSM8KDataset(
                split=args.dataset_split,
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
            )
        logger.info(f"  {len(self.env_dataset)} training examples loaded")

        # ── Reward function (injected from environment or fallback) ────
        if reward_fn is not None:
            self.reward_fn = reward_fn
        else:
            # Backward compat: default to GSM8K reward
            from rl.sync_rl.reward import compute_rewards_batch
            logger.info("No reward_fn provided, falling back to GSM8K rewards...")
            self.reward_fn = compute_rewards_batch

        # ── vLLM client (NCCL weight sync) ─────────────────────────────
        host = args.vllm_server_host
        port = args.vllm_server_port
        logger.info(
            f"Connecting to vLLM server at {host}:{port}..."
        )
        self.vllm_client = VLLMClient(
            host=host,
            port=port,
            group_port=args.group_port,
            connection_timeout=args.vllm_server_timeout,
        )
        self.vllm_client.init_communicator()
        logger.info("NCCL communicator initialised")

        # ── Orchestrator (async batch generation) ──────────────────────
        vllm_base_url = f"http://{host}:{port}/v1"
        self.orchestrator = Orchestrator(
            server_base_url=vllm_base_url,
            max_concurrent=args.max_concurrent,
            client_timeout=args.generation_timeout,
            model_name=args.model_name,
            num_generations=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            dataset=self.env_dataset,
            reward_fn=self.reward_fn,
            tokenizer=self.processing_class,
            max_prompt_length=args.max_prompt_length,
            generation_timeout=args.generation_timeout,
            inflight_weight_updates=args.inflight_weight_updates,
            max_off_policy_steps=args.max_off_policy_steps,
            continuous_batching=args.continuous_batching,
            pool_size=args.pool_size,
            log_router=log_router,
            algorithm=args.algorithm,
            dapo_overlong_max_length=args.dapo_overlong_max_length,
            dapo_overlong_cache=args.dapo_overlong_cache,
        )
        self.orchestrator.start()
        if not args.continuous_batching:
            self.orchestrator.submit_batch(0)  # kick off first batch
        logger.info(
            "Orchestrator started"
            + (" (continuous batching, pool_size="
               f"{args.pool_size})" if args.continuous_batching
               else ", first batch submitted")
        )
        if args.inflight_weight_updates:
            logger.info(
                f"In-flight weight updates ENABLED "
                f"(max_off_policy_steps={args.max_off_policy_steps})"
            )

        # ── Evaluator (periodic eval on held-out test set) ──────────────
        if args.eval_steps > 0 and eval_dataset is not None and len(eval_dataset) > 0:
            eval_output_dir = os.path.join(args.output_dir, "evals")
            self.evaluator = Evaluator(
                server_base_url=vllm_base_url,
                model_name=args.model_name,
                eval_dataset=eval_dataset,
                reward_details_fn=reward_details_fn or (
                    lambda c, g: {"total": self.reward_fn([c], [g])[0]}
                ),
                num_problems=args.eval_num_problems,
                temperature=args.eval_temperature,
                max_new_tokens=args.eval_max_new_tokens,
                output_dir=eval_output_dir,
                log_router=log_router,
            )
            logger.info(
                f"Evaluator enabled: every {args.eval_steps} steps, "
                f"{args.eval_num_problems} problems, saved to {eval_output_dir}"
            )
        else:
            self.evaluator = None
            if args.eval_steps > 0:
                logger.warning(
                    "eval_steps > 0 but no eval_dataset provided – "
                    "evaluation disabled"
                )

        # ── Metric accumulator ─────────────────────────────────────────
        self._grpo_metrics: Dict[str, list] = defaultdict(list)

        # ── Rollout buffer for wandb completions table ─────────────────
        self._grpo_rollouts: Optional[Dict[str, Any]] = None

        # ── Remove noisy HF Trainer callbacks when log_router is active ─
        if self.log_router:
            from transformers.trainer_callback import (
                PrinterCallback,
                ProgressCallback,
            )
            self.remove_callback(PrinterCallback)
            self.remove_callback(ProgressCallback)

    # ================================================================
    #  Prevent DataParallel wrapping
    # ================================================================

    def _wrap_model(self, model, training=True, dataloader=None):
        """Skip DataParallel wrapping – we manage GPU placement manually."""
        return model

    # ================================================================
    #  Optimizer – AdamW
    # ================================================================

    def create_optimizer(self):
        """
        Create AdamW optimizer for all trainable parameters.

        Uses adam_beta1, adam_beta2, learning_rate, and weight_decay from
        the training config (AsyncGRPOConfig / TrainingArguments).
        """
        if self.optimizer is not None:
            return self.optimizer

        model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model

        # For PEFT/LoRA models, only optimize trainable parameters
        if is_peft_model(model):
            params = [p for p in model.parameters() if p.requires_grad]
        else:
            params = list(model.parameters())

        logger.info(f"Optimizer: AdamW ({len(params)} trainable params)")

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )

        return self.optimizer

    # ================================================================
    #  Dataloader – dummy (real data comes from the orchestrator)
    # ================================================================

    def get_train_dataloader(self) -> DataLoader:
        """Dummy dataloader whose length equals ``max_steps``."""

        class _StepsDataset(Dataset):
            def __init__(self, n: int):
                self.n = n

            def __len__(self) -> int:
                return self.n

            def __getitem__(self, idx: int) -> dict:
                return {"labels": 0}

        return DataLoader(_StepsDataset(self.args.max_steps))

    # ================================================================
    #  Core: training_step
    # ================================================================

    def training_step(
        self,
        model: nn.Module,
        inputs: Any = None,
        num_items_in_batch: Any = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Async GRPO training step:

        1. Sync updated weights to vLLM server (skip first step)
        2. Submit next batch for generation (pipeline)
        3. Get current batch from orchestrator (blocks if not ready)
           3b. If batch is empty (all prompts discarded), skip step
        4. Single forward pass -> trainer log probs (with gradients)
        5. GRPO loss (clipped surrogate + KL penalty)
        6. accelerator.backward(loss)
        7. Return detached loss
        """
        # ── 1. Weight sync ─────────────────────────────────────────────
        t0 = time.time()
        if self.state.global_step > 0:
            self._sync_vllm_weights()
        t_sync = time.time() - t0

        # ── 2. Submit next batch (pipeline; no-op for continuous batching)
        self.orchestrator.submit_batch(self.state.global_step + 1)

        # ── 3. Get current batch ───────────────────────────────────────
        batch: Batch = self.orchestrator.get_batch(self.state.global_step)

        # ── 3b. Handle empty batch (all prompts off-policy) ────────────
        if batch.num_valid_prompts == 0:
            msg = (
                f"step {self.state.global_step}: all prompts discarded "
                f"(off-policy v{batch.policy_version_min}->"
                f"v{batch.policy_version_max}) – skipping"
            )
            logger.warning(msg)
            if self.log_router:
                self.log_router.log_trainer(f"[yellow]{msg}[/yellow]")

            # Record the discard metrics even for skipped steps
            for key, value in batch.metrics.items():
                self._grpo_metrics[key].append(value)
            self._grpo_metrics["train/sync_s"].append(t_sync)
            self._grpo_metrics["train/skipped_step"].append(1.0)

            # Return zero loss (no gradients to accumulate)
            zero_loss = torch.tensor(0.0, device=self.accelerator.device)
            zero_loss.requires_grad_(True)
            self.accelerator.backward(zero_loss)
            return zero_loss.detach()

        model.train()
        device = self.accelerator.device

        # Move batch tensors to the training device
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        loss_mask = batch.loss_mask.to(device)
        inference_logprobs = batch.inference_logprobs.to(device)
        advantages = batch.advantages.to(device)

        # ── 4 + 5 + 6. Micro-batched forward + loss + backward ────────
        t0 = time.time()
        torch.cuda.empty_cache()

        total_seqs = input_ids.shape[0]
        micro_bs = self.args.micro_batch_size
        num_micro = (total_seqs + micro_bs - 1) // micro_bs
        inv_total = 1.0 / float(total_seqs)

        total_loss = torch.zeros((), device=device)
        accumulated_stats: Dict[str, float] = defaultdict(float)
        total_masked_tokens = 0
        total_mask_elements = 0
        total_entropy_sum = 0.0
        total_entropy_tokens = 0

        for i in range(num_micro):
            s = i * micro_bs
            e = min(s + micro_bs, total_seqs)
            mb_size = e - s

            mb_logprobs, mb_entropies = self._get_logprobs(
                model, input_ids[s:e], attention_mask[s:e],
            )
            mb_loss_mask = loss_mask[s:e, 1:]

            # Accumulate masked entropy for logging
            with torch.no_grad():
                masked_ent = (mb_entropies * mb_loss_mask).sum().item()
                ent_tokens = mb_loss_mask.sum().item()
                total_entropy_sum += masked_ent
                total_entropy_tokens += ent_tokens

            with self.compute_loss_context_manager():
                mb_loss, mb_stats = self.compute_loss(
                    model,
                    {
                        "trainer_logprobs": mb_logprobs,
                        "inference_logprobs": inference_logprobs[s:e],
                        "advantages": advantages[s:e],
                        "loss_mask": mb_loss_mask,
                    },
                    return_outputs=True,
                )

            # Scale loss by micro-batch fraction for correct gradient averaging
            scale = float(mb_size) * inv_total
            self.accelerator.backward(mb_loss * scale)

            total_loss = total_loss + mb_loss.detach() * scale
            for k, v in mb_stats.items():
                accumulated_stats[k] += v * scale

            # Track mask stats across micro-batches
            total_masked_tokens += int(mb_loss_mask.sum().item())
            total_mask_elements += mb_loss_mask.numel()

            # Free memory between micro-batches
            del mb_logprobs, mb_entropies, mb_loss
            torch.cuda.empty_cache()

        loss = total_loss
        stats = dict(accumulated_stats)
        t_train = time.time() - t0

        # ── Compute mean entropy over completion tokens ─────────────────
        mean_entropy = (
            total_entropy_sum / max(total_entropy_tokens, 1)
        )
        stats["train/entropy"] = mean_entropy

        # ── Accumulate GRPO metrics ────────────────────────────────────
        for key, value in batch.metrics.items():
            self._grpo_metrics[key].append(value)

        stats["train/sync_s"] = t_sync
        stats["train/train_s"] = t_train
        stats["train/skipped_step"] = 0.0
        stats["train/masked_fraction"] = 1.0 - (
            total_masked_tokens / max(total_mask_elements, 1)
        )

        for key, value in stats.items():
            self._grpo_metrics[key].append(value)

        # ── Store rollout for wandb ────────────────────────────────────
        self._grpo_rollouts = {
            "prompts": batch.prompts,
            "completions": batch.completions,
            "rewards": batch.rewards,
        }

        # ── Buffer trainer line (flushed in log() with lr/grad_norm) ───
        if self.log_router:
            reward_mean = batch.metrics.get('rewards/mean', 0)
            total_steps = self.args.max_steps
            ver_info = (
                f" | v{batch.policy_version_min}-"
                f"v{batch.policy_version_max}"
            ) if self.args.inflight_weight_updates else ""
            discard_info = ""
            n_disc = batch.metrics.get("train/num_discarded_prompts", 0)
            if n_disc > 0:
                discard_info = (
                    f" | [yellow]disc {int(n_disc)}[/yellow]"
                )
            self._pending_trainer_line = (
                f"[bold white]step {self.state.global_step}/{total_steps}"
                f"[/bold white] | "
                f"loss [bright_yellow]{loss.item():.4f}[/bright_yellow] | "
                f"reward [bright_green]{reward_mean:.3f}[/bright_green] | "
                f"ent [bright_blue]{mean_entropy:.3f}[/bright_blue]"
                f"{ver_info}{discard_info}"
            )

        # ── Console summary (fallback) ─────────────────────────────────
        if not self.log_router:
            logger.info(
                f"step {self.state.global_step} | "
                f"loss={loss.item():.4f} | "
                f"reward={batch.metrics.get('rewards/mean', 0):.3f} | "
                f"entropy={mean_entropy:.3f} | "
                f"kl={stats.get('loss/kl', 0):.4f} | "
                f"gen={batch.generation_time:.1f}s | "
                f"train={t_train:.1f}s | sync={t_sync:.1f}s"
            )

        # ── 7. Return ──────────────────────────────────────────────────
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
        Compute the RL loss (GRPO, CISPO, or DAPO depending on config).

        Delegates to the standalone loss function from ``rl/algorithms/``.
        """
        if self.args.algorithm == "prime":
            loss, stats = prime_loss(
                trainer_log_probs=inputs["trainer_logprobs"],
                inference_log_probs=inputs["inference_logprobs"],
                advantages=inputs["advantages"],
                completion_mask=inputs["loss_mask"],
                token_mask_low=self.args.prime_token_mask_low,
                token_mask_high=self.args.prime_token_mask_high,
                max_log_ratio=self.args.max_log_ratio,
            )
        elif self.args.algorithm == "cispo":
            loss, stats = cispo_loss(
                trainer_log_probs=inputs["trainer_logprobs"],
                inference_log_probs=inputs["inference_logprobs"],
                advantages=inputs["advantages"],
                completion_mask=inputs["loss_mask"],
                epsilon_lower=self.args.cispo_epsilon_low,
                epsilon_upper=self.args.cispo_epsilon_high,
                max_log_ratio=self.args.max_log_ratio,
            )
        elif self.args.algorithm == "dapo":
            loss, stats = dapo_loss(
                trainer_log_probs=inputs["trainer_logprobs"],
                inference_log_probs=inputs["inference_logprobs"],
                advantages=inputs["advantages"],
                completion_mask=inputs["loss_mask"],
                epsilon_lower=self.args.epsilon_lower,
                epsilon_upper=self.args.epsilon_upper,
                max_log_ratio=self.args.max_log_ratio,
            )
        else:
            loss, stats = grpo_loss(
                trainer_log_probs=inputs["trainer_logprobs"],
                inference_log_probs=inputs["inference_logprobs"],
                advantages=inputs["advantages"],
                completion_mask=inputs["loss_mask"],
                epsilon_lower=self.args.epsilon_lower,
                epsilon_upper=self.args.epsilon_upper,
                beta=self.args.beta,
                max_log_ratio=self.args.max_log_ratio,
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
    ) -> tuple:
        """
        Compute per-token log probabilities and entropy for a full sequence.

        Processes in mini-batches of ``chunk_size`` to avoid OOM.

        Returns:
            logprobs:  (B, L-1) – shifted so that logprobs[:, t]
                       corresponds to log P(input_ids[:, t+1] | context).
            entropies: (B, L-1) – per-token Shannon entropy (nats) at each
                       shifted position, computed from the full vocabulary
                       distribution.
        """
        all_logprobs = []
        all_entropies = []
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
            logits = logits[:, :-1, :]    # (chunk, L-1, V)
            targets = chunk_ids[:, 1:]    # (chunk, L-1)

            chunk_logprobs = selective_log_softmax(logits, targets)
            all_logprobs.append(chunk_logprobs)

            # Compute per-token entropy (no grad needed)
            with torch.no_grad():
                chunk_ent = entropy_from_logits(logits)  # (chunk, L-1)
            all_entropies.append(chunk_ent)

            del logits  # free the large tensor immediately

        return (
            torch.cat(all_logprobs, dim=0),    # (B, L-1)
            torch.cat(all_entropies, dim=0),   # (B, L-1)
        )

    # ================================================================
    #  vLLM weight sync (NCCL-based)
    # ================================================================

    def _sync_vllm_weights(self) -> None:
        """
        Push updated policy weights to the vLLM server via NCCL.

        Two modes:
          - Legacy (inflight_weight_updates=False): waits for any in-flight
            generation to finish before pushing weights.
          - In-flight (inflight_weight_updates=True, PipelineRL-style):
            pushes weights immediately -- vLLM briefly pauses its workers
            to receive each parameter via NCCL, then resumes generation.
            After sync, increments ``orchestrator.current_policy_version``
            so the orchestrator can track per-prompt off-policyness.

        When continuous batching is active the pool is paused before sync
        and resumed after, so that in-flight generation drains and vLLM
        workers are free to process ``collective_rpc`` calls without
        contention from concurrent generation requests.

        For PEFT/LoRA models the adapters are temporarily merged into
        the base weights so vLLM receives a standard state dict.
        """
        inflight = self.args.inflight_weight_updates

        # Only pause pool if we're about to sync weights (skip at step 0)
        should_pause = self.args.continuous_batching and self.state.global_step > 0
        
        # ── Pause continuous pool to eliminate worker contention ───────
        if should_pause:
            self.orchestrator.pause_pool()
            # Wait for in-flight generation to drain (max 10s)
            for _ in range(100):
                if self.orchestrator._pool_in_flight <= 0:
                    break
                time.sleep(0.1)
            if self.log_router:
                remaining = self.orchestrator._pool_in_flight
                self.log_router.log_inference(
                    f"[cyan]Pool paused for weight sync "
                    f"({remaining} tasks still in flight)[/cyan]"
                )

        if not inflight:
            # Legacy mode: wait for generation to finish before syncing.
            #
            # Two different wait strategies depending on batching mode:
            #   - Continuous batching: is_generating stays True for the
            #     lifetime of the pool (set at startup, cleared at shutdown),
            #     so we can only rely on _pool_in_flight reaching 0 after
            #     the pool has been paused above.
            #   - Batch-at-a-time: is_generating toggles per batch, so we
            #     wait for it to become False.
            waits = 0
            if self.args.continuous_batching:
                while self.orchestrator._pool_in_flight > 0:
                    time.sleep(0.5)
                    waits += 1
                    if waits % 10 == 0:
                        if self.log_router:
                            self.log_router.log_inference(
                                f"[yellow]Waiting for pool to drain before syncing "
                                f"(pool_in_flight={self.orchestrator._pool_in_flight})[/yellow]"
                            )
                        else:
                            logger.info(
                                f"Waiting for pool to drain "
                                f"(pool_in_flight={self.orchestrator._pool_in_flight})..."
                            )
            else:
                while self.orchestrator.is_generating:
                    time.sleep(0.5)
                    waits += 1
                    if waits % 10 == 0:
                        if self.log_router:
                            self.log_router.log_inference(
                                f"[yellow]Waiting for generation to finish before syncing "
                                f"(is_generating={self.orchestrator.is_generating})[/yellow]"
                            )
                        else:
                            logger.info(
                                "Waiting for generation to finish before syncing..."
                            )

        unwrapped = self.accelerator.unwrap_model(self.model)

        ver = self.orchestrator.current_policy_version
        if self.log_router:
            mode_tag = "in-flight" if inflight else "blocking"
            self.log_router.log_inference(
                f"[cyan]Starting weight sync to vLLM server "
                f"({mode_tag}, v{ver} -> v{ver + 1})...[/cyan]"
            )

        if is_peft_model(unwrapped):
            # Merge LoRA adapters -> base weights
            unwrapped.merge_adapter()

            # Build a clean state dict with original parameter names
            for name, param in unwrapped.named_parameters():
                name = name.removeprefix("base_model.model.").replace(
                    ".base_layer", "",
                )
                if unwrapped.prefix in name:
                    continue
                if "original_module" in name:
                    continue
                name = name.replace("modules_to_save.default.", "")
                self.vllm_client.update_named_param(name, param.data)

            # Unmerge so LoRA training can continue
            unwrapped.unmerge_adapter()

            if self.log_router:
                self.log_router.log_inference(
                    "[cyan]Synced merged LoRA weights to vLLM[/cyan]"
                )
            else:
                logger.info("Synced merged LoRA weights -> vLLM")
        else:
            # Full model: sync each parameter
            for name, param in unwrapped.named_parameters():
                self.vllm_client.update_named_param(name, param.data)

            if self.log_router:
                self.log_router.log_inference(
                    "[cyan]Synced policy weights to vLLM[/cyan]"
                )
            else:
                logger.info("Synced policy weights -> vLLM")

        # Reset KV cache + wait for background tasks
        self.vllm_client.reset_prefix_cache()
        while self.vllm_client.get_num_background_tasks() > 0:
            time.sleep(0.5)

        # Bump the policy version so the orchestrator knows new weights
        # are live.  This is a simple int write, atomic under CPython GIL.
        self.orchestrator.current_policy_version += 1
        if self.log_router:
            self.log_router.log_inference(
                f"[cyan]Policy version -> "
                f"{self.orchestrator.current_policy_version}[/cyan]"
            )

        # ── Resume continuous pool ────────────────────────────────────
        if self.args.continuous_batching:
            self.orchestrator.resume_pool()
            if self.log_router:
                self.log_router.log_inference(
                    "[cyan]Pool resumed after weight sync[/cyan]"
                )

    # ================================================================
    #  Training loop cleanup
    # ================================================================

    def _inner_training_loop(self, *args, **kwargs):
        """Override to stop orchestrator when training ends."""
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            self.orchestrator.stop()

    # ================================================================
    #  Logging (merge GRPO metrics into Trainer logs)
    # ================================================================

    # ── Essential metrics whitelist ────────────────────────────────────
    _ESSENTIAL_METRICS = {
        "rewards/mean",
        "rewards/std",
        "loss/total",
        "loss/pg_loss",
        "loss/kl",                       # GRPO only (absent for CISPO)
        "train/completion_len_mean",
        "train/completion_len_max",
        "train/entropy",
        "rewards/zero_var_group_frac",
        "train/dapo_dynamic_filtered",   # DAPO: prompts filtered (all-correct/all-wrong)
        "train/dapo_overlong_penalized", # DAPO: completions penalized for length
        "train/clip_fraction",
        "train/importance_ratio",
        "train/mask_fraction",            # Prime: fraction of tokens masked out
        "train/mask_fraction_low",        # Prime: masked because ratio too low
        "train/mask_fraction_high",       # Prime: masked because ratio too high
        "loss/mismatch_kl",               # Prime: KL between old and new policy
        "learning_rate",
        "grad_norm",
    }

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        """
        Override to inject accumulated GRPO metrics into every log call
        and optionally log a wandb completions table.

        Only essential metrics are forwarded to wandb / the HF logger
        to keep dashboards clean.
        """
        # Average the accumulated GRPO metrics
        grpo_avg = {
            key: sum(vals) / len(vals)
            for key, vals in self._grpo_metrics.items()
            if vals
        }
        all_logs = {**logs, **grpo_avg}

        # Filter to only essential metrics (+ any eval/* metrics)
        logs = {
            k: v for k, v in all_logs.items()
            if k in self._ESSENTIAL_METRICS or k.startswith("eval/")
        }

        # ── wandb completions table ────────────────────────────────────
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
                                columns=[
                                    "step", "prompt", "completion", "reward",
                                ],
                                data=table_data,
                            )
                        },
                        commit=False,
                    )
            except ImportError:
                pass

        # ── Print evaluation examples every N steps ────────────────────
        if (
            self._grpo_rollouts is not None
            and self.state.global_step > 0
            and self.state.global_step % self.args.eval_display_steps == 0
        ):
            rollouts = self._grpo_rollouts
            print_examples(
                prompts=rollouts["prompts"],
                completions=rollouts["completions"],
                rewards=rollouts["rewards"],
                step=self.state.global_step,
                num_samples=self.args.eval_num_samples,
                num_generations=self.args.num_generations,
            )

        # ── Periodic evaluation on held-out test set ─────────────────
        if (
            self.evaluator is not None
            and self.state.global_step > 0
            and self.state.global_step % self.args.eval_steps == 0
        ):
            eval_summary = self.evaluator.run_eval(self.state.global_step)
            # Log eval metrics to wandb alongside training metrics
            if eval_summary:
                eval_logs = {
                    f"eval/{k}": v
                    for k, v in eval_summary.items()
                    if isinstance(v, (int, float))
                }
                logs.update(eval_logs)

        # ── Log router: combine buffered step line + lr/grad_norm ──────
        if self.log_router and hasattr(self, "_pending_trainer_line"):
            line = self._pending_trainer_line
            lr = logs.get("learning_rate")
            gn = logs.get("grad_norm")
            if lr is not None:
                line += f" | lr [cyan]{lr:.2e}[/cyan]"
            if gn is not None:
                line += f" | gn [magenta]{gn:.4f}[/magenta]"
            self.log_router.log_trainer(line)
            del self._pending_trainer_line

        super().log(logs, start_time)
        self._grpo_metrics.clear()
