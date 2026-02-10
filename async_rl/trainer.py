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
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from accelerate.utils import is_peft_model
from peft import PeftConfig, get_peft_model
from transformers.trainer import Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Parent dir for cross-package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl.grpo import grpo_loss

from async_rl.config import AsyncGRPOConfig
from async_rl.client import VLLMClient
from async_rl.display import print_examples
from async_rl.orchestrator import Orchestrator, Batch
from sync_rl.data import GSM8KDataset

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

        # ── GSM8K dataset ──────────────────────────────────────────────
        logger.info("Loading GSM8K dataset...")
        self.gsm8k_dataset = GSM8KDataset(
            split=args.dataset_split,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
        )
        logger.info(f"  {len(self.gsm8k_dataset)} training examples loaded")

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
            dataset=self.gsm8k_dataset,
            tokenizer=self.processing_class,
            max_prompt_length=args.max_prompt_length,
            generation_timeout=args.generation_timeout,
            inflight_weight_updates=args.inflight_weight_updates,
            max_off_policy_steps=args.max_off_policy_steps,
            continuous_batching=args.continuous_batching,
            pool_size=args.pool_size,
            log_router=log_router,
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
    #  Optimizer – use Muon for matrix params, AdamW for others
    # ================================================================

    def create_optimizer(self):
        """
        Create optimizer using Muon for 2D weight matrices and AdamW for
        embeddings, heads, biases, and other non-2D parameters.
        
        Based on the MuonWithAuxAdam pattern from:
        https://github.com/KellerJordan/modded-nanogpt
        """
        from muon import SingleDeviceMuon
        
        if self.optimizer is not None:
            return self.optimizer

        model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
        
        # Collect parameters by type
        # For PEFT/LoRA models, only optimize trainable parameters
        if is_peft_model(model):
            all_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        else:
            all_params = [(n, p) for n, p in model.named_parameters()]
        
        # Separate parameters:
        # - Muon: 2D parameters (weight matrices) not in embeddings/heads
        # - AdamW: embeddings, heads, biases, layer norms, and other non-2D params
        muon_params = []
        adamw_params = []
        
        for name, param in all_params:
            # Use AdamW for:
            # - embeddings
            # - output head (lm_head)
            # - biases
            # - layer norms
            # - any non-2D parameters
            if ("embed" in name or 
                "lm_head" in name or
                "bias" in name or
                "norm" in name or
                param.ndim < 2):
                adamw_params.append(param)
            else:
                # Use Muon for 2D weight matrices in hidden layers
                muon_params.append(param)
        
        logger.info(f"Optimizer: Muon ({len(muon_params)} params) + "
                   f"AdamW ({len(adamw_params)} params)")
        
        # Create a combined optimizer using SingleDeviceMuon and AdamW
        # We'll use a custom class that manages both optimizers
        class SingleDeviceMuonWithAdamW(torch.optim.Optimizer):
            """
            Combined optimizer using SingleDeviceMuon for matrix params and AdamW for others.
            Single-device version that doesn't require torch.distributed.
            """
            def __init__(self, muon_params, adamw_params, lr, weight_decay, momentum=0.95, 
                        betas=(0.9, 0.999), eps=1e-8):
                # Create sub-optimizers
                self.muon_opt = SingleDeviceMuon(muon_params, lr=lr, weight_decay=weight_decay, 
                                                 momentum=momentum) if muon_params else None
                self.adamw_opt = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay,
                                                   betas=betas, eps=eps) if adamw_params else None
                
                # Combine param groups for the parent class
                all_param_groups = []
                if self.muon_opt:
                    all_param_groups.extend(self.muon_opt.param_groups)
                if self.adamw_opt:
                    all_param_groups.extend(self.adamw_opt.param_groups)
                
                # Initialize parent with dummy defaults
                self.defaults = {}
                self.param_groups = all_param_groups
                # Combine states
                self.state = {}
                if self.muon_opt:
                    self.state.update(self.muon_opt.state)
                if self.adamw_opt:
                    self.state.update(self.adamw_opt.state)
            
            @torch.no_grad()
            def step(self, closure=None):
                """Step both optimizers."""
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()
                
                if self.muon_opt:
                    self.muon_opt.step()
                if self.adamw_opt:
                    self.adamw_opt.step()
                
                return loss
            
            def zero_grad(self, set_to_none=False):
                """Zero gradients for both optimizers."""
                if self.muon_opt:
                    self.muon_opt.zero_grad(set_to_none=set_to_none)
                if self.adamw_opt:
                    self.adamw_opt.zero_grad(set_to_none=set_to_none)
            
            def state_dict(self):
                """Combine state dicts from both optimizers."""
                state_dict = {
                    'muon': self.muon_opt.state_dict() if self.muon_opt else None,
                    'adamw': self.adamw_opt.state_dict() if self.adamw_opt else None,
                }
                return state_dict
            
            def load_state_dict(self, state_dict):
                """Load state dicts for both optimizers."""
                if self.muon_opt and state_dict.get('muon'):
                    self.muon_opt.load_state_dict(state_dict['muon'])
                if self.adamw_opt and state_dict.get('adamw'):
                    self.adamw_opt.load_state_dict(state_dict['adamw'])
        
        self.optimizer = SingleDeviceMuonWithAdamW(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            momentum=0.95,
            betas=(0.9, 0.999),
            eps=1e-8
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

        # ── 4. Forward pass -> trainer logprobs ────────────────────────
        t0 = time.time()
        torch.cuda.empty_cache()
        trainer_logprobs = self._get_logprobs(model, input_ids, attention_mask)

        # Shift loss_mask to match the shifted logprobs (logits[:, :-1])
        loss_mask_shifted = loss_mask[:, 1:]

        # ── 5 + 6. GRPO loss + backward ───────────────────────────────
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
        for key, value in batch.metrics.items():
            self._grpo_metrics[key].append(value)

        stats["train/sync_s"] = t_sync
        stats["train/train_s"] = t_train
        stats["train/skipped_step"] = 0.0
        stats["train/masked_fraction"] = 1.0 - (
            loss_mask_shifted.sum() / loss_mask_shifted.numel()
        ).item()

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
                f"reward [bright_green]{reward_mean:.3f}[/bright_green]"
                f"{ver_info}{discard_info}"
            )

        # ── Console summary (fallback) ─────────────────────────────────
        if not self.log_router:
            logger.info(
                f"step {self.state.global_step} | "
                f"loss={loss.item():.4f} | "
                f"reward={batch.metrics.get('rewards/mean', 0):.3f} | "
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
        Compute the GRPO loss (clipped surrogate + KL penalty).

        Delegates to the standalone ``grpo_loss`` from ``rl/grpo.py``.
        """
        loss, stats = grpo_loss(
            trainer_log_probs=inputs["trainer_logprobs"],
            inference_log_probs=inputs["inference_logprobs"],
            advantages=inputs["advantages"],
            completion_mask=inputs["loss_mask"],
            epsilon=self.args.epsilon,
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

        Processes in mini-batches of ``chunk_size`` to avoid OOM.

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
            logits = logits[:, :-1, :]    # (chunk, L-1, V)
            targets = chunk_ids[:, 1:]    # (chunk, L-1)

            chunk_logprobs = selective_log_softmax(logits, targets)
            all_logprobs.append(chunk_logprobs)

            del logits  # free the large tensor immediately

        return torch.cat(all_logprobs, dim=0)  # (B, L-1)

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

        # ── Pause continuous pool to eliminate worker contention ───────
        if self.args.continuous_batching:
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
            # Legacy mode: wait for generation to finish
            waits = 0
            while self.orchestrator.is_generating:
                time.sleep(0.5)
                waits += 1
                if waits % 10 == 0:
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
