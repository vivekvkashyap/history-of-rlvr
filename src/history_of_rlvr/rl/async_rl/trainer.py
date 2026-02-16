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

from history_of_rlvr.rl.algorithms.grpo.grpo import grpo_loss
from history_of_rlvr.rl.algorithms.dr_grpo.dr_grpo import dr_grpo_loss
from history_of_rlvr.rl.algorithms.cispo.cispo import cispo_loss
from history_of_rlvr.rl.algorithms.dapo.dapo import dapo_loss
from history_of_rlvr.rl.algorithms.prime.prime import prime_loss
from history_of_rlvr.rl.algorithms.gspo.gspo import gspo_loss

from history_of_rlvr.rl.async_rl.config import AsyncGRPOConfig
from history_of_rlvr.rl.async_rl.client import VLLMClient
from history_of_rlvr.rl.async_rl.display import print_examples
from history_of_rlvr.rl.async_rl.evaluator import Evaluator
from history_of_rlvr.rl.async_rl.orchestrator import Orchestrator, Batch

logger = logging.getLogger(__name__)


def selective_log_softmax(
    logits: torch.Tensor, index: torch.Tensor,
) -> torch.Tensor:
    per_token_logps = []
    for row_logits, row_labels in zip(logits, index):
        row_logps = torch.nn.functional.log_softmax(row_logits, dim=-1)
        row_per_token = row_logps.gather(
            dim=-1, index=row_labels.unsqueeze(-1),
        ).squeeze(-1)
        per_token_logps.append(row_per_token)
    return torch.stack(per_token_logps)


def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    original_shape = logits.shape[:-1]
    num_classes = logits.shape[-1]

    flat_logits = logits.reshape(-1, num_classes)

    entropies = []
    for chunk in flat_logits.split(chunk_size, dim=0):
        logps = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        entropies.append(chunk_entropy)

    entropies = torch.cat(entropies, dim=0)
    return entropies.reshape(original_shape)


class AsyncGRPOTrainer(Trainer):
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
        warnings_issued = getattr(model, "warnings_issued", None)
        if isinstance(warnings_issued, dict):
            warnings_issued["estimate_tokens"] = True

        self.log_router = log_router

        if args.use_lora and isinstance(args.lora_config, PeftConfig):
            model = get_peft_model(model, args.lora_config)

        if args.gradient_checkpointing:
            model.enable_input_require_grads()

        super().__init__(
            model=model,
            args=args,
            processing_class=processing_class,
            **kwargs,
        )

        assert isinstance(self.processing_class, PreTrainedTokenizerBase)
        if self.processing_class.pad_token is None:
            self.processing_class.pad_token = self.processing_class.eos_token
        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token_id = (
                self.processing_class.eos_token_id
            )

        self._trainer_device = torch.device(f"cuda:{args.trainer_gpu_id}")

        if dataset is not None:
            self.env_dataset = dataset
        else:
            from rl.sync_rl.data import GSM8KDataset
            logger.info("No dataset provided, falling back to GSM8K...")
            self.env_dataset = GSM8KDataset(
                split=args.dataset_split,
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
            )
        logger.info(f"  {len(self.env_dataset)} training examples loaded")

        if reward_fn is not None:
            self.reward_fn = reward_fn
        else:
            from rl.sync_rl.reward import compute_rewards_batch
            logger.info("No reward_fn provided, falling back to GSM8K rewards...")
            self.reward_fn = compute_rewards_batch

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
            normalize_advantages_by_std=args.normalize_advantages_by_std,
            overlong_max_length=args.overlong_max_length,
            overlong_cache=args.overlong_cache,
            use_overlong_penalty=args.use_overlong_penalty,
            use_dynamic_sampling=args.use_dynamic_sampling,
        )
        self.orchestrator.start()
        if not args.continuous_batching:
            self.orchestrator.submit_batch(0)
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

        self._grpo_metrics: Dict[str, list] = defaultdict(list)
        self._grpo_rollouts: Optional[Dict[str, Any]] = None
        if self.log_router:
            from transformers.trainer_callback import (
                PrinterCallback,
                ProgressCallback,
            )
            self.remove_callback(PrinterCallback)
            self.remove_callback(ProgressCallback)

        self._log_config_summary()

    def _log_config_summary(self):
        cfg = self.args
        sep = "=" * 60
        lines = [
            sep,
            "RUN CONFIGURATION",
            sep,
            f"  algorithm:            {cfg.algorithm}",
            f"  model:                {cfg.model_name}",
            f"  use_lora:             {cfg.use_lora}" + (f" (rank={cfg.lora_rank})" if cfg.use_lora else ""),
            f"  learning_rate:        {cfg.learning_rate}",
            f"  weight_decay:         {cfg.weight_decay}",
            f"  max_grad_norm:        {cfg.max_grad_norm}",
            f"  lr_scheduler:         {cfg.lr_scheduler_type}",
            f"  warmup_ratio:         {cfg.warmup_ratio}",
            f"  max_steps:            {cfg.max_steps}",
            f"  batch_size:           {cfg.batch_size}",
            f"  micro_batch_size:     {cfg.micro_batch_size}",
            f"  num_generations:      {cfg.num_generations}",
            f"  max_new_tokens:       {cfg.max_new_tokens}",
            f"  temperature:          {cfg.temperature}",
            f"  normalize_adv_std:    {cfg.normalize_advantages_by_std}",
        ]

        lines += [
            f"  epsilon_lower:        {cfg.epsilon_lower}",
            f"  epsilon_upper:        {cfg.epsilon_upper}",
        ]
        if cfg.algorithm in ("grpo", "dr_grpo"):
            lines.append(f"  beta (KL):            {cfg.beta}")
        if cfg.algorithm == "dr_grpo":
            lines.append(f"  max_tokens (norm):    {cfg.max_new_tokens}")
        elif cfg.algorithm == "prime":
            lines += [
                f"  token_mask_low:       {cfg.prime_token_mask_low}",
                f"  token_mask_high:      {cfg.prime_token_mask_high}",
            ]
        lines.append(f"  loss_reduction:       {cfg.loss_reduction}")

        lines += [
            f"  max_log_ratio:        {cfg.max_log_ratio}",
            f"  continuous_batching:  {cfg.continuous_batching}" + (f" (pool_size={cfg.pool_size})" if cfg.continuous_batching else ""),
            f"  inflight_updates:     {cfg.inflight_weight_updates}" + (f" (max_off_policy={cfg.max_off_policy_steps})" if cfg.inflight_weight_updates else ""),
            f"  overlong_penalty:     {cfg.use_overlong_penalty}",
            f"  dynamic_sampling:     {cfg.use_dynamic_sampling}",
            f"  seed:                 {cfg.seed}",
            f"  bf16:                 {cfg.bf16}",
            f"  gradient_checkpoint:  {cfg.gradient_checkpointing}",
            f"  output_dir:           {cfg.output_dir}",
            sep,
        ]

        for line in lines:
            logger.info(line)
        if self.log_router:
            for line in lines:
                self.log_router.log_trainer(f"[dim]{line}[/dim]")

    def _wrap_model(self, model, training=True, dataloader=None):
        return model

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model

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

    def get_train_dataloader(self) -> DataLoader:
        class _StepsDataset(Dataset):
            def __init__(self, n: int):
                self.n = n

            def __len__(self) -> int:
                return self.n

            def __getitem__(self, idx: int) -> dict:
                return {"labels": 0}

        return DataLoader(_StepsDataset(self.args.max_steps))

    def training_step(
        self,
        model: nn.Module,
        inputs: Any = None,
        num_items_in_batch: Any = None,
        **kwargs,
    ) -> torch.Tensor:
        t0 = time.time()
        if self.state.global_step > 0:
            self._sync_vllm_weights()
        t_sync = time.time() - t0

        self.orchestrator.submit_batch(self.state.global_step + 1)

        batch: Batch = self.orchestrator.get_batch(self.state.global_step)

        if batch.num_valid_prompts == 0:
            msg = (
                f"step {self.state.global_step}: all prompts discarded "
                f"(off-policy v{batch.policy_version_min}->"
                f"v{batch.policy_version_max}) – skipping"
            )
            logger.warning(msg)
            if self.log_router:
                self.log_router.log_trainer(f"[yellow]{msg}[/yellow]")

            for key, value in batch.metrics.items():
                self._grpo_metrics[key].append(value)
            self._grpo_metrics["train/sync_s"].append(t_sync)
            self._grpo_metrics["train/skipped_step"].append(1.0)

            zero_loss = torch.tensor(0.0, device=self.accelerator.device)
            zero_loss.requires_grad_(True)
            self.accelerator.backward(zero_loss)
            return zero_loss.detach()

        model.train()
        device = self.accelerator.device

        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        loss_mask = batch.loss_mask.to(device)
        inference_logprobs = batch.inference_logprobs.to(device)
        advantages = batch.advantages.to(device)

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

            scale = float(mb_size) * inv_total
            self.accelerator.backward(mb_loss * scale)

            total_loss = total_loss + mb_loss.detach() * scale
            for k, v in mb_stats.items():
                accumulated_stats[k] += v * scale

            total_masked_tokens += int(mb_loss_mask.sum().item())
            total_mask_elements += mb_loss_mask.numel()

            del mb_logprobs, mb_entropies, mb_loss
            torch.cuda.empty_cache()

        loss = total_loss
        stats = dict(accumulated_stats)
        t_train = time.time() - t0

        mean_entropy = (
            total_entropy_sum / max(total_entropy_tokens, 1)
        )
        stats["train/entropy"] = mean_entropy

        step_time = t_train + t_sync
        tokens_per_second = total_masked_tokens / max(step_time, 1e-8)
        stats["train/throughput_tokens_per_sec"] = tokens_per_second
        stats["train/step_time_s"] = step_time
        stats["train/gen_time_s"] = batch.generation_time

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

        self._grpo_rollouts = {
            "prompts": batch.prompts,
            "completions": batch.completions,
            "rewards": batch.rewards,
        }

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

        return loss.detach()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ):
        common = dict(
            trainer_log_probs=inputs["trainer_logprobs"],
            inference_log_probs=inputs["inference_logprobs"],
            advantages=inputs["advantages"],
            completion_mask=inputs["loss_mask"],
            max_log_ratio=self.args.max_log_ratio,
            loss_reduction=self.args.loss_reduction,
        )

        if self.args.algorithm == "gspo":
            loss, stats = gspo_loss(
                trainer_log_probs=inputs["trainer_logprobs"],
                inference_log_probs=inputs["inference_logprobs"],
                advantages=inputs["advantages"],
                completion_mask=inputs["loss_mask"],
                epsilon_lower=self.args.epsilon_lower,
                epsilon_upper=self.args.epsilon_upper,
                max_log_ratio=self.args.max_log_ratio,
            )
        elif self.args.algorithm == "dr_grpo":
            loss, stats = dr_grpo_loss(
                trainer_log_probs=inputs["trainer_logprobs"],
                inference_log_probs=inputs["inference_logprobs"],
                advantages=inputs["advantages"],
                completion_mask=inputs["loss_mask"],
                epsilon_lower=self.args.epsilon_lower,
                epsilon_upper=self.args.epsilon_upper,
                beta=self.args.beta,
                max_log_ratio=self.args.max_log_ratio,
                max_tokens=self.args.max_new_tokens,
            )
        elif self.args.algorithm == "prime":
            loss, stats = prime_loss(
                **common,
                token_mask_low=self.args.prime_token_mask_low,
                token_mask_high=self.args.prime_token_mask_high,
            )
        elif self.args.algorithm == "cispo":
            loss, stats = cispo_loss(
                **common,
                epsilon_lower=self.args.epsilon_lower,
                epsilon_upper=self.args.epsilon_upper,
            )
        elif self.args.algorithm == "dapo":
            loss, stats = dapo_loss(
                **common,
                epsilon_lower=self.args.epsilon_lower,
                epsilon_upper=self.args.epsilon_upper,
            )
        else:
            loss, stats = grpo_loss(
                **common,
                epsilon_lower=self.args.epsilon_lower,
                epsilon_upper=self.args.epsilon_upper,
                beta=self.args.beta,
            )

        if return_outputs:
            return loss, stats
        return loss

    def _get_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int = 4,
    ) -> tuple:
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

            logits = logits[:, :-1, :]
            targets = chunk_ids[:, 1:]

            chunk_logprobs = selective_log_softmax(logits, targets)
            all_logprobs.append(chunk_logprobs)

            with torch.no_grad():
                chunk_ent = entropy_from_logits(logits)
            all_entropies.append(chunk_ent)

            del logits

        return (
            torch.cat(all_logprobs, dim=0),
            torch.cat(all_entropies, dim=0),
        )

    def _sync_vllm_weights(self) -> None:
        inflight = self.args.inflight_weight_updates
        should_pause = self.args.continuous_batching and self.state.global_step > 0
        if should_pause:
            self.orchestrator.pause_pool()
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
            unwrapped.merge_adapter()
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

            unwrapped.unmerge_adapter()

            if self.log_router:
                self.log_router.log_inference(
                    "[cyan]Synced merged LoRA weights to vLLM[/cyan]"
                )
            else:
                logger.info("Synced merged LoRA weights -> vLLM")
        else:
            for name, param in unwrapped.named_parameters():
                self.vllm_client.update_named_param(name, param.data)

            if self.log_router:
                self.log_router.log_inference(
                    "[cyan]Synced policy weights to vLLM[/cyan]"
                )
            else:
                logger.info("Synced policy weights -> vLLM")

        self.vllm_client.reset_prefix_cache()
        while self.vllm_client.get_num_background_tasks() > 0:
            time.sleep(0.5)

        self.orchestrator.current_policy_version += 1
        if self.log_router:
            self.log_router.log_inference(
                f"[cyan]Policy version -> "
                f"{self.orchestrator.current_policy_version}[/cyan]"
            )

        if self.args.continuous_batching:
            self.orchestrator.resume_pool()
            if self.log_router:
                self.log_router.log_inference(
                    "[cyan]Pool resumed after weight sync[/cyan]"
                )

    def _inner_training_loop(self, *args, **kwargs):
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            self.orchestrator.stop()

    _ESSENTIAL_METRICS = {
        "rewards/mean",
        "rewards/std",
        "loss/total",
        "loss/pg_loss",
        "loss/kl",
        "loss/mismatch_kl",
        "train/completion_len_mean",
        "train/completion_len_max",
        "train/entropy",
        "train/throughput_tokens_per_sec",
        "train/step_time_s",
        "train/gen_time_s",
        "rewards/zero_var_group_frac",
        "train/dynamic_filtered",
        "train/overlong_penalized",
        "train/clip_fraction",
        "train/importance_ratio",
        "train/mask_fraction",
        "train/mask_fraction_low",
        "train/mask_fraction_high",
        "train/seq_importance_ratio",
        "train/seq_clip_fraction",
        "learning_rate",
        "grad_norm",
    }

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        grpo_avg = {
            key: sum(vals) / len(vals)
            for key, vals in self._grpo_metrics.items()
            if vals
        }
        all_logs = {**logs, **grpo_avg}
        logs = {
            k: v for k, v in all_logs.items()
            if k in self._ESSENTIAL_METRICS or k.startswith("eval/")
        }

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

        if (
            self.evaluator is not None
            and self.state.global_step > 0
            and self.state.global_step % self.args.eval_steps == 0
        ):
            eval_summary = self.evaluator.run_eval(self.state.global_step)
            if eval_summary:
                eval_logs = {
                    f"eval/{k}": v
                    for k, v in eval_summary.items()
                    if isinstance(v, (int, float))
                }
                logs.update(eval_logs)

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
