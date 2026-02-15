import asyncio
import logging
import os
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx
import torch
from openai import AsyncOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from rl.algorithms.advantages import compute_group_advantages

logger = logging.getLogger(__name__)


@dataclass
class Batch:
    batch_id: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    inference_logprobs: torch.Tensor
    advantages: torch.Tensor
    num_valid_prompts: int = -1
    policy_version_min: int = 0
    policy_version_max: int = 0
    generation_time: float = 0.0
    prompts: List[str] = field(default_factory=list)
    completions: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class Orchestrator:
    def __init__(
        self,
        *,
        server_base_url: str,
        max_concurrent: int = 100,
        client_timeout: float = 600.0,
        model_name: str,
        num_generations: int = 8,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        batch_size: int = 4,
        dataset: Any = None,
        reward_fn: Callable[[List[str], List[str]], List[float]] = None,
        tokenizer: Any,
        max_prompt_length: int = 512,
        generation_timeout: float = 600.0,
        inflight_weight_updates: bool = False,
        max_off_policy_steps: int = 1,
        continuous_batching: bool = False,
        pool_size: int = 16,
        log_router: Any = None,
        algorithm: str = "grpo",
        normalize_advantages_by_std: bool = False,
        overlong_max_length: int = 16384,
        overlong_cache: int = 4096,
        use_overlong_penalty: bool = False,
        use_dynamic_sampling: bool = False,
    ):
        self.server_base_url = server_base_url
        self.max_concurrent = max_concurrent
        self.client_timeout = client_timeout
        self.model_name = model_name
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.num_prompts = batch_size // num_generations
        self.dataset = dataset
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.generation_timeout = generation_timeout
        self.inflight_weight_updates = inflight_weight_updates
        self.max_off_policy_steps = max_off_policy_steps
        self.continuous_batching = continuous_batching
        self.pool_size = pool_size
        self.log_router = log_router
        self.algorithm = algorithm
        self.normalize_advantages_by_std = normalize_advantages_by_std
        self.overlong_max_length = overlong_max_length
        self.overlong_cache = overlong_cache
        self.use_overlong_penalty = use_overlong_penalty
        self.use_dynamic_sampling = use_dynamic_sampling

        self.request_queue: queue.Queue[Optional[int]] = queue.Queue()
        self.result_queue: queue.Queue[Batch] = queue.Queue()
        self.completed_batches: Dict[int, Batch] = {}

        self.is_generating = False
        self.current_policy_version: int = 0

        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.worker_loop: Optional[asyncio.AbstractEventLoop] = None
        self.client: Optional[AsyncOpenAI] = None

        self._pool_active = threading.Event()
        self._pool_active.set()
        self._pool_in_flight: int = 0

    def start(self) -> None:
        if self.continuous_batching:
            target = self._continuous_pool_worker
            name = "ContinuousBatchPool"
        else:
            target = self._generation_worker
            name = "AsyncBatchGenerator"

        self.worker_thread = threading.Thread(
            target=target,
            daemon=True,
            name=name,
        )
        self.worker_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if not self.continuous_batching:
            self.request_queue.put(None)  # poison pill (batch-at-a-time only)
        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)

    def pause_pool(self) -> None:
        if self.continuous_batching:
            self._pool_active.clear()

    def resume_pool(self) -> None:
        if self.continuous_batching:
            self._pool_active.set()

    def submit_batch(self, batch_id: int) -> None:
        if self.continuous_batching:
            return  # pool manages its own work
        self.request_queue.put(batch_id)

    def get_batch(self, batch_id: int) -> Batch:
        start_time = time.time()
        while True:
            if batch_id in self.completed_batches:
                return self.completed_batches.pop(batch_id)

            try:
                result = self.result_queue.get(timeout=0.1)
                self.completed_batches[result.batch_id] = result
                if result.batch_id == batch_id:
                    return self.completed_batches.pop(batch_id)
            except queue.Empty:
                pass

            if time.time() - start_time > self.generation_timeout:
                raise TimeoutError(
                    f"Batch {batch_id} timed out after "
                    f"{self.generation_timeout}s"
                )

    def _generation_worker(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.worker_loop = loop

        self.client = AsyncOpenAI(
            base_url=self.server_base_url,
            api_key="EMPTY",
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=self.max_concurrent),
                timeout=self.client_timeout,
            ),
        )

        try:
            while not self.stop_event.is_set():
                try:
                    batch_id = self.request_queue.get(timeout=0.1)
                    if batch_id is None:
                        break
                    result = loop.run_until_complete(
                        self._generate_batch(batch_id)
                    )
                    self.result_queue.put(result)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in generation worker: {e}")
                    raise
        finally:
            loop.run_until_complete(self.client.close())
            loop.close()
            asyncio.set_event_loop(None)

    def _continuous_pool_worker(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.worker_loop = loop

        self.client = AsyncOpenAI(
            base_url=self.server_base_url,
            api_key="EMPTY",
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=self.max_concurrent),
                timeout=self.client_timeout,
            ),
        )

        try:
            loop.run_until_complete(self._run_continuous_pool())
        except Exception as e:
            logger.error(f"Error in continuous pool worker: {e}")
            raise
        finally:
            loop.run_until_complete(self.client.close())
            loop.close()
            asyncio.set_event_loop(None)

    async def _run_continuous_pool(self) -> None:
        assert self.client is not None
        self.is_generating = True

        G = self.num_generations
        batch_id = 0
        t_batch_start = time.time()
        pending_results: List[tuple] = []  # (prompt_data, gen_result)
        results_queue: asyncio.Queue = asyncio.Queue()

        if self.log_router:
            self.log_router.log_inference(
                f"[bold cyan]Continuous batching pool started "
                f"(pool_size={self.pool_size}, "
                f"batch_size={self.batch_size}, "
                f"prompts_per_batch={self.num_prompts})[/bold cyan]"
            )

        async def _generate_one() -> None:
            prompt_data = self._sample_prompt()
            try:
                gen_result = await self._generate_single_prompt(
                    prompt_data["prompt"], G,
                )
                await results_queue.put((prompt_data, gen_result))
            except Exception as e:
                logger.error(f"Prompt generation failed: {e}")
                await results_queue.put(None)

        def _spawn_task() -> None:
            task = asyncio.ensure_future(_generate_one())
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)
            self._pool_in_flight = len(active_tasks)

        active_tasks: set = set()
        for _ in range(self.pool_size):
            _spawn_task()

        while not self.stop_event.is_set():
            try:
                result = await asyncio.wait_for(
                    results_queue.get(), timeout=0.5,
                )
            except asyncio.TimeoutError:
                self._pool_in_flight = len(active_tasks)

                if self._pool_active.is_set() and len(active_tasks) < self.pool_size:
                    deficit = self.pool_size - len(active_tasks)
                    for _ in range(deficit):
                        _spawn_task()
                continue

            self._pool_in_flight = len(active_tasks)

            if self._pool_active.is_set():
                _spawn_task()

            if result is None:
                continue

            pending_results.append(result)

            if len(pending_results) >= self.num_prompts:
                batch_items = pending_results[:self.num_prompts]
                pending_results = pending_results[self.num_prompts:]

                batch = self._assemble_batch(
                    batch_items, batch_id, t_batch_start,
                )
                self.result_queue.put(batch)
                batch_id += 1
                t_batch_start = time.time()

        self.is_generating = False
        self._pool_in_flight = 0
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    def _compute_overlong_penalty(
        self,
        completion_lengths: List[int],
    ) -> List[float]:
        penalties = []
        threshold = self.overlong_max_length - self.overlong_cache

        for length in completion_lengths:
            if length <= threshold:
                penalties.append(0.0)
            elif length <= self.overlong_max_length:
                penalties.append((threshold - length) / self.overlong_cache)
            else:
                penalties.append(-1.0)
        return penalties

    async def _generate_batch(self, batch_id: int) -> Batch:
        self.is_generating = True
        assert self.client is not None
        t_start = time.time()

        G = self.num_generations

        n = len(self.dataset)
        indices = random.sample(range(n), min(self.num_prompts, n))
        items = [self.dataset[i] for i in indices]
        prompts = [item["prompt"] for item in items]
        ground_truths = [item["ground_truth"] for item in items]
        B = len(prompts)

        if self.log_router:
            self.log_router.log_inference(
                f"Generating [bold]{B}[/bold]x[bold]{G}[/bold] completions "
                f"for batch [bold]{batch_id}[/bold]..."
                )

        tasks = []
        for prompt in prompts:
            tasks.append(
                self._generate_single_prompt(prompt, G)
            )

        results = await asyncio.gather(*tasks)

        all_start_versions: List[int] = []
        all_end_versions: List[int] = []
        for _, _, _, sv, ev in results:
            all_start_versions.append(sv)
            all_end_versions.append(ev)

        if self.inflight_weight_updates:
            valid_indices: List[int] = []
            for i, (sv, ev) in enumerate(
                zip(all_start_versions, all_end_versions)
            ):
                if ev - sv <= self.max_off_policy_steps:
                    valid_indices.append(i)

            num_discarded = B - len(valid_indices)

            if num_discarded > 0 and self.log_router:
                self.log_router.log_inference(
                    f"[yellow]Discarded {num_discarded}/{B} prompts "
                    f"(off-policy span > {self.max_off_policy_steps})[/yellow]"
                )

            if len(valid_indices) == 0:
                self.is_generating = False
                ver_min = min(all_start_versions) if all_start_versions else 0
                ver_max = max(all_end_versions) if all_end_versions else 0
                t_gen = time.time() - t_start
                if self.log_router:
                    self.log_router.log_inference(
                        f"[red]All {B} prompts discarded for batch "
                        f"{batch_id} – returning empty batch[/red]"
                    )
                empty = torch.zeros(0)
                return Batch(
                    batch_id=batch_id,
                    input_ids=empty, attention_mask=empty,
                    loss_mask=empty, inference_logprobs=empty,
                    advantages=empty,
                    num_valid_prompts=0,
                    policy_version_min=ver_min,
                    policy_version_max=ver_max,
                    generation_time=t_gen,
                    metrics={
                        "train/num_discarded_prompts": float(B),
                        "train/generate_s": t_gen,
                    },
                )

            results = [results[i] for i in valid_indices]
            prompts = [prompts[i] for i in valid_indices]
            ground_truths = [ground_truths[i] for i in valid_indices]
            all_start_versions = [all_start_versions[i] for i in valid_indices]
            all_end_versions = [all_end_versions[i] for i in valid_indices]
            B = len(prompts)
        else:
            num_discarded = 0
            valid_indices = list(range(B))

        all_completions: List[str] = []
        all_token_ids: List[List[int]] = []
        all_logprobs: List[List[float]] = []

        for comps, tok_ids, lps, _, _ in results:
            all_completions.extend(comps)
            all_token_ids.extend(tok_ids)
            all_logprobs.extend(lps)

        t_gen = time.time() - t_start

        comp_lengths = [len(ids) for ids in all_token_ids]
        avg_comp_len = sum(comp_lengths) / max(len(comp_lengths), 1)
        max_comp_len = float(max(comp_lengths)) if comp_lengths else 0.0
        total_tokens = sum(comp_lengths)
        tok_per_s = total_tokens / t_gen if t_gen > 0 else 0.0

        if self.log_router:
            self.log_router.log_inference(
                f"Generated [bold]{B * G}[/bold] completions in "
                f"[yellow]{t_gen:.1f}s[/yellow]  "
                f"([dim]{tok_per_s:.0f} tok/s, avg {avg_comp_len:.0f} tok, "
                f"max {max_comp_len:.0f} tok[/dim])"
                )

        expanded_gts = [gt for gt in ground_truths for _ in range(G)]
        rewards = self.reward_fn(all_completions, expanded_gts)

        num_overlong_penalized = 0
        if self.use_overlong_penalty:
            overlong_penalties = self._compute_overlong_penalty(comp_lengths)
            rewards = [r + p for r, p in zip(rewards, overlong_penalties)]
            num_overlong_penalized = sum(1 for p in overlong_penalties if p != 0.0)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        grouped_rewards_pre = rewards_tensor.view(-1, G)
        group_std_pre = grouped_rewards_pre.std(dim=1)
        zero_var_frac = (group_std_pre == 0).float().mean().item()

        num_dynamic_filtered = 0
        if self.use_dynamic_sampling:
            grouped_rewards_ds = rewards_tensor.view(-1, G)
            correct_counts = (grouped_rewards_ds > 0).sum(dim=1)

            valid_mask = (correct_counts > 0) & (correct_counts < G)
            valid_ds_indices = valid_mask.nonzero(as_tuple=False).squeeze(1).tolist()

            num_dynamic_filtered = B - len(valid_ds_indices)

            if len(valid_ds_indices) == 0:
                self.is_generating = False
                ver_min = min(all_start_versions) if all_start_versions else 0
                ver_max = max(all_end_versions) if all_end_versions else 0
                t_total = time.time() - t_start
                if self.log_router:
                    self.log_router.log_inference(
                        f"[red]Dynamic Sampling: all {B} prompts "
                        f"filtered – returning empty batch[/red]"
                    )
                empty = torch.zeros(0)
                return Batch(
                    batch_id=batch_id,
                    input_ids=empty, attention_mask=empty,
                    loss_mask=empty, inference_logprobs=empty,
                    advantages=empty,
                    num_valid_prompts=0,
                    policy_version_min=ver_min,
                    policy_version_max=ver_max,
                    generation_time=t_total,
                    metrics={
                        "train/num_discarded_prompts": float(num_discarded),
                        "train/dynamic_filtered": float(B),
                        "train/generate_s": t_total,
                    },
                )

            if num_dynamic_filtered > 0:
                if self.log_router:
                    self.log_router.log_inference(
                        f"[yellow]Dynamic Sampling: filtered "
                        f"{num_dynamic_filtered}/{B} prompts[/yellow]"
                    )

                prompts = [prompts[i] for i in valid_ds_indices]
                ground_truths = [ground_truths[i] for i in valid_ds_indices]

                keep_flat = []
                for i in valid_ds_indices:
                    keep_flat.extend(range(i * G, (i + 1) * G))

                all_completions = [all_completions[j] for j in keep_flat]
                all_token_ids = [all_token_ids[j] for j in keep_flat]
                all_logprobs = [all_logprobs[j] for j in keep_flat]
                comp_lengths = [comp_lengths[j] for j in keep_flat]

                all_start_versions = [all_start_versions[i] for i in valid_ds_indices]
                all_end_versions = [all_end_versions[i] for i in valid_ds_indices]

                rewards_tensor = rewards_tensor[torch.tensor(keep_flat, dtype=torch.long)]
                rewards = rewards_tensor.tolist()
                B = len(prompts)

        advantages = compute_group_advantages(
            rewards_tensor, G,
            normalize_by_std=self.normalize_advantages_by_std,
        )

        expanded_prompts = [p for p in prompts for _ in range(G)]
        batch_tensors = self._prepare_tensors(
            expanded_prompts, all_token_ids, all_logprobs,
        )

        self.is_generating = False

        ver_min = min(all_start_versions) if all_start_versions else 0
        ver_max = max(all_end_versions) if all_end_versions else 0

        metrics = {
            "rewards/mean": rewards_tensor.mean().item(),
            "rewards/std": rewards_tensor.std().item(),
            "rewards/advantage_mean": advantages.mean().item(),
            "rewards/zero_var_group_frac": zero_var_frac,
            "train/completion_len_mean": avg_comp_len,
            "train/completion_len_max": max_comp_len,
            "train/generate_s": t_gen,
            "train/num_discarded_prompts": float(num_discarded),
            "train/policy_version_min": float(ver_min),
            "train/policy_version_max": float(ver_max),
        }
        if self.use_dynamic_sampling:
            metrics["train/dynamic_filtered"] = float(num_dynamic_filtered)
        if self.use_overlong_penalty:
            metrics["train/overlong_penalized"] = float(num_overlong_penalized)

        return Batch(
            batch_id=batch_id,
            input_ids=batch_tensors["input_ids"],
            attention_mask=batch_tensors["attention_mask"],
            loss_mask=batch_tensors["loss_mask"],
            inference_logprobs=batch_tensors["inference_logprobs"],
            advantages=advantages,
            num_valid_prompts=B,
            policy_version_min=ver_min,
            policy_version_max=ver_max,
            generation_time=t_gen,
            prompts=prompts,
            completions=all_completions,
            rewards=rewards,
            metrics=metrics,
        )

    async def _generate_single_prompt(
        self,
        prompt: str,
        n: int,
    ) -> tuple:
        assert self.client is not None

        start_version = self.current_policy_version

        response = await self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            n=n,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["<|im_end|>", "<|endoftext|>"],
            logprobs=1,
            extra_body={"skip_special_tokens": False},
        )

        end_version = self.current_policy_version

        completions: List[str] = []
        token_ids_list: List[List[int]] = []
        logprobs_list: List[List[float]] = []

        for choice in response.choices:
            completions.append(choice.text)

            tok_ids: List[int] = []
            lps: List[float] = []
            if choice.logprobs and choice.logprobs.tokens:
                for i, token_logprob in enumerate(
                    choice.logprobs.token_logprobs or []
                ):
                    lps.append(
                        token_logprob if token_logprob is not None else 0.0
                    )

                encoded = self.tokenizer.encode(
                    choice.text, add_special_tokens=False,
                )
                tok_ids = encoded

                if len(lps) < len(tok_ids):
                    lps.extend([0.0] * (len(tok_ids) - len(lps)))
                elif len(lps) > len(tok_ids):
                    lps = lps[:len(tok_ids)]
            else:
                tok_ids = self.tokenizer.encode(
                    choice.text, add_special_tokens=False,
                )
                lps = [0.0] * len(tok_ids)

            token_ids_list.append(tok_ids)
            logprobs_list.append(lps)

        return completions, token_ids_list, logprobs_list, start_version, end_version

    def _sample_prompt(self) -> Dict[str, str]:
        idx = random.randint(0, len(self.dataset) - 1)
        item = self.dataset[idx]
        return {"prompt": item["prompt"], "ground_truth": item["ground_truth"]}

    def _assemble_batch(
        self,
        items: List[tuple],
        batch_id: int,
        t_start: float,
    ) -> Batch:
        G = self.num_generations

        prompts = [item[0]["prompt"] for item in items]
        ground_truths = [item[0]["ground_truth"] for item in items]
        results = [item[1] for item in items]
        B = len(prompts)

        all_start_versions: List[int] = [r[3] for r in results]
        all_end_versions: List[int] = [r[4] for r in results]

        if self.inflight_weight_updates:
            valid_indices: List[int] = []
            for i, (sv, ev) in enumerate(
                zip(all_start_versions, all_end_versions)
            ):
                if ev - sv <= self.max_off_policy_steps:
                    valid_indices.append(i)

            num_discarded = B - len(valid_indices)

            if num_discarded > 0 and self.log_router:
                self.log_router.log_inference(
                    f"[yellow]Discarded {num_discarded}/{B} prompts "
                    f"(off-policy span > {self.max_off_policy_steps})[/yellow]"
                )

            if len(valid_indices) == 0:
                ver_min = min(all_start_versions) if all_start_versions else 0
                ver_max = max(all_end_versions) if all_end_versions else 0
                t_gen = time.time() - t_start
                if self.log_router:
                    self.log_router.log_inference(
                        f"[red]All {B} prompts discarded for batch "
                        f"{batch_id} – returning empty batch[/red]"
                    )
                empty = torch.zeros(0)
                return Batch(
                    batch_id=batch_id,
                    input_ids=empty, attention_mask=empty,
                    loss_mask=empty, inference_logprobs=empty,
                    advantages=empty,
                    num_valid_prompts=0,
                    policy_version_min=ver_min,
                    policy_version_max=ver_max,
                    generation_time=t_gen,
                    metrics={
                        "train/num_discarded_prompts": float(B),
                        "train/generate_s": t_gen,
                    },
                )

            results = [results[i] for i in valid_indices]
            prompts = [prompts[i] for i in valid_indices]
            ground_truths = [ground_truths[i] for i in valid_indices]
            all_start_versions = [all_start_versions[i] for i in valid_indices]
            all_end_versions = [all_end_versions[i] for i in valid_indices]
            B = len(prompts)
        else:
            num_discarded = 0

        all_completions: List[str] = []
        all_token_ids: List[List[int]] = []
        all_logprobs: List[List[float]] = []

        for comps, tok_ids, lps, _, _ in results:
            all_completions.extend(comps)
            all_token_ids.extend(tok_ids)
            all_logprobs.extend(lps)

        t_gen = time.time() - t_start

        comp_lengths = [len(ids) for ids in all_token_ids]
        avg_comp_len = sum(comp_lengths) / max(len(comp_lengths), 1)
        max_comp_len = float(max(comp_lengths)) if comp_lengths else 0.0
        total_tokens = sum(comp_lengths)
        tok_per_s = total_tokens / t_gen if t_gen > 0 else 0.0

        if self.log_router:
            self.log_router.log_inference(
                f"Assembled batch [bold]{batch_id}[/bold]: "
                f"[bold]{B * G}[/bold] completions in "
                f"[yellow]{t_gen:.1f}s[/yellow]  "
                f"([dim]{tok_per_s:.0f} tok/s, avg {avg_comp_len:.0f} tok, "
                f"max {max_comp_len:.0f} tok[/dim])"
                )

        expanded_gts = [gt for gt in ground_truths for _ in range(G)]
        rewards = self.reward_fn(all_completions, expanded_gts)

        num_overlong_penalized = 0
        if self.use_overlong_penalty:
            overlong_penalties = self._compute_overlong_penalty(comp_lengths)
            rewards = [r + p for r, p in zip(rewards, overlong_penalties)]
            num_overlong_penalized = sum(1 for p in overlong_penalties if p != 0.0)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        grouped_rewards_pre = rewards_tensor.view(-1, G)
        group_std_pre = grouped_rewards_pre.std(dim=1)
        zero_var_frac = (group_std_pre == 0).float().mean().item()

        num_dynamic_filtered = 0
        if self.use_dynamic_sampling:
            grouped_rewards_ds = rewards_tensor.view(-1, G)
            correct_counts = (grouped_rewards_ds > 0).sum(dim=1)

            valid_mask = (correct_counts > 0) & (correct_counts < G)
            valid_ds_indices = valid_mask.nonzero(as_tuple=False).squeeze(1).tolist()

            num_dynamic_filtered = B - len(valid_ds_indices)

            if len(valid_ds_indices) == 0:
                ver_min = min(all_start_versions) if all_start_versions else 0
                ver_max = max(all_end_versions) if all_end_versions else 0
                t_total = time.time() - t_start
                if self.log_router:
                    self.log_router.log_inference(
                        f"[red]Dynamic Sampling: all {B} prompts "
                        f"filtered – returning empty batch[/red]"
                    )
                empty = torch.zeros(0)
                return Batch(
                    batch_id=batch_id,
                    input_ids=empty, attention_mask=empty,
                    loss_mask=empty, inference_logprobs=empty,
                    advantages=empty,
                    num_valid_prompts=0,
                    policy_version_min=ver_min,
                    policy_version_max=ver_max,
                    generation_time=t_total,
                    metrics={
                        "train/num_discarded_prompts": float(num_discarded),
                        "train/dynamic_filtered": float(B),
                        "train/generate_s": t_total,
                    },
                )

            if num_dynamic_filtered > 0:
                if self.log_router:
                    self.log_router.log_inference(
                        f"[yellow]Dynamic Sampling: filtered "
                        f"{num_dynamic_filtered}/{B} prompts[/yellow]"
                    )

                prompts = [prompts[i] for i in valid_ds_indices]
                ground_truths = [ground_truths[i] for i in valid_ds_indices]

                keep_flat = []
                for i in valid_ds_indices:
                    keep_flat.extend(range(i * G, (i + 1) * G))

                all_completions = [all_completions[j] for j in keep_flat]
                all_token_ids = [all_token_ids[j] for j in keep_flat]
                all_logprobs = [all_logprobs[j] for j in keep_flat]
                comp_lengths = [comp_lengths[j] for j in keep_flat]

                all_start_versions = [all_start_versions[i] for i in valid_ds_indices]
                all_end_versions = [all_end_versions[i] for i in valid_ds_indices]

                rewards_tensor = rewards_tensor[torch.tensor(keep_flat, dtype=torch.long)]
                rewards = rewards_tensor.tolist()
                B = len(prompts)

        advantages = compute_group_advantages(
            rewards_tensor, G,
            normalize_by_std=self.normalize_advantages_by_std,
        )

        expanded_prompts = [p for p in prompts for _ in range(G)]
        batch_tensors = self._prepare_tensors(
            expanded_prompts, all_token_ids, all_logprobs,
        )

        ver_min = min(all_start_versions) if all_start_versions else 0
        ver_max = max(all_end_versions) if all_end_versions else 0

        metrics = {
            "rewards/mean": rewards_tensor.mean().item(),
            "rewards/std": rewards_tensor.std().item(),
            "rewards/advantage_mean": advantages.mean().item(),
            "rewards/zero_var_group_frac": zero_var_frac,
            "train/completion_len_mean": avg_comp_len,
            "train/completion_len_max": max_comp_len,
            "train/generate_s": t_gen,
            "train/num_discarded_prompts": float(num_discarded),
            "train/policy_version_min": float(ver_min),
            "train/policy_version_max": float(ver_max),
        }
        if self.use_dynamic_sampling:
            metrics["train/dynamic_filtered"] = float(num_dynamic_filtered)
        if self.use_overlong_penalty:
            metrics["train/overlong_penalized"] = float(num_overlong_penalized)

        return Batch(
            batch_id=batch_id,
            input_ids=batch_tensors["input_ids"],
            attention_mask=batch_tensors["attention_mask"],
            loss_mask=batch_tensors["loss_mask"],
            inference_logprobs=batch_tensors["inference_logprobs"],
            advantages=advantages,
            num_valid_prompts=B,
            policy_version_min=ver_min,
            policy_version_max=ver_max,
            generation_time=t_gen,
            prompts=prompts,
            completions=all_completions,
            rewards=rewards,
            metrics=metrics,
        )

    def _prepare_tensors(
        self,
        prompts: List[str],
        completion_token_ids: List[List[int]],
        completion_logprobs: List[List[float]],
    ) -> Dict[str, torch.Tensor]:
        tokenizer = self.tokenizer

        prompt_encodings = tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_prompt_length,
        )

        all_input_ids: List[List[int]] = []
        all_loss_masks: List[List[int]] = []
        prompt_lengths: List[int] = []

        for i in range(len(prompts)):
            p_ids = prompt_encodings["input_ids"][i]
            c_ids = completion_token_ids[i]
            all_input_ids.append(p_ids + c_ids)
            all_loss_masks.append([0] * len(p_ids) + [1] * len(c_ids))
            prompt_lengths.append(len(p_ids))

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

        inference_lps = torch.zeros(
            len(prompts), max_len - 1, dtype=torch.float32,
        )
        for i in range(len(prompts)):
            c_lps = completion_logprobs[i]
            start_pos = prompt_lengths[i] - 1
            end_pos = start_pos + len(c_lps)
            fit = min(end_pos, max_len - 1) - start_pos
            if fit > 0:
                inference_lps[i, start_pos:start_pos + fit] = torch.tensor(
                    c_lps[:fit], dtype=torch.float32,
                )

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
            "loss_mask": torch.tensor(padded_loss, dtype=torch.float),
            "inference_logprobs": inference_lps,
        }
