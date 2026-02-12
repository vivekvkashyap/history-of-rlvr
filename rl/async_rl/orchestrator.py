"""
Asynchronous batch generation orchestrator for GRPO training.

Runs a daemon thread with its own asyncio event loop that generates
completions via the vLLM OpenAI-compatible API, computes rewards and
GRPO group-relative advantages, then packages the result into a Batch
object that the trainer thread can consume.

Two operating modes:
  - **Batch-at-a-time** (``continuous_batching=False``): the trainer
    explicitly submits batch IDs; the orchestrator generates them one at
    a time.  Communication uses ``request_queue`` / ``result_queue``.
  - **Continuous batching** (``continuous_batching=True``): the
    orchestrator maintains a saturated pool of ``pool_size`` concurrent
    prompt-generation tasks.  Whenever a task completes its slot is
    immediately repopulated, keeping the vLLM server at peak throughput.
    Once ``batch_size // num_generations`` prompt results accumulate
    they are assembled into a ``Batch`` and placed on ``result_queue``.

Supports two weight-sync modes:
  - Legacy (inflight_weight_updates=False): trainer waits for generation
    to finish before syncing weights, using ``is_generating`` flag.
  - In-flight (inflight_weight_updates=True, PipelineRL-style): weights
    are pushed mid-generation.  Each prompt records the policy version at
    the start and end of its generation.  Prompts whose version span
    exceeds ``max_off_policy_steps`` are discarded before training.

Simplified from the verifiers-rl orchestrator: no Environment
abstraction, no multi-step trajectories -- just single-turn generation.

The dataset and reward function are injected via constructor parameters,
so the orchestrator is environment-agnostic.
"""

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

# Parent dir for cross-package imports (history_of_rlvr/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from rl.algorithms.grpo.grpo import compute_group_advantages

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
#  Batch result dataclass
# ════════════════════════════════════════════════════════════════════════


@dataclass
class Batch:
    """
    Result from one round of async generation.

    Contains everything the trainer needs to do a forward pass + loss
    computation, plus metadata for logging.

    ``num_valid_prompts`` indicates how many prompts survived off-policy
    filtering.  When 0, the trainer should skip the step.
    """
    batch_id: int

    # ── Tensors for training (built by the orchestrator) ───────────────
    input_ids: torch.Tensor          # (B*G, L)   padded token ids
    attention_mask: torch.Tensor     # (B*G, L)   1 for real, 0 for padding
    loss_mask: torch.Tensor          # (B*G, L)   1 for completion tokens
    inference_logprobs: torch.Tensor # (B*G, L-1) vLLM logprobs aligned
    advantages: torch.Tensor         # (B*G,)     group-relative advantages

    # ── Off-policy tracking ────────────────────────────────────────────
    num_valid_prompts: int = -1      # prompts after filtering (-1 = no filtering)
    policy_version_min: int = 0      # min policy version seen in this batch
    policy_version_max: int = 0      # max policy version seen in this batch

    # ── Logging ────────────────────────────────────────────────────────
    generation_time: float = 0.0
    prompts: List[str] = field(default_factory=list)
    completions: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ════════════════════════════════════════════════════════════════════════


class Orchestrator:
    """
    Manages asynchronous batch generation in parallel with RL training.

    Two operating modes:

    **Batch-at-a-time** (``continuous_batching=False``):
        orchestrator.start()
        orchestrator.submit_batch(0)          # kick off first batch
        for step in range(max_steps):
            orchestrator.submit_batch(step + 1)  # pipeline next batch
            batch = orchestrator.get_batch(step)  # block for current
            ... training on batch ...
        orchestrator.stop()

    **Continuous batching** (``continuous_batching=True``):
        orchestrator.start()   # pool auto-generates immediately
        for step in range(max_steps):
            batch = orchestrator.get_batch(step)  # block for next batch
            ... training on batch ...
        orchestrator.stop()

    In continuous batching mode the orchestrator maintains ``pool_size``
    concurrent prompt-generation tasks.  Whenever a task completes its
    slot is immediately repopulated, keeping the vLLM server saturated.
    Once ``batch_size // num_generations`` prompt results accumulate
    they are assembled into a ``Batch`` and delivered via the result queue.
    """

    def __init__(
        self,
        *,
        # ── vLLM server connection ─────────────────────────────────────
        server_base_url: str,
        max_concurrent: int = 100,
        client_timeout: float = 600.0,
        # ── Generation parameters ──────────────────────────────────────
        model_name: str,
        num_generations: int = 8,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        # ── Batch ──────────────────────────────────────────────────────
        batch_size: int = 4,
        # ── Dataset (duck-typed: needs __len__ + __getitem__ → {"prompt", "ground_truth"})
        dataset: Any = None,
        # ── Reward function: (completions, ground_truths) → list[float]
        reward_fn: Callable[[List[str], List[str]], List[float]] = None,
        # ── Tokenizer (for prompt tokenisation) ────────────────────────
        tokenizer: Any,
        max_prompt_length: int = 512,
        # ── Timeouts ──────────────────────────────────────────────────
        generation_timeout: float = 600.0,
        # ── In-flight weight updates ──────────────────────────────────
        inflight_weight_updates: bool = False,
        max_off_policy_steps: int = 1,
        # ── Continuous batching ───────────────────────────────────────
        continuous_batching: bool = False,
        pool_size: int = 16,
        # ── Log router (optional) ─────────────────────────────────────
        log_router: Any = None,
    ):
        self.server_base_url = server_base_url
        self.max_concurrent = max_concurrent
        self.client_timeout = client_timeout
        self.model_name = model_name
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size                          # total rollouts per step
        self.num_prompts = batch_size // num_generations       # prompts per step
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

        # Queues for thread communication
        self.request_queue: queue.Queue[Optional[int]] = queue.Queue()
        self.result_queue: queue.Queue[Batch] = queue.Queue()
        self.completed_batches: Dict[int, Batch] = {}

        # State flag: True while a generation request is being processed
        self.is_generating = False

        # Policy version counter – written by the trainer thread after
        # each successful weight sync, read by the orchestrator thread.
        # CPython GIL makes single-int read/write atomic across threads.
        self.current_policy_version: int = 0

        # Thread management
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.worker_loop: Optional[asyncio.AbstractEventLoop] = None
        self.client: Optional[AsyncOpenAI] = None

        # ── Continuous pool pause/resume (for weight sync) ────────────
        # When cleared, the pool stops spawning new tasks so that
        # in-flight generation drains and vLLM workers become free for
        # weight sync without contention.
        self._pool_active = threading.Event()
        self._pool_active.set()  # starts active
        self._pool_in_flight: int = 0  # number of in-flight generation tasks

    # ════════════════════════════════════════════════════════════════════
    #  Public API (called from the trainer thread)
    # ════════════════════════════════════════════════════════════════════

    def start(self) -> None:
        """Start the async generation worker thread."""
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
        """Stop the worker thread."""
        self.stop_event.set()
        if not self.continuous_batching:
            self.request_queue.put(None)  # poison pill (batch-at-a-time only)
        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)

    def pause_pool(self) -> None:
        """
        Pause the continuous batching pool.

        Stops spawning new generation tasks so that in-flight requests
        naturally drain.  Call before weight sync to eliminate contention
        with ``engine.collective_rpc`` on the vLLM server.

        No-op when ``continuous_batching`` is disabled.
        """
        if self.continuous_batching:
            self._pool_active.clear()

    def resume_pool(self) -> None:
        """
        Resume the continuous batching pool after a pause.

        The pool will immediately start refilling to ``pool_size``
        concurrent tasks.

        No-op when ``continuous_batching`` is disabled.
        """
        if self.continuous_batching:
            self._pool_active.set()

    def submit_batch(self, batch_id: int) -> None:
        """
        Submit a batch generation request (non-blocking).

        No-op when ``continuous_batching`` is enabled because the pool
        auto-generates prompts continuously.
        """
        if self.continuous_batching:
            return  # pool manages its own work
        self.request_queue.put(batch_id)

    def get_batch(self, batch_id: int) -> Batch:
        """
        Get a completed batch result.  Blocks until the batch is ready.

        Raises TimeoutError if the batch doesn't complete within
        ``generation_timeout`` seconds.
        """
        start_time = time.time()
        while True:
            # Check cache first
            if batch_id in self.completed_batches:
                return self.completed_batches.pop(batch_id)

            # Poll the result queue
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

    # ════════════════════════════════════════════════════════════════════
    #  Worker thread
    # ════════════════════════════════════════════════════════════════════

    def _generation_worker(self) -> None:
        """Worker thread entry: runs an asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.worker_loop = loop

        # Create the async OpenAI client inside the worker thread
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
                    if batch_id is None:  # poison pill
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

    # ════════════════════════════════════════════════════════════════════
    #  Continuous batching pool (alternative to batch-at-a-time worker)
    # ════════════════════════════════════════════════════════════════════

    def _continuous_pool_worker(self) -> None:
        """
        Worker thread entry for continuous batching mode.

        Runs an asyncio event loop that keeps ``pool_size`` prompt-level
        generation tasks in flight at all times.  Completed results
        accumulate in a buffer; once ``num_prompts`` results are ready a
        ``Batch`` is assembled and placed on ``result_queue``.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.worker_loop = loop

        # Create the async OpenAI client inside the worker thread
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
        """
        Keep ``pool_size`` prompt-generation tasks always in flight.

        When any task completes its slot is immediately repopulated so
        that the vLLM server always sees a constant number of requests,
        sustaining peak inference throughput without synchronous batch
        boundaries.

        Once ``num_prompts`` prompt results have accumulated they are
        assembled into a ``Batch`` (with off-policy filtering, rewards,
        advantages, and tensor preparation) and placed on ``result_queue``
        for the trainer thread.

        Note: batch assembly (reward computation, tokenisation, tensor
        preparation) runs synchronously on the event loop because the
        HuggingFace fast tokenizer is not thread-safe.  The assembly
        time is small relative to generation latency so the stall is
        negligible.
        """
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
            """Generate completions for one sampled prompt."""
            prompt_data = self._sample_prompt()
            try:
                gen_result = await self._generate_single_prompt(
                    prompt_data["prompt"], G,
                )
                await results_queue.put((prompt_data, gen_result))
            except Exception as e:
                logger.error(f"Prompt generation failed: {e}")
                # Put an error marker so the slot can still be repopulated
                await results_queue.put(None)

        def _spawn_task() -> None:
            """Spawn a single generation task and track it."""
            task = asyncio.ensure_future(_generate_one())
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)
            self._pool_in_flight = len(active_tasks)

        # Seed the pool with pool_size concurrent tasks
        active_tasks: set = set()
        for _ in range(self.pool_size):
            _spawn_task()

        # Main loop: collect results, repopulate slots, assemble batches
        while not self.stop_event.is_set():
            # Wait for the next completed result (with a short timeout so
            # we can check the stop event periodically)
            try:
                result = await asyncio.wait_for(
                    results_queue.get(), timeout=0.5,
                )
            except asyncio.TimeoutError:
                # While paused, update in-flight count (tasks are draining)
                self._pool_in_flight = len(active_tasks)

                # While paused and pool is drained, refill when resumed
                if self._pool_active.is_set() and len(active_tasks) < self.pool_size:
                    deficit = self.pool_size - len(active_tasks)
                    for _ in range(deficit):
                        _spawn_task()
                continue

            # Update in-flight count after a task completed
            self._pool_in_flight = len(active_tasks)

            # Only repopulate the slot if the pool is active (not paused
            # for weight sync).  When paused, in-flight tasks naturally
            # drain, freeing vLLM workers for contention-free weight sync.
            if self._pool_active.is_set():
                _spawn_task()

            # Skip failed generations
            if result is None:
                continue

            pending_results.append(result)

            # When enough prompt results have accumulated, assemble a Batch
            if len(pending_results) >= self.num_prompts:
                batch_items = pending_results[:self.num_prompts]
                pending_results = pending_results[self.num_prompts:]

                batch = self._assemble_batch(
                    batch_items, batch_id, t_batch_start,
                )
                self.result_queue.put(batch)
                batch_id += 1
                t_batch_start = time.time()

        # ── Drain: cancel remaining tasks on shutdown ─────────────────
        self.is_generating = False
        self._pool_in_flight = 0
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    # ════════════════════════════════════════════════════════════════════
    #  Batch generation (runs inside the worker's asyncio loop)
    # ════════════════════════════════════════════════════════════════════

    async def _generate_batch(self, batch_id: int) -> Batch:
        """
        Generate a full batch of completions via the vLLM server.

        Steps:
          1. Sample num_prompts prompts from the dataset
          2. For each prompt, request num_generations completions
          2b. (In-flight mode) Filter off-policy prompts
          3. Compute rewards via the injected reward function
          4. Compute GRPO group-relative advantages
          5. Tokenize & package into tensors for the trainer
        """
        self.is_generating = True
        assert self.client is not None
        t_start = time.time()

        G = self.num_generations

        # ── 1. Sample prompts ──────────────────────────────────────────
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

        # ── 2. Generate completions via OpenAI API ─────────────────────
        # We send each prompt as a separate request with n=G
        tasks = []
        for prompt in prompts:
            tasks.append(
                self._generate_single_prompt(prompt, G)
            )

        results = await asyncio.gather(*tasks)

        # ── 2b. Off-policy filtering (in-flight mode) ─────────────────
        # Each result is (comps, tok_ids, lps, start_ver, end_ver).
        # Keep only prompts whose policy-version span is acceptable.
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
                # All prompts were too off-policy – return an empty batch
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

            # Re-index to keep only valid prompts
            results = [results[i] for i in valid_indices]
            prompts = [prompts[i] for i in valid_indices]
            ground_truths = [ground_truths[i] for i in valid_indices]
            all_start_versions = [all_start_versions[i] for i in valid_indices]
            all_end_versions = [all_end_versions[i] for i in valid_indices]
            B = len(prompts)
        else:
            num_discarded = 0
            valid_indices = list(range(B))

        # Flatten per-prompt results into flat lists
        all_completions: List[str] = []
        all_token_ids: List[List[int]] = []
        all_logprobs: List[List[float]] = []

        for comps, tok_ids, lps, _, _ in results:
            all_completions.extend(comps)
            all_token_ids.extend(tok_ids)
            all_logprobs.extend(lps)

        t_gen = time.time() - t_start

        # ── Post-generation stats ──────────────────────────────────────
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

        # ── 3. Compute rewards ─────────────────────────────────────────
        expanded_gts = [gt for gt in ground_truths for _ in range(G)]
        rewards = self.reward_fn(all_completions, expanded_gts)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        # ── 4. GRPO group-relative advantages ──────────────────────────
        advantages = compute_group_advantages(rewards_tensor, G)

        # ── 4b. Zero-variance group fraction ─────────────────────────
        grouped_rewards = rewards_tensor.view(-1, G)
        group_std = grouped_rewards.std(dim=1)
        zero_var_frac = (group_std == 0).float().mean().item()

        # ── 5. Tokenize & build tensors ────────────────────────────────
        expanded_prompts = [p for p in prompts for _ in range(G)]
        batch_tensors = self._prepare_tensors(
            expanded_prompts, all_token_ids, all_logprobs,
        )

        self.is_generating = False

        # ── Policy version range for logging ───────────────────────────
        ver_min = min(all_start_versions) if all_start_versions else 0
        ver_max = max(all_end_versions) if all_end_versions else 0

        # ── Metrics ────────────────────────────────────────────────────
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
        """
        Generate n completions for a single prompt using the OpenAI API.

        Returns:
            (completions, token_ids, logprobs, start_version, end_version)
            where start/end_version track which policy versions were active
            at the start and end of the vLLM generation call.
        """
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

            # Extract token IDs and logprobs from the response
            tok_ids: List[int] = []
            lps: List[float] = []
            if choice.logprobs and choice.logprobs.tokens:
                # Use the tokenizer to get token IDs from the tokens
                for i, token_logprob in enumerate(
                    choice.logprobs.token_logprobs or []
                ):
                    lps.append(
                        token_logprob if token_logprob is not None else 0.0
                    )

                # Tokenize the completion text to get token IDs
                encoded = self.tokenizer.encode(
                    choice.text, add_special_tokens=False,
                )
                tok_ids = encoded

                # Ensure logprobs and token_ids have matching lengths
                if len(lps) < len(tok_ids):
                    lps.extend([0.0] * (len(tok_ids) - len(lps)))
                elif len(lps) > len(tok_ids):
                    lps = lps[:len(tok_ids)]
            else:
                # Fallback: tokenize and use zero logprobs
                tok_ids = self.tokenizer.encode(
                    choice.text, add_special_tokens=False,
                )
                lps = [0.0] * len(tok_ids)

            token_ids_list.append(tok_ids)
            logprobs_list.append(lps)

        return completions, token_ids_list, logprobs_list, start_version, end_version

    # ════════════════════════════════════════════════════════════════════
    #  Prompt sampling helper
    # ════════════════════════════════════════════════════════════════════

    def _sample_prompt(self) -> Dict[str, str]:
        """
        Sample a single prompt + ground_truth from the dataset.

        Returns:
            {"prompt": str, "ground_truth": str}
        """
        idx = random.randint(0, len(self.dataset) - 1)
        item = self.dataset[idx]
        return {"prompt": item["prompt"], "ground_truth": item["ground_truth"]}

    # ════════════════════════════════════════════════════════════════════
    #  Batch assembly (used by continuous batching pool)
    # ════════════════════════════════════════════════════════════════════

    def _assemble_batch(
        self,
        items: List[tuple],
        batch_id: int,
        t_start: float,
    ) -> Batch:
        """
        Assemble a Batch from a list of completed prompt results.

        Each element of *items* is a tuple of
        ``(prompt_data, generation_result)`` where:
          - ``prompt_data`` is ``{"prompt": str, "ground_truth": str}``
          - ``generation_result`` is the 5-tuple returned by
            ``_generate_single_prompt``:
            ``(completions, token_ids, logprobs, start_ver, end_ver)``

        Performs off-policy filtering, reward computation, GRPO advantage
        computation, and tensor preparation -- the same post-processing
        that ``_generate_batch`` does, but over individually-collected
        prompt results rather than a synchronous batch.
        """
        G = self.num_generations

        prompts = [item[0]["prompt"] for item in items]
        ground_truths = [item[0]["ground_truth"] for item in items]
        results = [item[1] for item in items]
        B = len(prompts)

        # ── Off-policy filtering ──────────────────────────────────────
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

            # Re-index to keep only valid prompts
            results = [results[i] for i in valid_indices]
            prompts = [prompts[i] for i in valid_indices]
            ground_truths = [ground_truths[i] for i in valid_indices]
            all_start_versions = [all_start_versions[i] for i in valid_indices]
            all_end_versions = [all_end_versions[i] for i in valid_indices]
            B = len(prompts)
        else:
            num_discarded = 0

        # ── Flatten per-prompt results ────────────────────────────────
        all_completions: List[str] = []
        all_token_ids: List[List[int]] = []
        all_logprobs: List[List[float]] = []

        for comps, tok_ids, lps, _, _ in results:
            all_completions.extend(comps)
            all_token_ids.extend(tok_ids)
            all_logprobs.extend(lps)

        t_gen = time.time() - t_start

        # ── Post-generation stats ─────────────────────────────────────
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

        # ── Compute rewards ───────────────────────────────────────────
        expanded_gts = [gt for gt in ground_truths for _ in range(G)]
        rewards = self.reward_fn(all_completions, expanded_gts)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        # ── GRPO group-relative advantages ────────────────────────────
        advantages = compute_group_advantages(rewards_tensor, G)

        # ── Zero-variance group fraction ──────────────────────────────
        grouped_rewards = rewards_tensor.view(-1, G)
        group_std = grouped_rewards.std(dim=1)
        zero_var_frac = (group_std == 0).float().mean().item()

        # ── Tokenize & build tensors ──────────────────────────────────
        expanded_prompts = [p for p in prompts for _ in range(G)]
        batch_tensors = self._prepare_tensors(
            expanded_prompts, all_token_ids, all_logprobs,
        )

        # ── Policy version range ──────────────────────────────────────
        ver_min = min(all_start_versions) if all_start_versions else 0
        ver_max = max(all_end_versions) if all_end_versions else 0

        # ── Metrics ───────────────────────────────────────────────────
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

    # ════════════════════════════════════════════════════════════════════
    #  Tensor preparation (mirrors sync_rl trainer._prepare_grpo_inputs)
    # ════════════════════════════════════════════════════════════════════

    def _prepare_tensors(
        self,
        prompts: List[str],
        completion_token_ids: List[List[int]],
        completion_logprobs: List[List[float]],
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize prompt + completion pairs, right-pad, and build masks.

        Returns:
            input_ids:          (B*G, L)    padded token ids
            attention_mask:     (B*G, L)    1 for real tokens, 0 for padding
            loss_mask:          (B*G, L)    1 for completion tokens only
            inference_logprobs: (B*G, L-1)  vLLM logprobs aligned with shifted seq
        """
        tokenizer = self.tokenizer

        # Tokenize prompts
        prompt_encodings = tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_prompt_length,
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
