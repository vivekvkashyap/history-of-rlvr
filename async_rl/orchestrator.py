"""
Asynchronous batch generation orchestrator for GRPO training.

Runs a daemon thread with its own asyncio event loop that generates
completions via the vLLM OpenAI-compatible API, computes rewards and
GRPO group-relative advantages, then packages the result into a Batch
object that the trainer thread can consume.

Communication with the main (trainer) thread uses two queues:
  - request_queue: trainer submits batch IDs to generate
  - result_queue:  worker delivers completed Batch objects

The trainer can check ``orchestrator.is_generating`` to know whether
a generation request is in flight (used to coordinate weight sync --
we don't want to update weights while the server is mid-generation).

Simplified from the verifiers-rl orchestrator: no Environment
abstraction, no multi-step trajectories -- just single-turn GSM8K.
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
from typing import Any, Dict, List, Optional

import httpx
import torch
from openai import AsyncOpenAI

# Parent dir for cross-package imports (history_of_rlvr/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl.grpo import compute_group_advantages
from sync_rl.data import GSM8KDataset
from sync_rl.reward import compute_rewards_batch

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
    """
    batch_id: int

    # ── Tensors for training (built by the orchestrator) ───────────────
    input_ids: torch.Tensor          # (B*G, L)   padded token ids
    attention_mask: torch.Tensor     # (B*G, L)   1 for real, 0 for padding
    loss_mask: torch.Tensor          # (B*G, L)   1 for completion tokens
    inference_logprobs: torch.Tensor # (B*G, L-1) vLLM logprobs aligned
    advantages: torch.Tensor         # (B*G,)     group-relative advantages

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

    Usage from the trainer:
        orchestrator = Orchestrator(...)
        orchestrator.start()
        orchestrator.submit_batch(0)          # kick off first batch

        for step in range(max_steps):
            orchestrator.submit_batch(step + 1)  # pipeline next batch
            batch = orchestrator.get_batch(step)  # block for current
            ... training on batch ...

        orchestrator.stop()
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
        # ── Dataset ────────────────────────────────────────────────────
        dataset: GSM8KDataset,
        # ── Tokenizer (for prompt tokenisation) ────────────────────────
        tokenizer: Any,
        max_prompt_length: int = 512,
        # ── Timeouts ──────────────────────────────────────────────────
        generation_timeout: float = 600.0,
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
        self.batch_size = batch_size
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.generation_timeout = generation_timeout
        self.log_router = log_router

        # Queues for thread communication
        self.request_queue: queue.Queue[Optional[int]] = queue.Queue()
        self.result_queue: queue.Queue[Batch] = queue.Queue()
        self.completed_batches: Dict[int, Batch] = {}

        # State flag: True while a generation request is being processed
        self.is_generating = False

        # Thread management
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.worker_loop: Optional[asyncio.AbstractEventLoop] = None
        self.client: Optional[AsyncOpenAI] = None

    # ════════════════════════════════════════════════════════════════════
    #  Public API (called from the trainer thread)
    # ════════════════════════════════════════════════════════════════════

    def start(self) -> None:
        """Start the async generation worker thread."""
        self.worker_thread = threading.Thread(
            target=self._generation_worker,
            daemon=True,
            name="AsyncBatchGenerator",
        )
        self.worker_thread.start()

    def stop(self) -> None:
        """Stop the worker thread."""
        self.stop_event.set()
        self.request_queue.put(None)  # poison pill
        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)

    def submit_batch(self, batch_id: int) -> None:
        """Submit a batch generation request (non-blocking)."""
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
    #  Batch generation (runs inside the worker's asyncio loop)
    # ════════════════════════════════════════════════════════════════════

    async def _generate_batch(self, batch_id: int) -> Batch:
        """
        Generate a full batch of completions via the vLLM server.

        Steps:
          1. Sample batch_size prompts from GSM8K
          2. For each prompt, request num_generations completions
          3. Compute binary rewards (math answer verification)
          4. Compute GRPO group-relative advantages
          5. Tokenize & package into tensors for the trainer
        """
        self.is_generating = True
        assert self.client is not None
        t_start = time.time()

        G = self.num_generations

        # ── 1. Sample prompts ──────────────────────────────────────────
        n = len(self.dataset)
        indices = random.sample(range(n), min(self.batch_size, n))
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
        all_completions: List[str] = []
        all_token_ids: List[List[int]] = []
        all_logprobs: List[List[float]] = []

        tasks = []
        for prompt in prompts:
            tasks.append(
                self._generate_single_prompt(prompt, G)
            )

        results = await asyncio.gather(*tasks)

        for comps, tok_ids, lps in results:
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
        rewards = compute_rewards_batch(all_completions, expanded_gts)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        # ── 4. GRPO group-relative advantages ──────────────────────────
        advantages = compute_group_advantages(rewards_tensor, G)

        # ── 5. Tokenize & build tensors ────────────────────────────────
        expanded_prompts = [p for p in prompts for _ in range(G)]
        batch_tensors = self._prepare_tensors(
            expanded_prompts, all_token_ids, all_logprobs,
        )

        self.is_generating = False

        # ── Metrics ────────────────────────────────────────────────────
        metrics = {
            "rewards/mean": rewards_tensor.mean().item(),
            "rewards/std": rewards_tensor.std().item(),
            "rewards/advantage_mean": advantages.mean().item(),
            "train/completion_len_mean": avg_comp_len,
            "train/completion_len_max": max_comp_len,
            "train/generate_s": t_gen,
        }

        return Batch(
            batch_id=batch_id,
            input_ids=batch_tensors["input_ids"],
            attention_mask=batch_tensors["attention_mask"],
            loss_mask=batch_tensors["loss_mask"],
            inference_logprobs=batch_tensors["inference_logprobs"],
            advantages=advantages,
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
            (completions, token_ids, logprobs) for the n completions
        """
        assert self.client is not None

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

        return completions, token_ids_list, logprobs_list

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
