"""
Periodic evaluation during GRPO training.

Generates completions on a held-out test set using the vLLM server,
computes per-component reward breakdowns, and saves detailed results
as JSON files in ``{output_dir}/evals/``.

This lets you diagnose *why* rewards aren't increasing:
  - Is the model getting answers correct but failing format?
  - Is answer extraction working?
  - Which reward components are improving vs. degrading?

Each eval produces a JSON file like ``eval_step_000050.json`` with:
  - Aggregate summary (accuracy, mean reward, per-component means)
  - Per-problem details (question, completion, extracted answer, reward breakdown)
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Runs periodic evaluation on a held-out test set.

    Uses the same vLLM server that generates training rollouts.
    Completions are generated with low temperature (near-greedy) for
    more stable evaluation metrics.

    Parameters
    ----------
    server_base_url : str
        vLLM OpenAI-compatible API base URL (e.g. ``http://0.0.0.0:8000/v1``)
    model_name : str
        Model identifier for the API call.
    eval_dataset : list[dict]
        Test problems, each with ``"prompt"``, ``"ground_truth"``, ``"question"``.
    reward_details_fn : callable
        ``(completion, ground_truth) -> dict`` returning per-component scores.
        Must include a ``"total"`` key.
    num_problems : int
        Number of test problems to evaluate per run.
    temperature : float
        Sampling temperature for eval completions.
    max_new_tokens : int
        Max tokens per eval completion.
    output_dir : str
        Directory to save JSON eval results.
    log_router : optional
        Rich log router for formatted console output.
    """

    def __init__(
        self,
        server_base_url: str,
        model_name: str,
        eval_dataset: List[Dict[str, str]],
        reward_details_fn: Callable[[str, str], Dict[str, Any]],
        num_problems: int = 50,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        output_dir: str = "outputs/evals",
        log_router: Any = None,
    ):
        self.server_base_url = server_base_url
        self.model_name = model_name
        self.eval_dataset = eval_dataset
        self.reward_details_fn = reward_details_fn
        self.num_problems = min(num_problems, len(eval_dataset))
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.output_dir = output_dir
        self.log_router = log_router

        os.makedirs(output_dir, exist_ok=True)

        # Sample a fixed subset of eval problems (same across all evals
        # for consistent tracking across steps)
        rng = random.Random(42)
        indices = rng.sample(range(len(eval_dataset)), self.num_problems)
        self.eval_problems = [eval_dataset[i] for i in sorted(indices)]

        logger.info(
            f"Evaluator initialized: {self.num_problems} problems, "
            f"temp={self.temperature}, output_dir={self.output_dir}"
        )

    # ================================================================
    #  Public API
    # ================================================================

    def run_eval(self, step: int) -> Dict[str, Any]:
        """
        Run evaluation and save results to a JSON file.

        Returns the summary dict with aggregate metrics.
        """
        try:
            return asyncio.run(self._run_eval_async(step))
        except Exception as e:
            logger.error(f"Evaluation failed at step {step}: {e}")
            if self.log_router:
                self.log_router.log_trainer(
                    f"[bold red]EVAL step {step} FAILED: {e}[/bold red]"
                )
            return {}

    # ================================================================
    #  Async internals
    # ================================================================

    async def _run_eval_async(self, step: int) -> Dict[str, Any]:
        """Generate completions and compute rewards asynchronously."""
        client = AsyncOpenAI(
            base_url=self.server_base_url,
            api_key="dummy",
        )

        t0 = time.time()

        if self.log_router:
            self.log_router.log_trainer(
                f"[bold blue]━━━ EVAL step {step} ━━━ "
                f"generating {self.num_problems} completions...[/bold blue]"
            )

        # Generate all completions concurrently (with semaphore to avoid
        # overwhelming the server)
        semaphore = asyncio.Semaphore(32)
        tasks = [
            self._generate_with_semaphore(client, semaphore, problem["prompt"])
            for problem in self.eval_problems
        ]
        completions = await asyncio.gather(*tasks, return_exceptions=True)

        generation_time = time.time() - t0

        # Process results
        results = []
        for i, (problem, completion) in enumerate(
            zip(self.eval_problems, completions)
        ):
            if isinstance(completion, Exception):
                logger.warning(
                    f"Eval generation failed for problem {i}: {completion}"
                )
                completion = ""

            gt = problem["ground_truth"]
            question = problem.get("question", "")

            # Get detailed reward breakdown from environment
            details = self.reward_details_fn(completion, gt)

            # Separate extracted_answer (string) from numeric scores
            extracted_answer = details.pop("extracted_answer", "")

            result = {
                "question": question,
                "ground_truth": gt,
                "completion": completion,
                "extracted_answer": extracted_answer,
                "correct": details.get("correctness", 0.0) > 0,
                "reward_total": details.get("total", 0.0),
                "reward_components": {
                    k: v for k, v in details.items()
                    if k != "total" and isinstance(v, (int, float))
                },
            }
            results.append(result)

        # Compute summary statistics
        summary = self._compute_summary(results)
        summary["generation_time_s"] = round(generation_time, 1)
        summary["num_problems"] = len(results)

        # Build full output
        output = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_problems": self.num_problems,
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "model_name": self.model_name,
            },
            "summary": summary,
            "problems": results,
        }

        # Save to file
        filepath = os.path.join(self.output_dir, f"eval_step_{step:06d}.json")
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)

        # Log summary to console
        self._log_summary(step, summary, filepath)

        await client.close()
        return summary

    async def _generate_with_semaphore(
        self,
        client: AsyncOpenAI,
        semaphore: asyncio.Semaphore,
        prompt: str,
    ) -> str:
        """Generate a single completion with concurrency control."""
        async with semaphore:
            return await self._generate_completion(client, prompt)

    async def _generate_completion(
        self, client: AsyncOpenAI, prompt: str,
    ) -> str:
        """Generate a single completion via the vLLM OpenAI API."""
        try:
            messages = self._parse_chatml(prompt)

            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=max(self.temperature, 0.01),  # vLLM needs temp > 0
                n=1,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Eval generation error: {e}")
            return ""

    # ================================================================
    #  Helpers
    # ================================================================

    @staticmethod
    def _parse_chatml(prompt: str) -> List[Dict[str, str]]:
        """Parse a ChatML-formatted prompt string into OpenAI messages."""
        messages = []
        parts = prompt.split("<|im_start|>")
        for part in parts:
            if not part.strip():
                continue
            # Remove <|im_end|> and trailing whitespace
            part = part.replace("<|im_end|>", "").strip()
            if part.startswith("system\n"):
                messages.append({
                    "role": "system",
                    "content": part[len("system\n"):].strip(),
                })
            elif part.startswith("user\n"):
                messages.append({
                    "role": "user",
                    "content": part[len("user\n"):].strip(),
                })
            elif part.startswith("assistant\n"):
                content = part[len("assistant\n"):].strip()
                if content:
                    messages.append({
                        "role": "assistant",
                        "content": content,
                    })
        return messages

    @staticmethod
    def _compute_summary(results: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics from per-problem results."""
        n = len(results)
        if n == 0:
            return {"mean_reward": 0.0, "accuracy": 0.0}

        summary: Dict[str, Any] = {}

        # ── Total reward stats ────────────────────────────────────────
        rewards = [r["reward_total"] for r in results]
        summary["mean_reward"] = round(sum(rewards) / n, 4)
        summary["min_reward"] = round(min(rewards), 4)
        summary["max_reward"] = round(max(rewards), 4)

        # ── Accuracy (correctness > 0) ────────────────────────────────
        num_correct = sum(1 for r in results if r["correct"])
        summary["accuracy"] = round(num_correct / n, 4)
        summary["num_correct"] = num_correct

        # ── Per-component means ───────────────────────────────────────
        all_components: set = set()
        for r in results:
            all_components.update(r["reward_components"].keys())

        for comp in sorted(all_components):
            vals = [
                r["reward_components"].get(comp, 0.0)
                for r in results
                if isinstance(r["reward_components"].get(comp), (int, float))
            ]
            if vals:
                summary[f"mean_{comp}"] = round(sum(vals) / len(vals), 4)

        # ── Answer extraction rate ────────────────────────────────────
        num_extracted = sum(
            1 for r in results if r.get("extracted_answer", "").strip()
        )
        summary["extraction_rate"] = round(num_extracted / n, 4)

        # ── Format compliance (soft_format > 0 if it exists) ─────────
        if "soft_format" in all_components:
            num_formatted = sum(
                1 for r in results
                if r["reward_components"].get("soft_format", 0) > 0
            )
            summary["format_rate"] = round(num_formatted / n, 4)

        return summary

    def _log_summary(
        self, step: int, summary: Dict, filepath: str,
    ) -> None:
        """Log evaluation summary to console and/or log router."""
        acc = summary.get("accuracy", 0)
        mean_rew = summary.get("mean_reward", 0)
        n_correct = summary.get("num_correct", 0)
        n_total = summary.get("num_problems", self.num_problems)
        gen_time = summary.get("generation_time_s", 0)

        if self.log_router:
            # Header
            self.log_router.log_trainer(
                f"[bold blue]━━━ EVAL step {step} ━━━[/bold blue] "
                f"acc [bold green]{acc:.1%}[/bold green] "
                f"({n_correct}/{n_total}) | "
                f"reward [bold yellow]{mean_rew:.3f}[/bold yellow] | "
                f"gen {gen_time:.0f}s"
            )

            # Component breakdown
            for key in sorted(summary.keys()):
                if key.startswith("mean_") and isinstance(summary[key], (int, float)):
                    comp_name = key[5:]  # strip "mean_"
                    self.log_router.log_trainer(
                        f"  [blue]{comp_name:>15}[/blue]: "
                        f"[white]{summary[key]:.3f}[/white]"
                    )

            # Extraction / format rates
            ext_rate = summary.get("extraction_rate")
            if ext_rate is not None:
                self.log_router.log_trainer(
                    f"  [blue]{'extraction_rate':>15}[/blue]: "
                    f"[white]{ext_rate:.1%}[/white]"
                )
            fmt_rate = summary.get("format_rate")
            if fmt_rate is not None:
                self.log_router.log_trainer(
                    f"  [blue]{'format_rate':>15}[/blue]: "
                    f"[white]{fmt_rate:.1%}[/white]"
                )

            self.log_router.log_trainer(
                f"  [dim]saved → {filepath}[/dim]"
            )
        else:
            logger.info(
                f"[EVAL step {step}] "
                f"accuracy={acc:.1%} ({n_correct}/{n_total}) | "
                f"mean_reward={mean_rew:.3f} | "
                f"gen_time={gen_time:.0f}s | "
                f"saved → {filepath}"
            )
            for key in sorted(summary.keys()):
                if key.startswith("mean_") and isinstance(summary[key], (int, float)):
                    logger.info(f"  {key}: {summary[key]:.4f}")
