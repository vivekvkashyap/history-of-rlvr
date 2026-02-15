#!/usr/bin/env python3
"""
Standalone GSM8K evaluation script.

Evaluates a model on the GSM8K test set. Contains its own reward logic
(extracted from gsm8k_rl) and GSM8K-specific hyperparameters. No dependency
on environments.gsm8k_rl or environments.base.

Requires vLLM server running separately. Start it first, then run this script.

Usage for GRPO/DAPO LoRA checkpoint (outputs/gsm8k_rl_grpo/checkpoint-50):

    # Option A: Serve LoRA directly with vLLM
    # Terminal 1:
    ./scripts/serve_lora_for_eval.sh outputs/gsm8k_rl_grpo/checkpoint-50
    # Terminal 2:
    python scripts/eval_gsm8k.py --model_name gsm8k-lora --num_problems 1319

    # Option B: Merge LoRA first, then serve merged model
    python scripts/merge_lora_checkpoint.py outputs/gsm8k_rl_grpo/checkpoint-50
    vllm serve outputs/gsm8k_rl_grpo/merged-50 --port 8000 ...
    python scripts/eval_gsm8k.py --model_name outputs/gsm8k_rl_grpo/merged-50 --num_problems 1319

For base model only:
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 ...
    python scripts/eval_gsm8k.py --model_name Qwen/Qwen2.5-0.5B-Instruct --num_problems 500
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from openai import AsyncOpenAI

# Add project root for optional LogRouter
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ════════════════════════════════════════════════════════════════════════
#  GSM8K reward logic (from gsm8k_rl.py)
# ════════════════════════════════════════════════════════════════════════

_ANSWER_PATTERN = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)

GSM8K_SYSTEM_PROMPT = (
    "You are a careful math tutor. Think step-by-step. When you are ready, "
    "write the final numeric answer inside \\boxed{}, like \\boxed{42}."
)


def _extract_ground_truth(answer_str: str) -> str:
    """Extract the final numeric answer from a GSM8K answer string."""
    if "####" not in answer_str:
        return ""
    return answer_str.split("####")[1].strip().replace(",", "").replace("$", "")


def _extract_answer_from_boxed(completion: str) -> str:
    """Extract answer from \\boxed{...}."""
    match = _ANSWER_PATTERN.search(completion)
    if match:
        return match.group(1).strip().replace(",", "").replace("$", "")
    return ""


def _normalize_number(s: str) -> str:
    """Normalize a numeric string for comparison (e.g. '72.0' → '72')."""
    s = s.strip().replace(",", "").replace("$", "")
    if not s:
        return ""
    try:
        val = float(s)
        if not (val == val) or abs(val) == float("inf"):
            return s
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return s


def compute_reward_details(completion: str, ground_truth: str) -> Dict[str, Any]:
    """
    GSM8K reward: Binary correctness (1.0 or 0.0) from \\boxed{}.
    Format reward tracked but disabled via reward_coef=0.0.
    """
    extracted = _extract_answer_from_boxed(completion)
    is_correct = _normalize_number(extracted) == _normalize_number(ground_truth)
    has_format = _ANSWER_PATTERN.search(completion) is not None

    correctness = 1.0 if is_correct else 0.0
    format_reward = 0.5 if has_format else -0.5
    reward_coef = 0.0
    total = correctness + (reward_coef * format_reward)

    return {
        "correctness": correctness,
        "format_reward": format_reward,
        "reward_coef": reward_coef,
        "total": total,
        "extracted_answer": extracted,
        "has_boxed_format": has_format,
    }


# ════════════════════════════════════════════════════════════════════════
#  GSM8K hyperparameters
# ════════════════════════════════════════════════════════════════════════

GSM8K_DEFAULTS = {
    "dataset_name": "openai/gsm8k",
    "dataset_config": "main",
    "max_new_tokens": 512,
    "temperature": 0.0,  # near-greedy for stable eval
    "num_problems": 500,
}


async def _generate_one(
    client: AsyncOpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate a single completion. Pass messages directly; vLLM applies model's chat template."""
    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=max(temperature, 0.01),
        n=1,
    )
    return response.choices[0].message.content or ""


async def _run_eval(
    server_base_url: str,
    model_name: str,
    eval_problems: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    output_dir: str,
) -> Dict[str, Any]:
    """Run evaluation asynchronously."""
    client = AsyncOpenAI(base_url=server_base_url, api_key="dummy")
    t0 = time.time()

    print(f"Generating {len(eval_problems)} completions...")
    semaphore = asyncio.Semaphore(32)

    async def _gen(p):
        async with semaphore:
            return await _generate_one(
                client, model_name, p["messages"], max_new_tokens, temperature
            )

    completions = await asyncio.gather(
        *[_gen(p) for p in eval_problems],
        return_exceptions=True,
    )
    generation_time = time.time() - t0

    results = []
    for problem, completion in zip(eval_problems, completions):
        if isinstance(completion, Exception):
            completion = ""
        gt = problem["ground_truth"]
        details = compute_reward_details(str(completion), gt)
        extracted = details.pop("extracted_answer", "")
        results.append({
            "question": problem.get("question", ""),
            "ground_truth": gt,
            "completion": str(completion),
            "extracted_answer": extracted,
            "correct": details.get("correctness", 0.0) > 0,
            "reward_total": details.get("total", 0.0),
            "reward_components": {
                k: v for k, v in details.items()
                if k != "total" and isinstance(v, (int, float))
            },
        })

    n = len(results)
    if n == 0:
        await client.close()
        return {"accuracy": 0.0, "num_problems": 0}, ""

    num_correct = sum(1 for r in results if r["correct"])
    rewards = [r["reward_total"] for r in results]
    num_extracted = sum(1 for r in results if r.get("extracted_answer", "").strip())

    summary = {
        "mean_reward": round(sum(rewards) / n, 4),
        "min_reward": round(min(rewards), 4),
        "max_reward": round(max(rewards), 4),
        "accuracy": round(num_correct / n, 4),
        "num_correct": num_correct,
        "extraction_rate": round(num_extracted / n, 4),
        "generation_time_s": round(generation_time, 1),
        "num_problems": n,
    }
    for comp in sorted(set().union(*(r["reward_components"].keys() for r in results))):
        vals = [
            r["reward_components"].get(comp, 0.0)
            for r in results
            if isinstance(r["reward_components"].get(comp), (int, float))
        ]
        if vals:
            summary[f"mean_{comp}"] = round(sum(vals) / len(vals), 4)

    output = {
        "step": 0,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_problems": n,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "model_name": model_name,
            "dataset_name": GSM8K_DEFAULTS["dataset_name"],
        },
        "summary": summary,
        "problems": results,
    }

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "eval_gsm8k.json")
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    await client.close()
    return summary, filepath


def main():
    parser = argparse.ArgumentParser(
        description="Standalone GSM8K evaluation (no gsm8k_rl dependency)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name (must match vLLM server)",
    )
    parser.add_argument(
        "--vllm_server_host",
        type=str,
        default="0.0.0.0",
        help="vLLM server host",
    )
    parser.add_argument(
        "--vllm_server_port",
        type=int,
        default=8000,
        help="vLLM server port",
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=GSM8K_DEFAULTS["num_problems"],
        help="Number of test problems to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=GSM8K_DEFAULTS["temperature"],
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=GSM8K_DEFAULTS["max_new_tokens"],
        help="Max tokens per completion",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gsm8k_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=GSM8K_DEFAULTS["dataset_name"],
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=GSM8K_DEFAULTS["dataset_config"],
        help="Dataset config",
    )
    args = parser.parse_args()

    print()
    print("═" * 60)
    print("GSM8K Evaluation (standalone)")
    print("═" * 60)
    print(f"  Model        {args.model_name}")
    print(f"  Server       {args.vllm_server_host}:{args.vllm_server_port}")
    print(f"  Num problems {args.num_problems}")
    print(f"  Temperature  {args.temperature}")
    print(f"  Max tokens   {args.max_new_tokens}")
    print(f"  Output       {args.output_dir}")
    print()

    print(f"Loading GSM8K test set ({args.dataset_name})...")
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split="test",
    )
    # Use standard messages format; vLLM applies the model's chat template
    eval_data = [
        {
            "messages": [
                {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ],
            "ground_truth": _extract_ground_truth(item["answer"]),
            "question": item["question"],
        }
        for item in ds
    ]
    print(f"  {len(eval_data)} test problems loaded")

    rng = random.Random(42)
    n = min(args.num_problems, len(eval_data))
    indices = rng.sample(range(len(eval_data)), n)
    eval_problems = [eval_data[i] for i in sorted(indices)]
    print(f"  Sampled {len(eval_problems)} for evaluation")
    print()

    server_base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}/v1"
    eval_output_dir = os.path.join(args.output_dir, "evals")
    summary, filepath = asyncio.run(
        _run_eval(
            server_base_url=server_base_url,
            model_name=args.model_name,
            eval_problems=eval_problems,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output_dir=eval_output_dir,
        )
    )

    acc = summary.get("accuracy", 0)
    n_correct = summary.get("num_correct", 0)
    n_total = summary.get("num_problems", 0)
    print()
    print(f"Done. Results saved to {filepath}")
    print(f"  Accuracy: {acc:.1%} ({n_correct}/{n_total})")
    print(f"  Mean reward: {summary.get('mean_reward', 0):.4f}")
    print(f"  Extraction rate: {summary.get('extraction_rate', 0):.1%}")
    print(f"  Generation time: {summary.get('generation_time_s', 0):.1f}s")


if __name__ == "__main__":
    main()
