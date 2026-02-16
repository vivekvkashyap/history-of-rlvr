# History of RLVR

Minimal codebase tracking the progression of reinforcement learning with verifiable rewards (RLVR) for LLM reasoning. Each algorithm is implemented in a clean, self-contained file following the same pattern.

## Algorithms

| Algorithm | Reference | Key Idea |
|-----------|-----------|----------|
| **GRPO** | [DeepSeek-Math](https://arxiv.org/abs/2402.03300) | Clipped surrogate + KL penalty, token-level averaging |
| **Dr. GRPO** | [Understanding R1-Zero](https://arxiv.org/abs/2503.20783) | Unbiased GRPO -- removes length and std normalization biases |
| **DAPO** | [ByteDance](https://arxiv.org/abs/2503.14476) | Asymmetric clipping, no KL, dynamic sampling, overlong penalty |
| **CISPO** | [MiniMax-M1](https://arxiv.org/abs/2506.13585) | Clips IS-weight directly with stop-gradient |
| **Prime** | [Prime Intellect](https://github.com/PrimeIntellect-ai/prime-rl) | Token-level ratio masking with stop-gradient coefficient |
| **GSPO** | [Qwen3](https://arxiv.org/abs/2507.18071) | Sequence-level importance ratio and clipping |

All algorithms live in `rl/algorithms/` with identical signatures:

```python
def <algo>_loss(trainer_log_probs, inference_log_probs, advantages, completion_mask, ...) -> (loss, stats)
```

## Structure

```
environments/          # Task definitions (dataset + reward)
  gsm8k/               #   GSM8K math (RL + SFT)
  gpqa/                #   GPQA science QA (RL + SFT)
rl/
  algorithms/          # Loss functions (grpo, dr_grpo, dapo, cispo, prime, gspo)
  async_rl/            # Async trainer with vLLM server
  sync_rl/             # Synchronous trainer
sft/                   # Supervised fine-tuning trainer
scripts/               # Eval, LoRA merge, serving
```

## Running

### RL Training

```bash
# From the history_of_rlvr/ directory
python environments/gsm8k/gsm8k_rl.py \
    --algorithm grpo \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --max_steps 500 \
    --num_generations 16
```

Key config flags: `--algorithm` (grpo, dr_grpo, dapo, cispo, prime, gspo), `--epsilon_lower`, `--epsilon_upper`, `--beta`, `--loss_reduction` (token, sample), `--use_overlong_penalty`, `--use_dynamic_sampling`.

### SFT

```bash
python environments/gsm8k/gsm8k_sft.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --max_steps 500
```

### Eval

```bash
python scripts/eval_gsm8k.py --model_path outputs/async_rl/checkpoint-500
```

## Requirements

```bash
pip install -r requirements.txt
```
