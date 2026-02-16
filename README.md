# History of RLVR


## Why this repo exists

RL is hard. Anyone who has spent time in this space knows that. I think we should just accept it upfront rather than pretend otherwise.

Over the past year, I found myself going deep on RLVR. It all started with the DeepSeek R1 paper which I think genuinely changed everything for the open-source community, it showed that reasoning behaviour could emerge through RLVR in a way that felt real and reproducible, not just a benchmark trick. After that, I couldn't stop reading. GRPO, DAPO, CISPO, and the wave of papers that followed each one refining the process a little more. Some days it felt like the field was moving faster than I could keep up with.

At some point I realised I wanted one place where all of these algorithms lived, written by me, in a way I actually understood. Not a paper summary, not a borrowed codebase I half-understood — just clean implementations I had worked through myself. That's this repo.

I drew inspiration from two projects I genuinely respect:

- [`verifiers`](https://github.com/willccbb/verifiers) by [@willccbb](https://x.com/willccbb) — gold standard.
- [`ludic`](https://github.com/hallerite/ludic) by [@hallerite](https://x.com/hallerite) — which demonstrated how a clean algorithm-swapping interface could work in practice, with a clear separation between agent, environment, and the RL algorithm itself.

Both of them made me think harder about what a good abstraction looks like here.

The design philosophy is intentionally constrained: **this repo is not a framework for writing new algorithms**. It's for *using* the ones that already exist. You bring your environment — the RL task and the SFT setup — and you pick whichever algorithm fits your needs. That's the whole contract.

Currently the implementations are in place and working. I'll keep adding algorithms as the field evolves, because honestly, it doesn't look like it's slowing down anytime soon.

---

## Algorithms

| Algorithm | Reference | Key Idea |
|-----------|-----------|----------|
| **GRPO** | [DeepSeek-Math](https://arxiv.org/abs/2402.03300) | Clipped surrogate + KL penalty, token-level averaging |
| **Dr. GRPO** | [Understanding R1-Zero](https://arxiv.org/abs/2503.20783) | Unbiased GRPO -- removes length and std normalization biases |
| **DAPO** | [ByteDance](https://arxiv.org/abs/2503.14476) | Asymmetric clipping, no KL, dynamic sampling, overlong penalty |
| **CISPO** | [MiniMax-M1](https://arxiv.org/abs/2506.13585) | Clips IS-weight directly with stop-gradient |
| **Prime** | [Prime Intellect](https://github.com/PrimeIntellect-ai/prime-rl) | Token-level ratio masking with stop-gradient coefficient |
| **GSPO** | [Qwen3](https://arxiv.org/abs/2507.18071) | Sequence-level importance ratio and clipping |


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

## To-do

- Multi turn chats
- Add implementation for SAPO, GMPO.
- Add implementation for REINFORCE algorithm.
- Add on-policy like pointwise.

## Requirements

```bash
pip install -r requirements.txt
```
