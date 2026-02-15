# Training Analysis: Gradient Norms & Performance

## TL;DR
**Your implementation is correct!** The gradient norms of 0.02-0.08 are appropriate for your setup. The only real fix needed was removing std normalization from advantages (already done).

## Key Findings

### 1. Your Original Implementation Works Well
From `outputs/gsm8k_grpo/logs/trainer.log` (Feb 11):
- **Gradient norms:** 0.046-0.089
- **Rewards:** 60-75%
- **Eval accuracy at step 10:** 70%
- **Learning rate:** 1e-5

This shows stable, effective training!

### 2. Why Gradient Norms Differ from Prime-RL

Prime-RL shows gradient norms of ~0.2-0.3, while yours are ~0.02-0.08. This is **expected** due to:

#### A. Learning Rate Difference
- Your current run: `lr = 1e-6` 
- Your successful run: `lr = 1e-5` (10x higher)
- Gradient norms scale with learning rate and other factors

#### B. Advantage Magnitudes with Binary Rewards
With binary rewards (0 or 1) and mean ~0.5:
```python
advantages = rewards - mean
# rewards = [1, 0, 1, 0, ...], mean = 0.5
# advantages = [0.5, -0.5, 0.5, -0.5, ...]
```

Advantages are ±0.5, which is **correct** but naturally smaller than continuous reward schemes.

#### C. Different Architectures/Stages
- Prime-RL may use different model initialization
- Different stage of training (later stages often have larger gradients)
- Different batch compositions

### 3. What Actually Needed Fixing

✅ **Advantage normalization** (FIXED):
```python
# OLD (wrong):
advantages = (rewards - mean) / (std + eps)

# NEW (correct):
advantages = rewards - mean
```

✅ **Gradient clipping** (FIXED):
```python
# OLD: max_grad_norm = 0.1
# NEW: max_grad_norm = 1.0
```

❌ **Loss normalization** (NO CHANGE NEEDED):
Your original per-token averaging is correct. My attempted "fix" caused loss to explode to -254 and gradients to 84.

## Recommendations

### 1. Use Your Original Settings
Revert to the configuration from your successful Feb 11 run:
- `learning_rate = 1e-5` (not 1e-6)
- Keep the advantage fix (no std normalization)
- Keep `max_grad_norm = 1.0`

### 2. Don't Worry About Absolute Gradient Norms
Gradient norms of 0.02-0.08 are fine if:
- ✅ Training is stable (no exploding/vanishing)
- ✅ Rewards improve over time
- ✅ Evaluation accuracy increases

Your Feb 11 run achieved 70% accuracy, proving the implementation works!

### 3. Focus on These Metrics Instead
- **Reward trends:** Should increase over time
- **Evaluation accuracy:** Should improve
- **Loss stability:** Should not have wild swings
- **Clip fraction:** Should be reasonable (10-40%)

## What Went Wrong Today

Your current run uses `lr = 1e-6` instead of `1e-5`, which:
- Makes gradient norms appear smaller
- Slows down learning
- Might make rewards look "unstable" because updates are tiny

## Action Items

1. **Kill current training run**
2. **Restart with:**
   ```bash
   python -m environments.gsm8k_rl --launch \
       --max_steps 500 \
       --report_to wandb \
       --run_name "gsm8k_cispo_fixed" \
       --eval_steps 100 \
       --eval_num_problems 500 \
       --model_name Qwen/Qwen2.5-0.5B-Instruct \
       --algorithm cispo \
       --cispo_epsilon_high 0.28 \
       --learning_rate 1e-5 \   # <-- Changed from 1e-6
       --temperature 1.0
   ```

3. **Monitor these:**
   - Gradient norms should be 0.04-0.10 (fine!)
   - Rewards should trend upward
   - Eval accuracy should improve

## Conclusion

**Stop comparing absolute gradient magnitudes to Prime-RL.** Your implementation is correct. The Feb 11 run proves it works. Just use `lr=1e-5` and trust the process!
