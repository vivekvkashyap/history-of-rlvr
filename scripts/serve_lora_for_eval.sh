#!/bin/bash
#
# Start vLLM server with LoRA adapter for GSM8K GRPO evaluation.
#
# Your checkpoint at outputs/gsm8k_rl_grpo/checkpoint-50 is a LoRA adapter
# on base model Qwen/Qwen2.5-0.5B-Instruct.
#
# Run this script in one terminal, then in another:
#   cd history_of_rlvr
#   python scripts/eval_gsm8k.py --model_name gsm8k-lora --num_problems 1319
#
# To use full test set: --num_problems 1319 (GSM8K test has 1319 problems)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CHECKPOINT="${1:-outputs/gsm8k_rl_grpo/checkpoint-50}"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
LORA_NAME="gsm8k-lora"
PORT="${2:-8000}"

if [ ! -d "$CHECKPOINT" ]; then
  echo "Error: Checkpoint not found at $CHECKPOINT"
  exit 1
fi

# Resolve to absolute path for vLLM
CHECKPOINT_ABS="$(realpath "$CHECKPOINT")"
echo "Serving LoRA from: $CHECKPOINT_ABS"
echo "Model name for eval: $LORA_NAME"
echo "Port: $PORT"
echo ""
echo "After server starts, run in another terminal:"
echo "  python scripts/eval_gsm8k.py --model_name $LORA_NAME --num_problems 1319"
echo ""

vllm serve "$BASE_MODEL" \
  --enable-lora \
  --lora-modules "${LORA_NAME}=${CHECKPOINT_ABS}" \
  --max-lora-rank 16 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --max-model-len 1536 \
  --enforce-eager
