#!/bin/bash
# Run NegotiationToM evaluation with GPT-4o-mini
# Usage: bash run_negotiation.sh [shard] [total_shards]
SHARD=${1:-0}
TOTAL=${2:-1}

python openai_neg_eval.py \
  --model gpt-4o-mini \
  --task all \
  --data ../NegotiationToM.json \
  --shard "$SHARD" \
  --total-shards "$TOTAL" \
  --save-every 20
