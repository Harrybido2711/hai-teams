#!/usr/bin/env bash
# Run after all 5 SLURM array jobs complete.
# Merges shard outputs, deduplicates by uid, computes final scores.
#
# Usage: bash run_merge.sh

/projects/p32983/pythonenvs/hai-teams/bin/python ../merge_neg_results.py \
    --model gpt-4o-mini \
    --total-shards 5 \
    --results-root results
