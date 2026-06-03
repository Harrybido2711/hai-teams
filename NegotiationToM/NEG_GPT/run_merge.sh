#!/usr/bin/env bash
# Run after all 5 SLURM array jobs complete.
# Merges shard outputs, computes final scores, and prints a summary.
#
# Usage: bash run_merge.sh

/projects/p32983/pythonenvs/hai-teams/bin/python merge_shards.py \
    --model gpt-4o-mini \
    --total-shards 5 \
    --task all \
    --results-root results
