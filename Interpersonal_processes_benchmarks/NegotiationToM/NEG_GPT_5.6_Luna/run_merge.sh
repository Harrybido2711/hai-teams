#!/usr/bin/env bash
# Run after all 5 SLURM array jobs complete.
# Merges shard outputs, deduplicates by uid, computes final scores with the same shared scorer the
# shards used.
#
# Usage: bash run_merge.sh

/projects/p32983/pythonenvs/hai-teams/bin/python ../merge_neg_results.py \
    --model gpt-5.6-luna \
    --total-shards 5 \
    --results-root results
