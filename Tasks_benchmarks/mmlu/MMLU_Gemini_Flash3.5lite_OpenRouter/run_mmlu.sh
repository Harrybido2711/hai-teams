#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=16:00:00
#SBATCH --job-name=gemini35lite_mmlu
#SBATCH --output=log_shard%a.txt
#SBATCH --error=log_shard%a.err

# 5 shards, the project's measured ceiling (quest-cluster.md): 5 SLURM array tasks, each taking a
# contiguous fifth of every subject and writing <model>_shard<N>of5.jsonl. Concurrency does not
# raise the requests-per-DAY total -- that stays at one call per item -- only how fast it is spent.
#
# Submit from inside this folder. When all five finish, merge:
#   python ../merge_mmlu_shards.py --model google/gemini-3.5-flash-lite --model-dir . --total-shards 5
# The merge refuses to stay quiet about a missing shard: it reports which, merges what exists, and
# exits 1 so a partial number is never mistaken for a whole one.

module purge
export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gemini35lite_mmlu_eval.py \
    --model google/gemini-3.5-flash-lite \
    --subject all \
    --prompt v2 \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --total-shards 5
