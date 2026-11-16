#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=04:00:00
#SBATCH --job-name=deepseek_mmlu_s10
#SBATCH --output=/projects/p32983/Tasks_benchmarks/mmlu/MMLU_Deepseek/log_sample10_shard%a.txt
#SBATCH --error=/projects/p32983/Tasks_benchmarks/mmlu/MMLU_Deepseek/log_sample10_shard%a.err
#SBATCH --open-mode=append

# The 10% sample rerun (.claude/references/sample10-plan.md, WP3): the 401 items of ../sample10.json
# under this model's settled config, PROMPT_V2, every row carrying the per-row usage fields. Output:
# results_sample10/<Subject>/<model>_shard<N>of5.jsonl -- never results/. Written 2026-11-15 and NOT
# SUBMITTED: WP9 submits it, once the user has verified the scripts.
#
# 5 shards, the project's standing ceiling (quest-cluster.md): each takes a contiguous fifth of every
# subject's sampled items, ~80 rows, one stream with the SIGALRM watchdog on.
# Wall clock 4 h on `short` -- chosen, not measured on a shard:
# the final --limit 3 smoke on disk (2026-11-15) took 112 s for 39 rows, latency p50 1.78 s,
# max 36 s, no row cut. An earlier smoke had one row cut at 16384 that took 128 s over two calls;
# a row can take three calls of up to 900 s each, and the margin is for a run of those.
#
# Logs go to absolute paths in this folder, appended to rather than truncated, so a resubmitted
# shard keeps the earlier attempt's log above its own.
#
# Runs in /projects/p32983/Tasks_benchmarks/mmlu/MMLU_Deepseek
# -- it cds there itself, so it can be submitted from any directory.
# A shard that left a failed row exits 1 -- resubmit it; resume calls exactly those rows. Halt
# markers land here as sample10_shard<N>_{BILLING,QUOTA,FAILURE}_HALT.txt. When all five finish:
#   /projects/p32983/pythonenvs/hai-teams/bin/python ../merge_mmlu_shards.py \
#       --model deepseek-flash --model-dir . --total-shards 5 --sample sample10
# The merge exits 1 on a missing shard, a failed row, more than one config, or ids that are not
# exactly the sample's.

module purge
export PYTHONUNBUFFERED=1

# the runner resolves its paths from its own location, but the relative script name below
# does not: run from the folder whatever directory sbatch was called in
cd /projects/p32983/Tasks_benchmarks/mmlu/MMLU_Deepseek || exit 1

/projects/p32983/pythonenvs/hai-teams/bin/python deepseek_mmlu_sample.py \
    --model deepseek-flash \
    --sample sample10 \
    --subject all \
    --prompt v2 \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --total-shards 5
