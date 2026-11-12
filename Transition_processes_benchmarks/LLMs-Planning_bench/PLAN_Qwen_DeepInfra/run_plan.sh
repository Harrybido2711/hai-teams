#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=48:00:00
#SBATCH --job-name=qwen_plan
#SBATCH --output=log_%a.txt
#SBATCH --error=log_%a.err

# Five shards of one PlanBench sample, all running at once — 5 concurrent streams is this project's
# standing ceiling (quest-cluster.md). Shard N writes <results dir>/<model>_shardNof5.jsonl; the
# default sample is the 10% pilot (1,563 rows, results_pilot10/). Arguments after the script name
# reach the runner, e.g. `sbatch run_plan.sh --sample full` — but the full set is 15,568 rows, ~3,100
# a shard, and needs its wall clock re-sized from the pilot's latencies before it is submitted.
#
# Submit from inside this folder. Every shard rewrites the health files when it exits; the sample is
# whole when <model>_health_overall.csv says complete 1. Rebuild them by hand with:
#   python qwen_di_plan_eval.py --model Qwen/Qwen3.5-9B --health-only
# Scoring is NOT done here: it needs VAL and tarski, and runs offline (../plan_scorer.py).

module purge
export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python qwen_di_plan_eval.py \
    --model Qwen/Qwen3.5-9B \
    --sample pilot10 --shard "$SLURM_ARRAY_TASK_ID" --total-shards 5 "$@"
