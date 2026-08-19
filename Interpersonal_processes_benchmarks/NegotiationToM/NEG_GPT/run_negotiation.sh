#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=7-00:00:00
#SBATCH --job-name=neg_gpt
#SBATCH --output=log_shard%a.txt
#SBATCH --error=log_shard%a.err

# Full NegotiationToM run: 2380 dialogues split into 5 shards.
# --array=0-4 is a shard-count choice, not a Quest queue ceiling: it was long assumed to be
# "Quest allows at most 5 parallel array jobs" and all six models' sbatch scripts inherited
# --array=0-4 on that belief, but NEG_Gemma's per-task split (2026-08-05, jobs 8625800/8625801/
# 8625810) ran 15 array tasks from 3 separate submissions concurrently on 8 distinct compute
# nodes (sacct: all 15 started within 7s of each other, 04:14:26-04:14:33) with none queued
# behind another. See NegotiationToM/ISSUES.md, "Per-task arrays, not a bigger shard count" entry.
# Each job runs all 3 tasks (desire + belief + intention) for its shard.
#
# Run run_pilot.sh first and check log_pilot.txt before submitting this.
# After all 5 shards finish: bash run_merge.sh
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gpt_neg_eval.py \
    --model gpt-4o-mini \
    --task all \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --total-shards 5 \
    --save-every 20
