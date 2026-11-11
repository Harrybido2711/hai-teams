#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=12:00:00
#SBATCH --job-name=judge_wb_shard
#SBATCH --output=judge_wb_shard_log_%A_%a.txt
#SBATCH --error=judge_wb_shard_log_%A_%a.err

# One Wonderbread judge stage split into 5 shards, one array task each (the user's decision, 2026-11-11:
# a single stream took ~24 h on Qwen's 33,924 SOP line calls). Task k judges the SOPs at position
# k mod 5 of the set — whole SOPs — into <producer>_shard<k>of5.jsonl, and READS the canonical file for
# what an earlier unsharded run already finished, so nothing is judged twice. The array size and
# SHARDS must agree. Shards hold a shared lock on the canonical file for their life: a plain run or the
# merge refuses meanwhile, and a shard refuses beside a plain run. When all five finish, fold them in:
#
#   sbatch --export=ALL,PRODUCER=WB_Qwen_DeepInfra,TASK=sop --dependency=afterok:<this array's job id> run_judge_wb_merge.sh
#
# (afterok: if a shard fails, resubmit it with `--array=<k>` and submit the merge after that job.)
# Defaults are Qwen's DeepInfra rerun, SOP stage; override with `sbatch --export=ALL,TASK=qa,PRODUCER=...`
# (the merge script REQUIRES the same two variables). Submit from this folder.

module purge
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR" || exit 1
PY=/projects/p32983/pythonenvs/hai-teams/bin/python
PRODUCER=${PRODUCER:-WB_Qwen_DeepInfra}
TASK=${TASK:-sop}
SHARDS=5
[ "$SLURM_ARRAY_TASK_ID" -lt "$SHARDS" ] || { echo "array index $SLURM_ARRAY_TASK_ID >= $SHARDS shards"; exit 1; }
echo "producer=$PRODUCER task=$TASK shard=$SLURM_ARRAY_TASK_ID/$SHARDS args=$*"
exec "$PY" Wonderbread_bench/WB_Judges/wb_judge.py --task "$TASK" --set full --producer "$PRODUCER" \
    --shard "$SLURM_ARRAY_TASK_ID/$SHARDS" "$@"
