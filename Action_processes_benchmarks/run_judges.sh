#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=24:00:00
#SBATCH --job-name=judge_pilot
#SBATCH --output=judge_log_%A_%a.txt
#SBATCH --error=judge_log_%A_%a.err

# Judge pilot — one array task per candidate judge (four different models, so four streams: under
# the five-stream ceiling in quest-cluster.md). Each task runs every judged set for its judge:
#   MultiChallenge sample · Wonderbread QA anchor, QA sample · SOP anchor, SOP sample
# A stage that fails or halts (e.g. REFUSAL_HALT) does not stop the next; the task exits non-zero
# if any stage did. Each judge writes markers only under its own results/<judge>/ folders.
# Submit from this folder, after the generation jobs: `--dependency=afterok:<mc job>:<wb job>`.
# Arguments reach every stage: `<submit> run_judges.sh --limit 2` is a smoke run into smoke/.

module purge
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR" || exit 1
PY=/projects/p32983/pythonenvs/hai-teams/bin/python

JUDGES=(haiku-4.5 haiku-5.5 kimi-k2.6 sonnet-5.5)
J=${JUDGES[$SLURM_ARRAY_TASK_ID]}
echo "judge=$J args=$*"
fail=0
run() {
    echo "=== $(date +%H:%M:%S) $*"
    "$PY" "$@" || { echo "=== FAILED (exit $?): $*"; fail=1; }
}
run Multi-challenge_bench/MC_Judges/mc_judge.py --judge "$J" "$@"
run Wonderbread_bench/WB_Judges/wb_judge.py --judge "$J" --task qa --set anchor "$@"
run Wonderbread_bench/WB_Judges/wb_judge.py --judge "$J" --task qa --set sample "$@"
run Wonderbread_bench/WB_Judges/wb_judge.py --judge "$J" --task sop --set anchor "$@"
run Wonderbread_bench/WB_Judges/wb_judge.py --judge "$J" --task sop --set sample "$@"
echo "=== $(date +%H:%M:%S) judge=$J done, fail=$fail"
exit $fail
