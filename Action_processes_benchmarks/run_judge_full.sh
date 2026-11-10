#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=48:00:00
#SBATCH --job-name=judge_full
#SBATCH --output=judge_full_log_%A_%a.txt
#SBATCH --error=judge_full_log_%A_%a.err

# Full run — the chosen judge (Claude Haiku 5.5, the user's decision of 2026-10-31) grades every
# selected model's responses on the full sets. One array task per evaluated model: five streams on
# one OpenRouter key, at the five-stream ceiling in quest-cluster.md. Each task runs
#   MultiChallenge (273 items) · Wonderbread QA (120 answers) · Wonderbread SOP (486 SOPs)
# for its model; Deepseek has no Wonderbread folder (text-only, the user's decision) and runs
# MultiChallenge alone. SOP is ~90% of the calls (one per line, both directions, x N_REPS).
# A stage that fails or halts does not stop the next; the task exits non-zero if any stage did.
# Submit from this folder, after the generation jobs: `--dependency=afterok:<job>:<job>...`. One
# afterok over all nine jobs means one model's exit 1 or 2 holds back all five; if one fails, submit
# the others with `--array=<indices>` and that model's index once it is fixed.
# Arguments reach every stage: `<submit> run_judge_full.sh --limit 2` is a smoke run into smoke/.

module purge
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR" || exit 1
PY=/projects/p32983/pythonenvs/hai-teams/bin/python

# The judge is the runners' default, judge_core.JUDGE — named in one place, not here.
# Index 5, Qwen_DeepInfra, is Wonderbread only: Qwen's Wonderbread rerun on DeepInfra (the user's
# decision, 2026-11-09). Its MultiChallenge stays MC_Qwen (Together), so there is no MC_ folder and
# that stage is skipped. Before submitting it, archive the Together run's Qwen-Qwen3.5-9B.jsonl in
# WB_Judges/results/full/<judge>/{qa,sop}/ — both producers write that name, and load_done refuses
# to resume across their configs.
MODELS=(GPT_5.6_Luna Gemini_Flash3.5lite_OpenRouter Qwen Gemma_DeepInfra Deepseek Qwen_DeepInfra)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "model=$M args=$*"
fail=0
run() {
    echo "=== $(date +%H:%M:%S) $*"
    "$PY" "$@" || { echo "=== FAILED (exit $?): $*"; fail=1; }
}
if [ -d "Multi-challenge_bench/MC_$M" ]; then
    run Multi-challenge_bench/MC_Judges/mc_judge.py --set full --producer "MC_$M" "$@"
else
    echo "=== no Multi-challenge_bench/MC_$M — Wonderbread only"
fi
if [ -d "Wonderbread_bench/WB_$M" ]; then
    run Wonderbread_bench/WB_Judges/wb_judge.py --task qa --set full --producer "WB_$M" "$@"
    run Wonderbread_bench/WB_Judges/wb_judge.py --task sop --set full --producer "WB_$M" "$@"
else
    echo "=== no Wonderbread_bench/WB_$M — MultiChallenge only"
fi
echo "=== $(date +%H:%M:%S) model=$M done, fail=$fail"
exit $fail
