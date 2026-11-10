#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --array=0-4
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
# MultiChallenge alone, and Qwen's two benchmarks are split across indices 2 and 5 (below). SOP is ~90% of the calls (one per line, both directions, x N_REPS).
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
#
# Each index names its MultiChallenge and Wonderbread folders explicitly; "-" means that stage does
# not run. Qwen's Wonderbread moved to DeepInfra on 2026-11-09 (the user's decision, cap 10 -> 30):
# index 2 is now MultiChallenge only (MC_Qwen, Together), and index 5 is Wonderbread only
# (WB_Qwen_DeepInfra). They must never both judge Wonderbread: both producers' judge files are named
# Qwen-Qwen3.5-9B.jsonl, and two tasks writing it would leave one model's QA from one run and its SOP
# from the other. The default array stays 0-4 (five streams, the ceiling in quest-cluster.md);
# submit index 5 on its own with `--array=5`, after archiving the Together judge files out of
# WB_Judges/{results,smoke}/full/<judge>/{qa,sop}/ (wonderbread.md, run order step 6).
LABEL=(GPT_5.6_Luna Gemini_Flash3.5lite_OpenRouter Qwen Gemma_DeepInfra Deepseek Qwen_DeepInfra)
MC=(MC_GPT_5.6_Luna MC_Gemini_Flash3.5lite_OpenRouter MC_Qwen MC_Gemma_DeepInfra MC_Deepseek -)
WB=(WB_GPT_5.6_Luna WB_Gemini_Flash3.5lite_OpenRouter - WB_Gemma_DeepInfra - WB_Qwen_DeepInfra)
i=$SLURM_ARRAY_TASK_ID
M=${LABEL[$i]}
echo "model=$M mc=${MC[$i]} wb=${WB[$i]} args=$*"
fail=0
run() {
    echo "=== $(date +%H:%M:%S) $*"
    "$PY" "$@" || { echo "=== FAILED (exit $?): $*"; fail=1; }
}
if [ "${MC[$i]}" != "-" ]; then
    run Multi-challenge_bench/MC_Judges/mc_judge.py --set full --producer "${MC[$i]}" "$@"
else
    echo "=== no MultiChallenge for this index — Wonderbread only"
fi
if [ "${WB[$i]}" != "-" ]; then
    run Wonderbread_bench/WB_Judges/wb_judge.py --task qa --set full --producer "${WB[$i]}" "$@"
    run Wonderbread_bench/WB_Judges/wb_judge.py --task sop --set full --producer "${WB[$i]}" "$@"
else
    echo "=== no Wonderbread for this index — MultiChallenge only"
fi
echo "=== $(date +%H:%M:%S) model=$M done, fail=$fail"
exit $fail
