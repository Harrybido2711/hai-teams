#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=04:00:00
#SBATCH --job-name=gemini_mpgt
#SBATCH --output=log_%j.txt
#SBATCH --error=log_%j.err

# mpgt-eval zero-shot — gemini-3.5-flash-lite via OpenRouter: DST + goal x six upstream prompt styles x 29 dialogues = 348 calls, into
# results/zero-shot/. One job, one stream. Submit from inside this folder. Arguments reach the runner:
# a smoke run is `<submit> run_mpgt.sh --limit 1`, written to smoke/, never results/. Exits non-zero
# unless every unit has a usable response, and 2 if any hit the output cap.
# Needs ../../../Action_processes_benchmarks/judge_core.py on Quest beside this benchmark — the call
# layer is imported from there, so it is synced with this folder.

module purge
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR" || exit 1

/projects/p32983/pythonenvs/hai-teams/bin/python gemini35lite_mpgt_eval.py "$@"
