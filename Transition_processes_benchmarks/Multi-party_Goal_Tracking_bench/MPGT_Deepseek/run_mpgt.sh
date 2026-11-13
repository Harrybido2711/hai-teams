#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=12:00:00
#SBATCH --job-name=deepseek_mpgt
#SBATCH --output=log_%j.txt
#SBATCH --error=log_%j.err

# mpgt-eval zero-shot — deepseek-flash: DST + goal x six upstream prompt styles x 29 dialogues = 348 rows, into
# results/zero-shot/. Reasoning budget 8192: a row that reaches it gets a second, forced-answer call
# (see the runner's docstring). One job, one stream. Submit from inside this folder. Arguments reach
# the runner: a smoke run is `<submit> run_mpgt.sh --limit 1`, written to smoke/, never results/.
# Exits non-zero unless every unit has a usable response: 2 if any row ends at `length` (a forced
# answer hit its own 4096 cap, or a cut call returned no reasoning to hand back) — checked first;
# otherwise 1 if a forced answer reasoned anyway (`budget_exceeded`) or any row is not `ok`.
# Needs ../../../Action_processes_benchmarks/judge_core.py on Quest beside this benchmark — the call
# layer is imported from there, so it is synced with this folder.

module purge
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR" || exit 1

/projects/p32983/pythonenvs/hai-teams/bin/python deepseek_mpgt_eval.py "$@"
