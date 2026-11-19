#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=06:00:00
#SBATCH --job-name=gpt56luna_bbh_s10
#SBATCH --output=/projects/p32983/Tasks_benchmarks/bbh/BBH_GPT_5.6_Luna/log_sample10.txt
#SBATCH --error=/projects/p32983/Tasks_benchmarks/bbh/BBH_GPT_5.6_Luna/log_sample10.err
#SBATCH --open-mode=append

# The 10% sample rerun (2026-11-15; .claude/references/sample10-plan.md, bbh.md): the 484 items of
# ../sample10.json at this model's settled config, into results_sample10/ — never results/.
# NOT SUBMITTED: phase 3 waits for the user to verify the scripts (CLAUDE.md).
#
# The log paths above and the runner path below are absolute, so this can be submitted from any
# directory; the logs append, so a resubmit adds to them rather than wiping the earlier attempt's.
# The legacy log.txt/log.err are left alone. The runner resolves its data and output from __file__;
# it needs, beside its folder on Quest: ../bbh_eval_core.py (this change's version), ../sample10.json,
# ../data/, and ../.env holding OPENAI_API_KEY.
#
# One serial job, and no --workers: the SIGALRM watchdog only works on the main thread, so the core
# refuses --workers above 1 for --sample; 484 items do not need more (quest-cluster.md).
# Wall time: the local smoke ran 60 rows in 133 s (2026-11-15), ~18 min for 484 — a wide margin.
# Arguments after the script name reach the runner: `sbatch run_bbh_sample10.sh --limit 3` is a
# pilot into smoke_sample10/, whose halt markers are smoke_sample10_*_HALT.txt — the real run's are
# sample10_*_HALT.txt, in the model folder, and a pilot never clears them.
# Exit 1: a row failed every attempt, or was left unfinished at the cap — resubmit: resume calls
# exactly those rows and keeps their history. Exit 2: a forced answer filled its cap or reasoned
# again — read those rows.

module purge
export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python /projects/p32983/Tasks_benchmarks/bbh/BBH_GPT_5.6_Luna/gpt56luna_bbh_eval.py \
    --model gpt-5.6-luna \
    --task all \
    --sample sample10 \
    --prompt v1 \
    "$@"
