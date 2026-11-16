#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --job-name=luna_docvqa_s10
#SBATCH --output=/gpfs/projects/p32983/Tasks_benchmarks/DocVQA/DOCVQA_GPT_5.6_Luna/log_sample10.txt
#SBATCH --error=/gpfs/projects/p32983/Tasks_benchmarks/DocVQA/DOCVQA_GPT_5.6_Luna/log_sample10.err
#SBATCH --open-mode=append

# DocVQA 10% sample rerun (.claude/references/sample10-plan.md WP4) — gpt-5.6-luna, OpenAI platform.
# 539 questions from ../sample10.json, written to results_sample10/ only; results/ is never read or
# written. NOT SUBMITTED by the agent that wrote it: WP9, after the user has verified the scripts.
#
# One job, no array. DocVQA's shards share one key's daily quota (docvqa.md trap 1), and 539
# questions do not need them. The runner's 2.5 s sleep after each call is the benchmark's quota fix and stays.
#
# Submit from inside this folder:  cd /gpfs/projects/p32983/Tasks_benchmarks/DocVQA/DOCVQA_GPT_5.6_Luna && sbatch run_docvqa_sample10.sh
# Arguments after the script name reach the runner: `sbatch run_docvqa_sample10.sh --limit 20` is a
# pilot, written to smoke_sample10/. Resume is a plain resubmit: done rows are skipped, empty rows
# called again. Exit 1 = some rows empty (resubmit); exit 2 = a forced answer to read before scoring;
# exit 3 = thinking came back on (Gemma only: a config failure, do not resubmit).
# Halt markers land in this folder as sample10_*_HALT.txt (a --limit run's as smoke_sample10_*).
# Logs: absolute paths in THIS folder, so two models never share one, wherever sbatch runs from;
# append mode, so a resubmit adds to the last run's log instead of erasing it.

module purge
export PYTHONUNBUFFERED=1
cd /gpfs/projects/p32983/Tasks_benchmarks/DocVQA/DOCVQA_GPT_5.6_Luna || exit 1

/projects/p32983/pythonenvs/hai-teams/bin/python gpt56luna_docvqa_eval.py --sample sample10 "$@"
