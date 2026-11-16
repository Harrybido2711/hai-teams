#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=48:00:00
#SBATCH --job-name=qwen_di_docvqa_s10
#SBATCH --output=/gpfs/projects/p32983/Tasks_benchmarks/DocVQA/DOCVQA_Qwen_DeepInfra/log_sample10.txt
#SBATCH --error=/gpfs/projects/p32983/Tasks_benchmarks/DocVQA/DOCVQA_Qwen_DeepInfra/log_sample10.err
#SBATCH --open-mode=append

# DocVQA 10% sample rerun (.claude/references/sample10-plan.md WP4) — Qwen/Qwen3.5-9B, DeepInfra, reasoning off.
# 539 questions from ../sample10.json, written to results_sample10/ only; results/ is never read or
# written. NOT SUBMITTED by the agent that wrote it: WP9, after the user has verified the scripts.
#
# One job, no array. DocVQA's shards share one key's daily quota (docvqa.md trap 1), and 539
# questions do not need them. 48 h, not 24: DeepInfra's Qwen ran 17-46 s for a 12-token reply on 2026-11-09 (model-calls.md),
# and CutRetry can make a row three calls. Time one call before trusting this.
#
# Submit from inside this folder:  cd /gpfs/projects/p32983/Tasks_benchmarks/DocVQA/DOCVQA_Qwen_DeepInfra && sbatch run_docvqa_sample10.sh
# Arguments after the script name reach the runner: `sbatch run_docvqa_sample10.sh --limit 20` is a
# pilot, written to smoke_sample10/. Resume is a plain resubmit: done rows are skipped, empty rows
# called again. Exit 1 = some rows empty (resubmit); exit 2 = a forced answer to read before scoring;
# exit 3 = thinking came back on (Gemma only: a config failure, do not resubmit).
# Halt markers land in this folder as sample10_*_HALT.txt (a --limit run's as smoke_sample10_*).
# Logs: absolute paths in THIS folder, so two models never share one, wherever sbatch runs from;
# append mode, so a resubmit adds to the last run's log instead of erasing it.

module purge
export PYTHONUNBUFFERED=1
cd /gpfs/projects/p32983/Tasks_benchmarks/DocVQA/DOCVQA_Qwen_DeepInfra || exit 1

/projects/p32983/pythonenvs/hai-teams/bin/python qwen_di_docvqa_eval.py --sample sample10 "$@"
