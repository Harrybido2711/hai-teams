#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --job-name=gemini_or_docvqa_s10
#SBATCH --output=/gpfs/projects/p32983/Tasks_benchmarks/DocVQA/DOCVQA_Gemini_Flash3.5lite_OpenRouter/log_sample10.txt
#SBATCH --error=/gpfs/projects/p32983/Tasks_benchmarks/DocVQA/DOCVQA_Gemini_Flash3.5lite_OpenRouter/log_sample10.err
#SBATCH --open-mode=append

# DocVQA 10% sample rerun (.claude/references/sample10-plan.md WP4) — google/gemini-3.5-flash-lite, OpenRouter.
# 539 questions from ../sample10.json, written to results_sample10/ only; results/ is never read or
# written. NOT SUBMITTED by the agent that wrote it: WP9, after the user has verified the scripts.
#
# One job, no array. DocVQA's shards share one key's daily quota (docvqa.md trap 1), and 539
# questions do not need them. Gemini's native route ran ~3 s a question; 24 h is headroom, not an estimate.
#
# Submit from inside this folder:  cd /gpfs/projects/p32983/Tasks_benchmarks/DocVQA/DOCVQA_Gemini_Flash3.5lite_OpenRouter && sbatch run_docvqa_sample10.sh
# Arguments after the script name reach the runner: `sbatch run_docvqa_sample10.sh --limit 20` is a
# pilot, written to smoke_sample10/. Resume is a plain resubmit: done rows are skipped, empty rows
# called again. Exit 1 = some rows empty (resubmit); exit 2 = a forced answer to read before scoring;
# exit 3 = thinking came back on (Qwen and Gemma: stops at the first such row, writes
# sample10_THINKING_HALT.txt; a config failure, do not resubmit).
# Halt markers land in this folder as sample10_*_HALT.txt (a --limit run's as smoke_sample10_*).
# Logs: absolute paths in THIS folder, so two models never share one, wherever sbatch runs from;
# append mode, so a resubmit adds to the last run's log instead of erasing it.

module purge
export PYTHONUNBUFFERED=1
cd /gpfs/projects/p32983/Tasks_benchmarks/DocVQA/DOCVQA_Gemini_Flash3.5lite_OpenRouter || exit 1

/projects/p32983/pythonenvs/hai-teams/bin/python gemini35lite_or_docvqa_eval.py --sample sample10 "$@"
