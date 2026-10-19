#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=02:00:00
#SBATCH --job-name=luna_mc
#SBATCH --output=log_%j.txt
#SBATCH --error=log_%j.err

# MultiChallenge judge pilot — gpt-5.6-luna responses for the 27 sampled items. One job: 27 calls.
# Submit from inside this folder. Arguments reach the runner: a smoke run is
# `<submit> run_mc.sh --limit 3`, written to smoke/, never results/. Exits non-zero unless every
# sampled item has a usable response — the judges' job depends on that (afterok).

module purge
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR" || exit 1

/projects/p32983/pythonenvs/hai-teams/bin/python gpt56luna_mc_eval.py "$@"
