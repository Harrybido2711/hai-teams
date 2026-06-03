#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=01:00:00
#SBATCH --job-name=openai_neg_pilot
#SBATCH --output=log_pilot.txt
#SBATCH --error=log_pilot.err

# Pilot: 50 random samples — estimates cost and verifies correctness
# before submitting the full run_negotiation.sh job.
#
# Submit: sbatch run_pilot.sh
# Then check log_pilot.txt for cost/time projections.

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python openai_neg_pilot.py \
    --model gpt-4o-mini \
    --data ../NegotiationToM.json \
    --n 50 \
    --seed 42 \
    --out-dir results/pilot
