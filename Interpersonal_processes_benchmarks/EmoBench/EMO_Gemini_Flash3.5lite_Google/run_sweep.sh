#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=4:00:00
#SBATCH --job-name=g35lite_sweep
#SBATCH --output=log_sweep_%x_%j.txt
#SBATCH --error=log_sweep_%x_%j.err

# One arm of the parameter sweep. TAG, TEMP and BUDGET come from --export.
# Every arm pins --seed so a score difference is the parameter, not the sampler:
# without a seed this model returned two different answers to the same item in three calls.
module purge

ARGS=(--model gemini-3.5-flash-lite --task EU --save-every 25
      --max-output-tokens 8192 --seed 42 --tag "$TAG" --thinking-budget "$BUDGET")
if [ -n "$TEMP" ]; then ARGS+=(--temperature "$TEMP"); fi

/projects/p32983/pythonenvs/hai-teams/bin/python -u gemini35lite_emo_eval.py "${ARGS[@]}"
