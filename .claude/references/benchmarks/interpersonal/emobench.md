# EmoBench — benchmark card

Affect management. Upstream `Sahandfer/EmoBench`. No LLM judge — multiple-choice accuracy. Six
providers have results.

## Paths

| | Path |
|---|---|
| Local | `Interpersonal_processes_benchmarks/EmoBench` |
| Quest | `/gpfs/projects/p32983/EmoBench-master` — **the remote name is not the local one** |

## Layout

```
EmoBench/
├── EMO_{Gemini,Gemma,Qwen,Deepseek,XAI}/   <provider>_emo_eval.py · run_emobench.sh · results/
├── OpenAI_result/                          the sixth provider, named differently
├── data/EA.jsonl · data/EU.jsonl           the two tasks
├── src/ · Output template/                 vendored upstream, and the result templates
└── EMO_SCRIPT.md
```

There is no shared core module here — each runner is self-contained, so unlike NegotiationToM a
runner can be transferred alone.

## Expected counts

**400 items per task, 800 total** (`wc -l data/*.jsonl`, verified 2026-08-22). Tasks are `EA` and
`EU`. A run that produces materially fewer has skipped items, not finished early.

## Output and logs

```
<MODEL>/results/<EA|EU>/<model>_en.csv · <model>_en.jsonl · <model>_en_overall.csv
log.txt · log.err                         (the sbatch scripts write these fixed names)
```

The log filenames are fixed rather than per-shard: a second job in the same folder overwrites the
first one's log.

## Run order

`sbatch run_emobench.sh` per model folder. The script pins the model on the command line and runs
`--task all --save-every 20` under `--partition=long`, `--time=24:00:00`, 8 GB.

## Its own traps

- **`EMO_SCRIPT.md` is not about EmoBench.** Despite living in this folder, it is a repo-wide script
  analysis — its first sections cover BBH and DocVQA. It also points at `../CLAUDE.md`, a path that
  does not exist. Read it for the client/retry comparison table, not as EmoBench documentation.
- **The model name is passed on the command line, not read from the script name.** `run_emobench.sh`
  in `EMO_Gemma/` carries `--model google/gemma-4-31B-it`; changing the folder does not change the
  model, and a copied folder that keeps the old `--model` silently evaluates the wrong one.
