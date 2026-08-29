# EmoBench — benchmark card

Affect management. Upstream `Sahandfer/EmoBench`. No LLM judge — multiple-choice accuracy. Six
providers have results.

## Paths

| | Path |
|---|---|
| Local | `Interpersonal_processes_benchmarks/EmoBench` |
| Quest | `/gpfs/projects/p32983/Interpersonal_processes_benchmarks/EmoBench` — **the remote name is not the local one** |

## Layout

```
EmoBench/
├── EMO_{Gemini_Flash2.5,Gemma,Qwen,Deepseek,XAI}/  <provider>_emo_eval.py · run_emobench.sh · results/
├── EMO_Gemini_Flash3.5lite_{Google,OpenRouter}/    two routes to one model, written 2026-08-22
├── OpenAI_result/                          the sixth provider, named differently
├── data/EA.jsonl · data/EU.jsonl           the two tasks
├── src/ · Output template/                 vendored upstream, and the result templates
└── EMO_SCRIPT.md
```

There is no shared core module here — each runner is self-contained, so unlike NegotiationToM a
runner can be transferred alone.

## Expected counts

**200 scored items per task, 400 total.** Each `data/*.jsonl` holds 400 lines, but **half are
Chinese and every runner filters to `language == "en"`** — a run of 200 per task is complete, not
truncated. Confirmed against the finished 2.5 run: 200 rows in each of `results/EA` and `results/EU`
(verified 2026-08-22). Tasks are `EA` and `EU`.

Counting the file instead of the English subset is the mistake here: it makes a finished run look
half done.

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

## Reasoning: upstream has a flag for it

`--use_cot` (upstream `src/main.py:32`, documented in its README) is a **supported condition, not a
prompt edit**. `src/utils.py::get_response_format` swaps `response.yaml`'s `base` statement for
`cot` and prepends a `"reasoning"` key to the JSON conditions, so the model returns its reasoning as
data. Upstream also raises its own output cap from 50 to 2048 tokens for that branch.

**It is off by default upstream, and the five finished providers ran without it** — a CoT run and a
non-CoT run are different conditions and do not share a results table. The author did not merely
stay silent: the default `base` statement says *"Do not provide any additional information or
explanations"*.

The flash-lite runners **do not hardcode that answer**. They call `reasoning_visibility.resolve()`
(in this benchmark folder), which reads `README.md:79` — *"`--use_cot`: enables chain-of-thought
reasoning. Defaults to `False`."* — and returns both the value and the line, which is written onto
every row as `use_cot_source`. `--use-cot` / `--no-use-cot` override it and are recorded as the
source instead. With the flag unset, the prompt is byte-identical to the 2.5 runner's for EA and EU,
checked rather than assumed.

Two things this changes for a runner:

- **Hidden thinking and visible reasoning are separate.** A thinking cap at the API bounds the
  invisible half; CoT makes the model write the visible half. Capping one does not suppress the
  other, so record both.
- **CoT makes the JSON fragile.** The reasoning lands inside a string field, and a model that emits
  a raw newline there breaks `json.loads`. Verified: quotes survive, unescaped newlines do not. Those
  rows must be counted as parse failures rather than scored zero.

## Its own traps

- **`EMO_SCRIPT.md` is not about EmoBench.** Despite living in this folder, it is a repo-wide script
  analysis — its first sections cover BBH and DocVQA. It also points at `../CLAUDE.md`, a path that
  does not exist. Read it for the client/retry comparison table, not as EmoBench documentation — and **not for its BBH section, which was marked superseded on 2026-08-29** when every filename in it stopped existing. bbh's live record is `../tasks/bbh-parameters.md`.
- **The model name is passed on the command line, not read from the script name.** `run_emobench.sh`
  in `EMO_Gemma/` carries `--model google/gemma-4-31B-it`; changing the folder does not change the
  model, and a copied folder that keeps the old `--model` silently evaluates the wrong one.
