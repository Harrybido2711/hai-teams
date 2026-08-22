# MMLU — benchmark card

General task ability, multiple choice. Accuracy. No LLM judge.

## Paths

| | Path |
|---|---|
| Local | `Tasks_benchmarks/mmlu` |
| Quest | not verified — nothing points to a remote copy, but Quest was not checked (out of scope 2026-08-22). Confirm before assuming a remote path |

## Layout

Flat, with an inconsistency worth knowing before searching for a runner:

```
mmlu/
├── <13 subject>.json                       Business_ethics, Econometrics, … (verified 2026-08-22)
├── {deepseek,gemini,openai,qwen}_eval.py   four providers at the top level
├── {deepseek,gemini,openai}_rescore.py     re-derive scores without re-calling the model
├── <provider>_<subject>.csv                results, top level
└── gemma/ · xai/ · openai/ · llama/        the other providers live in subdirectories instead
```

**A provider missing from the top level is not a provider that was never run** — four of them are one
directory down. `ls */` before concluding anything about coverage. Those four subdirectories are
**named after models, so they are ours**; `dotenv/` is not a model and is not results.

## Expected counts

13 subject files. Per-subject item counts are not recorded anywhere in the repo; count them from the
JSON rather than assuming MMLU's published totals, since this is a subset.

## Its own traps

- **The rescore scripts are the cheap path.** A scoring change does not need a rerun: `*_rescore.py`
  re-derives the numbers from stored responses. Check for one before spending a provider call.
- **A `dotenv/` directory sits among the provider subdirectories** — it is not a provider.
