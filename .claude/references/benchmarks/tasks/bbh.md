# BIG-Bench Hard — benchmark card

General task ability. Exact match against a `"Final Answer: <answer>"` prompt contract. No LLM judge.

## Paths

| | Path |
|---|---|
| Local | `Tasks_benchmarks/bbh` |
| Quest | not verified — nothing points to a remote copy, but Quest was not checked (out of scope 2026-08-22). Confirm before assuming a remote path |

## Layout

Flat. **20 task JSONs** (`boolean_expressions.json`, `causal_judgement.json`, … verified 2026-08-22),
one `<provider>_eval.py` and `<provider>_eval_script.sh` per provider, and results as
`<provider>_<task>.csv` beside them. A `bin/` directory is also present.

## Provider coverage — the files on disk are not the answer

Result CSVs exist for **eight** prefixes: `deepseek gemini gemma kimi llama openai qwen xai`.
`PLAN.md` says seven. Both are correct: **this benchmark was run against more models than are
reported, and some of those runs were abandoned** (user, 2026-08-22). A leftover CSV is not a
retracted result and was not deleted, so it is still on disk.

Take the coverage claim from `PLAN.md` and `Results.xlsx`, never from `ls *.csv`. bbh is the one
benchmark here where counting files overstates what was run on purpose.

## Scoring

Exact match on the answer extracted after `Final Answer:`. The prompt contract is the scoring
contract — a runner that changes the phrasing changes the score without changing the model.

## Where the client details live

`Interpersonal_processes_benchmarks/EmoBench/EMO_SCRIPT.md` §1 tables every BBH runner's model id and
SDK — despite living in the EmoBench folder, that section is about this benchmark. It is the only
written record of which client each provider uses here.
