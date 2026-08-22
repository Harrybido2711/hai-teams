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

## Scoring — two different scorers, not one

Every runner extracts the text after `Final Answer:`, but **four score with strict string equality
and four with a six-branch lenient matcher** that accepts `B` for `(B)`, an option's text for its
letter, and comma-vs-space differences. The four strict-scored models are penalised for formatting
rather than reasoning, and nothing in the result CSVs records which scorer produced them.

Full parameter and scoring inventory, per script with `file:line`:
[`BBH_MODEL_PARAMS.md`](../../../../Tasks_benchmarks/bbh/BBH_MODEL_PARAMS.md) in the benchmark
folder. Read it before comparing any two providers here.

## Its own traps

- **The `splits` list in five of the eight scripts no longer matches the CSVs on disk** — they were
  narrowed to failing tasks during a re-run and never restored. A CSV present is not evidence the
  current script would regenerate it.
- **One failed call discards the whole task.** A `None` response reaches `re.search` and raises
  `TypeError`, which the per-split handler catches by moving to the next split — the rows collected
  so far are dropped and no CSV is written.
- **No script sets `seed`, and one runs at `temperature=1`.** Nothing here is reproducible.

## Where the client details live

`BBH_MODEL_PARAMS.md` in the benchmark folder is authoritative. `EmoBench/EMO_SCRIPT.md` §1 also
tables the BBH runners' model ids and SDKs — it predates the inventory and covers six of the eight;
where they disagree, the inventory was read later and cites lines.
