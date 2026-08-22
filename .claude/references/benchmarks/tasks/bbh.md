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

Every generation parameter each runner sets, with `file:line`:
[bbh-parameters.md](bbh-parameters.md). Read it before comparing any two providers here — it is also
where the two-file caveat above is spelled out per model.

## Its own traps

- **Two of the eight `.sh` files do not submit the file their name implies.** `gemma_eval_script.sh`
  runs `gemma_finish.py` and `qwen_eval_script.sh` runs `qwen_finish.py`, and the `_finish` twins
  differ from the `_eval` ones in `max_tokens`, client timeout, retry behaviour and task list. **Read
  the `.sh` before reading any runner here** — documenting these two from the `_eval.py` file gets
  four columns wrong, which is exactly how the first version of the inventory came out wrong.
- **The `splits` list in four of the eight submitted scripts is a narrowed re-run list** that was
  never restored. A CSV present is not evidence the script beside it would regenerate it, and no
  result file records which script version wrote it.
- **One failed call discards a whole task, in seven of eight.** A `None` response reaches `re.search`
  and raises `TypeError`; the per-split handler catches it and moves to the next split, dropping the
  rows collected so far without writing a CSV. `gemma_finish.py` returns `""` and is the exception.
- **No script sets `seed`, and one runs at `temperature=1`.** Nothing here is reproducible.

## Where the client details live

`BBH_MODEL_PARAMS.md` in the benchmark folder is authoritative. `EmoBench/EMO_SCRIPT.md` §1 also
tables the BBH runners' model ids and SDKs — it predates the inventory and covers six of the eight;
where they disagree, the inventory was read later and cites lines.
