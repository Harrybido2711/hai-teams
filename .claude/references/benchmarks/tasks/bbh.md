<!-- size-budget: 7500 -->
<!-- The card carries one job, not two: the scorer split and the workbook-provenance rule are
     both answers to "can I trust this number", and splitting them apart is how a reader ends
     up refreshing a column from a degraded _overall_results.csv. The six matcher branches and
     the folder layout already live in Tasks_benchmarks/bbh/README.md. -->
# BIG-Bench Hard — benchmark card

<!-- size-budget: 6000 -->
<!-- One job, not two: this is the card for one benchmark. It runs long because bbh carries eight
     providers, two different scorers and five recorded traps; the split that would shrink it
     (parameters) already happened into bbh-parameters.md. -->

General task ability. Exact match against a `"Final Answer: <answer>"` prompt contract. No LLM judge.

## Paths

| | Path |
|---|---|
| Local | `Tasks_benchmarks/bbh` |
| Quest | not verified — nothing points to a remote copy, but Quest was not checked (out of scope 2026-08-22). Confirm before assuming a remote path |

## Layout

**One folder per model since 2026-08-29.** `<bbh>/<model>/` — `deepseek gemini gemma kimi llama
openai qwen xai` — holds that model's `<model>_eval.py`, its `<model>_eval_script.sh`, its
`<model>_<task>.csv` results, its `<model>_overall_results.csv` and its own `_outlog`/`_errlog`.
The **20 task JSONs stay at the benchmark root** and are shared; so is `.env`. A vendored `bin/` +
`dotenv/` pip install is also there.

**Submit from the bbh root** (`<model>/<model>_eval_script.sh`): the job script's `--output` and
`--error` are relative to the submit directory. The runners themselves are cwd-independent — they
resolve every read and write from `__file__` via `MODEL_DIR` / `BBH_ROOT`.

`<model>/_superseded/` exists in `gemma/` and `kimi/` and holds an older duplicate that was parked
rather than deleted. Never read a number out of it.

## Provider coverage — the files on disk are not the answer

Result CSVs exist for **eight** prefixes: `deepseek gemini gemma kimi llama openai qwen xai`.
`PLAN.md` says seven. Both are correct: **this benchmark was run against more models than are
reported, and some of those runs were abandoned** (user, 2026-08-22). A leftover CSV is not a
retracted result and was not deleted, so it is still on disk.

Take the coverage claim from `PLAN.md` and the workbooks, never from `ls *.csv`. bbh is the one
benchmark here where counting files overstates what was run on purpose. Which workbook depends on the
question: `Results.xlsx` for what was run at all, `Final_Result.xlsx` for the six selected models that
are actually reported (`PLAN.md`).

## Scoring — two different scorers, not one

Every runner extracts the text after `Final Answer:`, then **five score with strict case-folded
equality** (`deepseek`, `gemini`, `kimi`, `llama`, `openai`) **and three with a six-branch lenient
matcher** (`xai`, `gemma`, `qwen`). *(This page said four and four until 2026-08-29; recounted from
the code it is five and three — by file 5 and 5, since gemma and qwen each have a `_finish` twin.)*

The five strict-scored models are penalised for formatting rather than reasoning, and nothing in a
result CSV records which scorer produced it. **Both workbooks now carry that split as a row on their
`Provenance` sheet**, because the bbh sheet is the one place where six columns are not scored the
same way. **What the six branches do, what the marker-missing
fallback costs, and which of the 20 tasks each branch rescues: `Tasks_benchmarks/bbh/README.md`.
Read it before comparing any two providers here.**

**`<model>/<model>_overall_results.csv` is not where the workbook's bbh numbers come from** —
established 2026-08-29 while rebuilding the workbooks. For five of the eight those files record a
later, degraded re-run: `gemini` has `date_understanding` at 0.064 where the workbook has 0.936, and
`xai` has `reasoning_about_colored_objects` at 0.12 where the workbook has 0.988. The shape gives the
cause away — near-zero on every lettered-choice task, unharmed on the short free-form ones — which is
the marker-missing fallback scoring whole responses after the model stopped emitting `Final Answer:`.
Several are also short, for the narrowed-`splits` reason below. **Do not refresh a bbh column from one
of these files without checking it against the per-task CSVs first.**

`gemma` is the exception, and checking is what established it: 20 tasks, 4,833 rows, an empty
`errlog`, and every per-task CSV mean equal to the overall CSV. That is why `Gemma` has a bbh column
from 2026-08-29 and the other seven columns do not come from these files.

Every generation parameter each runner sets, with `file:line`:
[bbh-parameters.md](bbh-parameters.md) — also where the two-file caveat above is spelled out per model.

## Its own traps

- **Two of the eight `.sh` files do not submit the file their name implies.** `gemma_eval_script.sh`
  runs `gemma_finish.py` and `qwen_eval_script.sh` runs `qwen_finish.py`, and the `_finish` twins
  differ from the `_eval` ones in `max_tokens`, client timeout, retry behaviour and task list. **Read
  the `.sh` before reading any runner here** — documenting these two from the `_eval.py` file gets
  four columns wrong, which is exactly how the first version of the inventory came out wrong.
- **The `splits` list in several submitted scripts is a narrowed re-run list** — `deepseek` runs 5
  tasks of 20, `gemma_eval.py` 4, `kimi` 1, `openai` 14, `qwen_finish.py` 3 — never restored. A CSV present is not evidence the script beside it would regenerate it, and no
  result file records which script version wrote it.
- **One failed call discards a whole task, in eight of the ten runner files.** A `None` response
  reaches `re.search` and raises `TypeError`; the per-split handler catches it and moves to the next
  split, dropping the rows collected so far without writing a CSV. `gemma_finish.py` and
  `qwen_finish.py` return `""` and are the exceptions.
- **`gemini_eval.py` still calls `gemini-2.5-flash`,** which was superseded on 2026-08-23 by
  `gemini-3.5-flash-lite` on OpenRouter ([../../model-parameters.md](../../model-parameters.md)).
  The runner has not been changed, so this table still describes what it does — updating the page
  without updating the code would make it a lie. Change both, or neither.
- **No script sets `seed`, and one runs at `temperature=1`.** Nothing here is reproducible — and
  that is now a gap rather than a limitation: `seed` is measured working on both Gemini flash-lite
  routes, so probe each provider here rather than assuming it is unavailable
  ([../../model-parameters.md](../../model-parameters.md) rule 6).

## Where the client details live

`Tasks_benchmarks/bbh/README.md` is the benchmark's own committed notes: the 20 tasks and their
sizes, the answer shapes, the six matcher branches, and the layout. *(An earlier version of this
page pointed at `BBH_MODEL_PARAMS.md` in the benchmark folder. No such file exists — checked
2026-08-29.)* `EmoBench/EMO_SCRIPT.md` §1 also
tables the BBH runners' model ids and SDKs — it predates the inventory and covers six of the eight;
where they disagree, the inventory was read later and cites lines.
