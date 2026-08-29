# BIG-Bench Hard — benchmark card

<!-- size-budget: 8000 -->
<!-- One job, not two: the card for one benchmark. It runs long because bbh carries eight
     providers, a broken run that must not be read as a result, and the incident that produced the
     project's one-scorer rule. The split that would shrink it already happened into
     bbh-parameters.md and the benchmark's own README. Raised again when the 20-task scope
     became a recorded decision rather than a gap someone might try to close, and again for
     scorer v3 — whose entry has to carry the 0.000 lesson, or it is just a version number. -->

General task ability. Exact match against a `"Final Answer: <answer>"` prompt contract. No LLM judge.

## Paths

| | Path |
|---|---|
| Local | `Tasks_benchmarks/bbh` |
| Quest | `/projects/p32983/bbh`, and it belongs to **`cpz1698`, not this account**. The 2026-08-29 restructure is local-only and has not been pushed there; do not transfer into it without asking |

## Layout — rebuilt 2026-08-29 on the EmoBench convention

`BBH_<Slot>/` per model, holding `<vendor>_bbh_eval.py`, `run_bbh.sh`, `log.txt`/`log.err` and
`results/<task>/<model-slug>.{jsonl,csv}` + `_overall.csv`, plus a `<model-slug>_bbh_overall.csv`
roll-up. The 20 task JSONs live in `data/` and are shared. **Result files are named after the model,
not the folder**, and `--model` sets both what is called and what is written, so a copied folder
cannot relabel another model's numbers.

Slots **with results**: `BBH_Deepseek` · `BBH_Gemma` · `BBH_Qwen` · `BBH_XAI` · `BBH_Kimi` ·
`BBH_Llama` · `BBH_Gemini_Flash2.5` · `BBH_GPT_4o_mini`. The last two are named for the superseded
models they call, so their numbers are not mistaken for the current slots'.

Slots **with a runner and no results**, added 2026-08-29:
`BBH_Gemini_Flash3.5lite_OpenRouter` (`google/gemini-3.5-flash-lite`, the settled route) and
`BBH_GPT_5.6_Luna` (`gpt-5.6-luna`, `reasoning_effort="low"`). These are two of the six. **Neither
has been piloted and their output caps are chosen rather than measured** — `--limit 20` and check
`no_marker` first. They are also the only two runners here that comply with the model-parameter
rule; the other eight set no reasoning or output cap, deliberately, so their rows stay comparable.

**20 tasks, 4,833 items, and that is the settled scope** — seven of upstream's 27 are not vendored
here, and the user decided on 2026-08-29 to keep it that way: what was run before is what this
benchmark reports on. Do not add them to "complete" it; that would leave old models at 20 tasks and
new ones at 27. Verified 2026-08-29: every example carries a non-empty `input` and `target`, and all
36,638 result rows cross-check against the data files with zero mismatches.

Full tree, the six matcher branches and the task inventory: **`Tasks_benchmarks/bbh/README.md`** —
the benchmark's own committed notes, and the place to read before touching it.

## Scoring — one matcher, imported, for every model

`bbh_eval_core.py::score_response`, imported by every runner so no model can be judged more or less
generously than another. It is at **`lenient_v5`**, and each version came from a correct answer
being scored 0 for how it was written. **Full history, what each branch does, and the one line
deliberately not crossed: [bbh-scoring.md](bbh-scoring.md).**

The rule of thumb worth carrying out of it: **a task at exactly 0.000 with `no_marker=0` and
`empty=0` is a scorer bug, not a result.**

## Its own traps

- **Gemini's results are a broken run, not a low score.** 62% of its 4,833 responses (3,002) stop
  mid-reasoning and never emit `Final Answer:`. The rescore gives ~0.34 against the ~0.93 the
  workbook reports, and the gap does **not** close leniently — the data is truncated, so the
  workbook's Gemini column came from a run that is not in this folder. **Re-run, do not rescore.**
  No output cap is set in the runner, so raising one is not the fix.
- **The other five reported models reproduce `Final_Result.xlsx` exactly** — 102 of 120 cells, all
  18 misses Gemini. The workbook already held the lenient numbers; the code and the stored `score`
  columns were what had not caught up.
- **`kimi-k2.5` and `Llama-4-Maverick` are not among the six.** bbh is the only benchmark either ran
  on; Kimi has 10 of 20 tasks. `PLAN.md` and `Final_Result.xlsx` are the coverage claim, never
  `ls`.
- **`_superseded/` in `BBH_Gemma` and `BBH_Kimi`** holds older duplicates in the pre-restructure
  format, parked rather than deleted. Never read a number out of one.
- **No `seed` anywhere, Kimi at `temperature=1`, and no reasoning or output cap on any runner.**
  Nothing here is reproducible. The caps are **deliberately still open**: setting one changes what
  the models emit and would make new rows incomparable with the 4,833 on disk. Cap them when bbh is
  re-run. Per-model parameters, with `file:line`: [bbh-parameters.md](bbh-parameters.md).

## Fixed on 2026-08-29 — do not re-report these

Four traps this page used to carry are gone with the rewrite, and are recorded only so a stale memory
of them is not acted on: the two `.sh` files that submitted a `_finish` twin rather than the runner
their name implied; the narrowed `splits` re-run lists (`--task all` is now the default and the task
list is a single constant in the core); the `None` response that raised inside the per-split handler
and discarded a whole 250-item task; and the per-runner copies of the scorer.
