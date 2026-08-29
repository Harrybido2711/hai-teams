# BIG-Bench Hard — benchmark card

<!-- size-budget: 7000 -->
<!-- One job, not two: the card for one benchmark. It runs long because bbh carries eight
     providers, a broken run that must not be read as a result, and the incident that produced the
     project's one-scorer rule. The split that would shrink it already happened into
     bbh-parameters.md and the benchmark's own README. Raised again when the 20-task scope
     became a recorded decision rather than a gap someone might try to close. -->

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

`bbh_eval_core.py::score_response` is the only scorer here and every runner imports it. **This is
the benchmark where the project's one-scorer rule came from** (`CLAUDE.md`; `script-skeleton.md`
rule 7): until 2026-08-29 five runners compared with `==` and three used the lenient matcher, and
rescoring the identical stored responses moved deepseek 0.448 → 0.961, kimi 0.243 → 0.884,
gpt-4o-mini 0.308 → 0.834, xai 0.509 → 0.940, llama 0.726 → 0.910. Gemma and Qwen did not move.

Every `_overall.csv` carries a `scorer` column — `SCORER_VERSION`, today **`lenient_v2`** — and
every row a `config` column (rule 8), so a number cannot be read without knowing what produced it.

**v3, 2026-08-29:** four branches for a concise answer with something wrapped around it — LaTeX
math, a unit noun after a number, a restated Yes/No, an option letter at the end. **+766 rows, 0
losses, across five models.** They fire only when the model emitted the marker, so a stray letter in
un-marked reasoning is never read as a choice. It found a real failure: Luna scored **0.000 on two
whole tasks** — `object_counting` and `web_of_lies` — with `no_marker=0` and `empty=0`, because it
answers `8 musical instruments` and `No, Ka does not tell the truth.`. **A task at exactly 0.000
with nothing empty is a scorer bug, not a result.** Luna's macro went 0.8102 → 0.9213.

Still unfixed and deliberately so: 48 Luna rows answer `Elanor does not tell the truth.` for gold
`No`. Crediting those needs negation parsing specific to `web_of_lies` — measured at +92 rows and 0
losses, but it would make the matcher task-aware rather than packaging-tolerant. Not adopted; the
user's call.

**v2, 2026-08-29:** the matcher also strips markdown emphasis. A Luna pilot answered
`**bootlegging, indifferent, trainman**` to gold `bootlegging indifferent trainman` and scored 0.
Across all 36,348 stored rows the fix gains 31 and loses 0. It moved two reported cells, both now
updated in `Final_Result.xlsx`: XAI `dyck_languages` 0.752 → 0.756, Qwen `formal_fallacies`
0.952 → 0.968.

**The prompt had the same problem as the scorer.** Six of the eight old runners sent a four-space
indented prompt; gemma and qwen sent the same text indented eight, through the `_finish` twins their
jobs submitted. The core now sends the six-runner original byte-for-byte — including the
indentation, which must not be tidied, or a new run differs from the rows on disk by the prompt as
well as the model.

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
