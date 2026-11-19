# BIG-Bench Hard — benchmark card

<!-- size-budget: 13500 -->
<!-- One job, not two: the card for one benchmark. It runs long because bbh carries eight
     providers, a broken run that must not be read as a result, and the incident that produced the
     project's one-scorer rule. The split that would shrink it already happened into
     bbh-parameters.md and the benchmark's own README. Raised again when the 20-task scope
     became a recorded decision rather than a gap someone might try to close, and again for
     scorer v3 — whose entry has to carry the 0.000 lesson, or it is just a version number.
     Raised to 13500 on 2026-11-15 for the 10% sample rerun: a second output directory in five
     folders, its serial-only rule, how a failed row and a thinking row are handled on resubmit,
     what parse_ok and the call count mean, and the two legacy bugs its test found are operating
     facts this card must hold. The field definitions stay in the core, not here. -->

General task ability. Exact match against a `"Final Answer: <answer>"` prompt contract. No LLM judge.

## Paths

| | Path |
|---|---|
| Local | `Tasks_benchmarks/bbh` |
| Quest | **`/projects/p32983/Tasks_benchmarks/bbh`** — this account's own copy, synced 2026-08-30, 43 files verified byte-identical by `md5sum`. Code and data only: results live locally and in git, because results flow *down* |
| Quest — not ours | `/projects/p32983/bbh` is **`cpz1698`'s**, flat, last touched 2026-06-09. Do not write into it |

## Layout — rebuilt 2026-08-29 on the EmoBench convention

`BBH_<Slot>/` per model, holding `<vendor>_bbh_eval.py`, `run_bbh.sh`, `log.txt`/`log.err` and
`results/<task>/<model-slug>.{jsonl,csv}` + `_overall.csv`, plus a `<model-slug>_bbh_overall.csv`
roll-up. The 20 task JSONs live in `data/` and are shared. **Result files are named after the model,
not the folder**, and `--model` sets both what is called and what is written, so a copied folder
cannot relabel another model's numbers.

**All ten slots have results**, and `BBH_Gemini_Flash2.5` / `BBH_GPT_4o_mini` are named for the
superseded models they call so their numbers are not mistaken for the current slots'. The two added
2026-08-29 — `BBH_Gemini_Flash3.5lite_OpenRouter` and `BBH_GPT_5.6_Luna` — are the only runners here
that comply with the model-parameter rule; the other eight set no reasoning or output cap,
deliberately, so their rows stay comparable. Their caps are still **chosen rather than measured**,
but both ran 4,833 rows with `no_marker=0`, so nothing was truncated at them.

**20 tasks, 4,833 items, and that is the settled scope** — seven of upstream's 27 are not vendored
here, and the user decided on 2026-08-29 to keep it that way: what was run before is what this
benchmark reports on. Do not add them to "complete" it; that would leave old models at 20 tasks and
new ones at 27. Verified 2026-08-29: every example carries a non-empty `input` and `target`, and all
36,638 result rows cross-check against the data files with zero mismatches.

Full tree, the six matcher branches and the task inventory: **`Tasks_benchmarks/bbh/README.md`** —
the benchmark's own committed notes, and the place to read before touching it.

## Results — the six are complete, verified 2026-09-07

Every slot is at the full 20 tasks / 4,833 rows except Kimi. Macro is the mean of the 20 per-task
`average_score` values read from **`results/<task>/<model>_overall.csv`** — not from the slot-level
`results/*_bbh_overall.csv`, which is stale for two slots (see the traps).

| Slot | Folder | Macro | no_marker | empty | Among the six |
|---|---|---|---|---|---|
| Gemma | `BBH_Gemma` | **0.9684** | 0 | 0 | yes |
| Deepseek | `BBH_Deepseek` | **0.9616** | 0 | 0 | yes |
| XAI | `BBH_XAI` | **0.9461** | 0 | 0 | yes |
| Gemini | `BBH_Gemini_Flash3.5lite_OpenRouter` | **0.9375** | 0 | 0 | yes |
| OpenAI | `BBH_GPT_5.6_Luna` | **0.9349** | 0 | 0 | yes |
| Qwen | `BBH_Qwen` | **0.9339** | 16 | 1 | yes |
| — | `BBH_Llama` | 0.9109 | 37 | 0 | no |
| — | `BBH_GPT_4o_mini` | 0.8451 | 4 | 0 | no — superseded |
| — | `BBH_Kimi` | 0.9310 *(10 tasks)* | 1 | 1 | no |
| — | `BBH_Gemini_Flash2.5` | 0.3455 | 3,002 | 0 | no — superseded, broken run |

Qwen's 17 unusable rows are all in `dyck_languages` and are scored wrong, not dropped; that task
reads 0.696. It is what the DeepInfra repair of 2026-08-30 left of 256.

**Per-task cells live in the workbooks, not here** — `Final_result.xlsx` § Big Bench Hard for the
selected models, `Tempo_results.xlsx` for all ten, each with the sources on its `Provenance` sheet. Both were
refreshed from these files on 2026-09-07: 76 cells changed in `Tempo_results.xlsx`, 0 in
`Final_result.xlsx`, which already matched.

**bbh ran locally, not on Quest** — `sacct` has no bbh job. Quest holds the code and data only.

## Scoring — one matcher, imported, for every model

`bbh_eval_core.py::score_response`, imported by every runner so no model can be judged more or less
generously than another. It is at **`lenient_v5`**, and each version came from a correct answer
being scored 0 for how it was written. **Full history, what each branch does, and the one line
deliberately not crossed: [bbh-scoring.md](bbh-scoring.md).**

The rule of thumb worth carrying out of it: **a task at exactly 0.000 with `no_marker=0` and
`empty=0` is a scorer bug, not a result.**

## The 10% sample rerun — written 2026-11-15, not submitted

Criteria 1, 2, 3 and 6 for the five selected models on 484 of the 4,833 items (`sample10.json` at
the bbh root). Method, scaling and the user's decisions: [sample10-plan.md](../../sample10-plan.md).
**It runs today's settled config, not the one behind the numbers above** (the user's D1 = B) —
per model in [bbh-parameters.md](bbh-parameters.md) § The 10% sample rerun.

| Model | Folder | Runner |
|---|---|---|
| GPT | `BBH_GPT_5.6_Luna` | `gpt56luna_bbh_eval.py --sample sample10` |
| Gemini | `BBH_Gemini_Flash3.5lite_OpenRouter` | `gemini35lite_bbh_eval.py --sample sample10` |
| Qwen | `BBH_Qwen_DeepInfra` (new; `BBH_Qwen` untouched) | `qwen_di_bbh_sample.py` |
| Gemma | `BBH_Gemma_DeepInfra` (new; `BBH_Gemma` untouched) | `gemma_di_bbh_sample.py` |
| Deepseek | `BBH_Deepseek`, beside the legacy runner | `deepseek_bbh_sample.py` |

- **A sample run reads and writes only `results_sample10/`** (`smoke_sample10/` under `--limit`),
  `results/`'s layout; `idx` indexes the whole task. Each row has the per-row fields of
  [evaluation-criteria.md](../../evaluation-criteria.md) plus `billed_*`, `billed_cost`,
  `billed_estimated_cost`, `attempts_without_usage`, `reasoning_chars`, `think_tag_calls`,
  `halted`, `last_call_cost`, `last_call_estimated_cost`, `prior_n_attempts`, `prior_failed_runs`,
  `resumed_after_failure` —
  defined in the comment above `BILLED_KEYS` in `bbh_eval_core.py`, names shared with MMLU and
  DocVQA. `billed_*` sum every call of every run of the row, booked as each response arrives (a
  reply with usage but no `choices` is billed, and counted once); the plain token fields are the
  answering attempt's. **True HTTP calls on an answered row = `n_attempts` + `cut_retry` +
  `forced_answer`** (`attempts_without_usage` counts calls, not attempts).
- **`parse_ok` = the scorer extracted a prediction**: the text after `Final Answer:`, or with no
  marker the whole response, which the matcher can still credit (a bare `(B)` scores 1). So it is
  never False on a row scored 1. "Did the model finish" is `no_marker`, as before.
- **A row that failed, or is left unfinished, is written, scored 0, and held.** Unfinished = a
  CutRetry row (Qwen, Deepseek) still cut at the cap with no forced answer. A resubmit calls exactly
  those again; until a new row replaces one it stays in every save, even if the resubmit is killed
  first, and the replacement folds in the history. **A row in flight when a halt or SIGTERM ends
  the run is written too**, with every call it had paid for: answered if its call had returned,
  else failed, with `halted` naming the stop ("SystemExit: 143").
  Exit 1 while any remain. The roll-up
  `<slug>_bbh_overall.csv` is rebuilt from every task file: `complete 1` only at 484 rows, none
  failed or unfinished, one config.
- **Qwen and Gemma on DeepInfra must answer without thinking.** The first row whose calls return
  `reasoning_content`, `<think>` tags in the content, or reasoning tokens halts the run: `sample10_THINKING_HALT.txt`, exit 3. A
  resubmit calls no row while that marker exists, or while a stored row shows thinking (the marker
  is then rewritten). Other markers: `sample10_*_HALT.txt`, and `smoke_sample10_*` for `--limit`.
- **Serial, on the main thread; `--workers` is refused for `--sample`.** The watchdog is SIGALRM,
  which works nowhere else; the client's timeout is per read, and a trickling server defeats it.
- **`test_bbh_sample10.py`**, Parts A to D. A: without `--sample`, the five legacy runners are
  identical to commit `1cc5bbd5` (files, requests, output). B: the sample path. C: the watchdog
  against a blocked socket. D: the real sample runners through the SDK and a mock server. **A is
  local-only** (it needs git; on Quest it prints SKIPPED). Only "ALL PASSED" is the full proof. Run
  it with an arm64 Python that has `openai`; the laptop's default x86 `python3` cannot run `git`.
- **Run order.** Smoke `--sample sample10 --limit 3` (60 rows a model, 2026-11-15); the user
  verifies; one transfer — core, five runners, five `run_bbh_sample10.sh`, `sample10.json`, the test
  — `md5sum` both sides; `sbatch run_bbh_sample10.sh` per model, from anywhere (absolute paths, logs
  appended to `log_sample10.*` in the folder). Exit 1 = failed or unfinished rows (resubmit), 2 = a
  forced answer to read, 3 = thinking came back. **Not on Quest yet**; whether Quest's `.env` has
  `DEEPINFRA_API_KEY` and `OPENROUTER_API_KEY`, and `/projects` resolves as written, is unverified.

## Its own traps

- **`BBH_Gemini_Flash2.5` is a broken run, not a low score.** 62% of its 4,833 responses (3,002)
  stop mid-reasoning and never emit `Final Answer:`, so no scorer can rescue them and 0.3455 is what
  it is. It is **not** re-run: the Gemini slot is now `gemini-3.5-flash-lite`, which is complete.
  The row keeps its own column in `Tempo_results.xlsx` and appears nowhere in `Final_result.xlsx`.
- **The slot-level `results/*_bbh_overall.csv` is stale for Gemma and Qwen.** The DeepInfra repair
  of 2026-08-30 re-ran only the affected tasks and overwrote each roll-up with just those —
  `MACRO_AVG_over_5_tasks` = 0.879 for Gemma, `over_13_tasks` = 0.9146 for Qwen, against the true
  0.9684 and 0.9339. **Aggregate the per-task files instead**; that is what both workbooks now do.
- **`Final_result.xlsx` reproduces the per-task files exactly** — all 126 six-slot cells, 0 blanks,
  checked 2026-09-07. `Tempo_results.xlsx` did not, and was refreshed from them.
- **`kimi-k2.5` and `Llama-4-Maverick` are not among the six.** bbh is the only benchmark either ran
  on; Kimi has 10 of 20 tasks. `PLAN.md` and `Final_result.xlsx` are the coverage claim, never
  `ls`.
- **`_superseded/` in `BBH_Gemma` and `BBH_Kimi`** holds older duplicates in the pre-restructure
  format, parked rather than deleted. Never read a number out of one.
- **The legacy serial loop ignores `--shard`.** `run_tasks` slices by shard only when `--workers` is
  above 1; serially every shard calls every item into its own tagged file — 628 requests for one
  "shard" of two 250-item tasks in the test. Unsharded runs, all bbh has made, are unaffected. Left
  as it was: fixing it changes the no-`--sample` path. The sample loop shards in both modes.
- **Legacy Gemini under `--workers`: a failed row's `backend=` is another row's.** It is read from a
  thread-local holding whatever that worker thread answered last, so a row whose every attempt
  raised carries a neighbour's backend — 8 runs of the baseline gave one such row two values. The
  sample path reads the backend from each response instead.
- **No `seed` anywhere, Kimi at `temperature=1`, and no reasoning or output cap on the eight older
  runners.** Nothing here is reproducible. Those caps are **deliberately still open**: setting one
  changes what the models emit and would make new rows incomparable with the 4,833 on disk. Cap them
  when bbh is re-run. Per-model parameters, with `file:line`: [bbh-parameters.md](bbh-parameters.md).

## Fixed on 2026-08-29 — do not re-report these

Four traps this page used to carry are gone with the rewrite, and are recorded only so a stale memory
of them is not acted on: the two `.sh` files that submitted a `_finish` twin rather than the runner
their name implied; the narrowed `splits` re-run lists (`--task all` is now the default and the task
list is a single constant in the core); the `None` response that raised inside the per-split handler
and discarded a whole 250-item task; and the per-runner copies of the scorer.
