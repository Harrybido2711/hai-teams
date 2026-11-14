<!-- size-budget: 12500 -->
<!-- One job: how to read and rebuild Results/results.db. Larger than the default because, since
     2026-11-13, it also carries what to watch in each benchmark's rows and how scores are combined
     across benchmarks — both put here at the user's instruction — for all ten benchmarks. -->
# Results database

`Results/results.db` is a SQLite copy of every recorded score and the six criteria, made at the
user's request on 2026-11-06 because a workbook is awkward for an agent to read. **Query it instead
of parsing the Excel files.** It needs nothing but `sqlite3`:

```bash
sqlite3 -header -column Results/results.db "SELECT * FROM final_scores WHERE benchmark = 'EmoBench'"
```

**It is derived, so never edit it by hand.** The workbooks stay what the user reads and what a result
is recorded into; `.claude/scripts/build_results_db.py` rebuilds the database from them and from the
criteria files. A `Final_result` cell that differs from `Tempo_results` stops the build; so does a
sheet without exactly one highlighted result row, or a `Summary` cell that differs from its sheet.

```bash
python3 .claude/scripts/build_results_db.py          # rebuild, after any change to a source
python3 .claude/scripts/build_results_db.py --check  # exit 1 if a source changed since the build,
                                                     # or the handed-out summary is behind it
```

## Tables

| Table | One row per | Holds |
|---|---|---|
| `benchmarks` | benchmark, all ten | process group, workbook sheet, and `result_row`: the label of the sheet's one highlighted row |
| `runs` | benchmark × workbook column | `model_config`; `selected` = 1 if the column is in `Final_result` |
| `scores` | benchmark × sub-benchmark × column | `value`; `level` is `sub`, `group`, `overall` or `standalone`; `parent` names the group; `position` is the sheet order |
| `criteria` | benchmark × selected model × task × metric | `value` or `value_text`, `reason` when blank, `source` file |
| `criteria_inputs` | benchmark × model with criteria | source file, rows, scorer, price source |
| `provenance` | row of a workbook's `Provenance` sheet | where a column's numbers came from, and its caveats |
| `sources` · `meta` | input file · build fact | md5 of each input, which `--check` compares |

Views: `final_scores` (selected models only, the numbers to report), `final_results` (of those, each
benchmark's result row: one number per benchmark and model) and `gaps` (criteria with no number, and
why).

## Reading it correctly

- **NULL is "no usable number", never a zero.** `Awareness`/`Qwen` perspective is a real 0.0.
- **`scores` holds every model.** Use `final_scores` for anything reported.
- **`criteria` holds the selected models only**, and `task = 'ALL'` is the benchmark's row: tasks
  pooled item-weighted, McNemar on the summed counts. Other `task` values are one task's own numbers,
  on the repeat subsample where the metric comes from repeats.
- **`level` matters when averaging.** A `group` or `overall` row is already an aggregate of `sub` rows.
- **One number per benchmark is `final_results`**: the row highlighted on the benchmark's sheet, at
  the user's instruction of 2026-11-13. `Results/result_rows.py` writes the highlight (and
  `Final_result`'s `Summary` sheet, those rows copied); run it after any recorder. What each row is:
  § Averaging across benchmarks.
- **`standalone` is never averaged.** Two rows: NegotiationToM's `All_EM`
  ([why](benchmarks/interpersonal/negotiationtom.md)) and MultiChallenge's `Overall (micro)`.
- **MultiChallenge and Wonderbread have no criteria numbers.** They are LLM-judged, and their
  `criteria` rows are gaps with a stated reason.
- **`value_text`** holds a non-numeric answer such as `not reported`.

```sql
-- one model across all six criteria on one benchmark
SELECT criterion_no, metric, value FROM criteria
WHERE benchmark = 'NegotiationToM' AND model_column = 'Gemini' AND task = 'ALL' ORDER BY 1, 2;

-- what is still missing
SELECT benchmark, model_column, criterion, reason FROM gaps ORDER BY 1, 2;
```

Criteria definitions: [evaluation-criteria.md](evaluation-criteria.md) and
[criteria-runs.md](criteria-runs.md).

## What to watch in each benchmark's rows

Marked at the user's request, 2026-11-13. One line per trap; the detail is the benchmark's page and
the workbook's `Provenance` sheet, which is the `provenance` table here:
`SELECT column_label, note FROM provenance WHERE workbook = 'Final_result' AND sheet = 'Emo'`.

| Benchmark | Watch for |
|---|---|
| BBH | **Near its ceiling.** `overall` is the plain mean of 20 `sub` rows, and on 15 of them some model scores 1.000, so models differ by little. Qwen's `dyck_languages` 0.696 counts 17 unusable rows as wrong. |
| MMLU | **13 subjects, not all of MMLU; two scorers and two prompts behind one sheet.** Gemini and OpenAI are scored by the lenient matcher, Qwen, Gemma and Deepseek by their runner's `==`; Deepseek was asked for the answer text, the others for a letter. OpenAI leads Deepseek by 0.0002, a gap the scorer difference alone could account for. |
| DocVQA | **Two `overall` rows of different kinds, and no `sub` rows.** `ANLS` is graded and is the reported metric; `Overall accuracy` is a binary column this project added. Never average the two. Deepseek is NULL: it takes no images. |
| EmoBench | **Small: 400 items**, so one item moves a `sub` row by 0.015–0.036. `eu_overall` weighs its four rows by item count (49 / 28 / 56 / 67), `ea_overall` is the plain mean of four rows of 50, and `overall` is the mean of those two groups, not of the eight rows. Gemma's column ran at `temperature=0.6`. |
| NegotiationToM | **Its `overall` row is ours, and `All_EM` is not one.** Upstream defines no total; the highlighted row is the mean of the four `sub` rows. `All_EM` is a fifth metric: a dialogue counts only if every one of its rows is right, so 0.04–0.09 is expected. The four `sub` rows are not four tasks: exact match for desire and for belief, then micro and macro F1 on the same intention answers. Gemma's `Belief_EM` is its thinking-off score, 0.49 against 0.62 with thinking on over 463 paired items. |
| AwareBench | **`overall` is five dimensions, not the `sub` rows.** `mission` is a `group` row over two `sub` rows; `mission_open-ended` is NULL for every model, its LLM-judged rows having been dropped. `perspective (2nd-order)` is 170 rows and a fifth of the score: Qwen's 0.0 is real (it declined every item) and Gemma's 0.129 is 124 declined, so both are mostly refusals. |
| MultiChallenge | **Every cell is a judge's grade**: Haiku 5.5, three replicates, its error against human labels unmeasured, recorded before an audit. Two summary rows: the macro `overall` is the headline, `Overall (micro)` is `standalone`. 272 of 273 items are scored, and an axis holds 41–113 of them. |
| Wonderbread | **QA rows are 1–3 where 1 is best, so lower is better** — the opposite of every other row here; SOP rows are 0–1, higher is better. A judge's grades, as MultiChallenge's. Upstream defines no composite: the `overall` row is ours, on 0–1 with the QA rows converted first (next section). Deepseek is NULL, not 0: it takes no images. Models are not scored on the same units — QA / SOP: Gemini and OpenAI 120 / 486, Qwen 113 / 482, Gemma 115 / 482. |
| Multi-party Goal Tracking | **Our automatic matcher, not upstream's person** (`mpgt_lenient_v4`). Two `group` rows, `dst` and `goal`, each the plain mean of six prompt-style `sub` rows; the `overall` row is ours, their mean. Low by nature — goal 0.06–0.24: models name goals in their own words. An unread slot counts as wrong, so Qwen's `dst_reasoning` 0 is an answer shape the reader cannot place. OpenAI, Gemini and Qwen are lower bounds: a list with the wrong number of answers is credited nothing (placed by position, OpenAI would lead `dst`). |
| PlanBench | **A pilot, not the full run** — its sheet is `PlanBench (pilot)`: every 10th instance, 1,513 scored rows a model, so a `sub` row rests on 10–199 rows. `overall` is pooled, correct ÷ rows, not the mean of the task rows. Qwen is a lower bound: 320 of its scored rows were cut at the output cap, and the pilot has no retry. `task_3_plan_verification_with_llm_plans` cannot pass 0.7 (3 of its 10 golds are null). The two obfuscated slices score 0 for every model and are in no row. |

**A column is a slot, not one configuration.** `runs.model_config` names what ran on each benchmark.
Qwen's and Gemma's route and thinking setting differ between sheets
([sample10-plan.md](sample10-plan.md) § Decisions for the user, D1). Deepseek's row 2 reads
`deepseek-reasoner` on every `Final_result` sheet, an id that now aliases `deepseek-flash` (the legacy
model behind it was retired 2026-07-24); which model produced the older Deepseek rows is not
confirmed.

## Averaging across benchmarks

Benchmarks weigh the same, and a model's score on one is its result row: `SELECT * FROM
final_results`. Recorded 2026-11-13 from the user's instructions; a line that is the planner's choice
says so. A row marked **ours** has no composite upstream. `Results/result_rows.py` computes three of
them from the sheet's own rows and writes them into the sheet; PlanBench's comes from its recorder.

| Benchmark | Its result row |
|---|---|
| BBH, MMLU, EmoBench, AwareBench, MultiChallenge | the `Overall Score` row the sheet was recorded with |
| DocVQA | `Overall accuracy` — the user asked for accuracy; `ANLS`, the other `overall` row, stays beside it |
| PlanBench | **ours**: correct ÷ rows over its task rows, pooled (upstream reports each task on its own) — **the pilot** until the full run is recorded |
| Wonderbread | **ours**: mean of its eight `sub` rows, each QA row first converted to 0–1 as `(3 − x) / 2` (1 → 1.0, 2 → 0.5, 3 → 0), SOP rows as stored. The user had it recorded |
| NegotiationToM | **ours**: mean of its four `sub` rows, `All_EM` left out — the planner's choice, not confirmed |
| Multi-party Goal Tracking | **ours**: mean of its two `group` rows, `dst` and `goal` — the planner's choice, not confirmed |

**The numbers themselves are printed in one place, `Results/RESULTS_SUMMARY.md`** — the summary
written to be handed to other people (the user, 2026-11-14): the ten results, each benchmark's
cautions, the means and both oracles. Its prose is hand-written; `python3 Results/summary_tables.py`
fills its two number tables from this database, and `--check` here fails until it has been rerun.
**It names the workbook and nothing else local** — no path, script, database, cluster or job — so
nothing from this file is pasted into it unread.

- **Two bases, side by side**: every benchmark with a recorded score, and the benchmarks every
  selected model has a score on. A model with a NULL on the first basis has no mean there (Deepseek is
  text-only: DocVQA and Wonderbread). It is never averaged over fewer benchmarks than the others, and
  never given a 0. Confirmed by the user.
- **Oracle 1** (the user's term): per benchmark, the highest score among the selected models; then
  the mean over benchmarks.
- **Oracle 2** (the user's term): per benchmark, the highest value of each `sub` row — for a
  Wonderbread QA row, the lowest raw score — combined into that benchmark's score; then the same
  mean. The user's example was bbh, a plain mean. Where the `overall` is not a plain mean of the `sub`
  rows, the benchmark's own weights are used so both oracles sit on one scale — the planner's
  reading. EmoBench: mean of EU and EA, EU weighted by items (49 / 28 / 56 / 67 in sheet order), EA
  plain. AwareBench: five equal dimensions, mission the mean of its explicit and implicit rows. Those
  weights reproduce every stored `overall`. Multi-party Goal Tracking: each task the plain mean of its
  six style rows, then the mean of the two tasks. PlanBench: a task row weighs its row count (the
  pilot's, in sheet order: 199, 149, 149, 10, 149, 112, then 149 five times).
- **The values** — each model's mean and both oracles, on both bases, with PlanBench's pilot and
  without — are the summary's second generated table. `summary_tables.py` holds the weights above and
  stops when they no longer reproduce a stored result row, as they will not once PlanBench's full
  run replaces the pilot's row counts.
