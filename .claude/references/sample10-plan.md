<!-- size-budget: 15500 -->
<!-- One job: the 10% sample rerun, split into work packages separate agents can take. Larger than
     the default because each package must stand alone for an agent with no other context. Delete
     this file once the run is recorded and its lasting facts have moved to their references. -->
# 10% sample rerun: plan and work packages

Requested by the user, 2026-11-06. **Status: plan only. Nothing is written, nothing is run.**

**Goal.** Get criteria 1, 2, 3 and 6 ([evaluation-criteria.md](evaluation-criteria.md)) for the
model × benchmark cells that have none, without rerunning whole benchmarks. Run a random 10% of each
sub-benchmark, then scale up to estimate the full benchmark's cost.

**Picking up a package?** Read § Rules for every package and your package. Nothing else is required.

## What runs

| Benchmark | Items | Sample | Strata | Models to run | Calls |
|---|---|---|---|---|---|
| BBH | 4,833 | 484 | 20 tasks | GPT, Gemini, Qwen, Gemma, Deepseek | 2,420 |
| MMLU | 3,943 | 401 | 13 subjects | GPT, Gemini, Qwen, Gemma, Deepseek | 2,005 |
| DocVQA | 5,349 | 539 | 9 question types | GPT, Gemini, Qwen, Gemma | 2,156 |
| EmoBench | 400 | 41 | 8 categories | Gemini, Qwen, Gemma, Deepseek | 164 |
| NegotiationToM | [its page](benchmarks/interpersonal/negotiationtom.md) | 238 of 2,380 dialogues, about 1,414 rows | 3 tasks | Qwen, Gemma, Deepseek | about 4,240 |

That is 21 runs and about 11,000 calls. **Dollar cost is not known yet**; WP7 projects it from the
smoke rows before anyone submits.

**What does not rerun, and why:**

- **AwareBench:** criteria 1, 2, 3 and 6 are already measured on the full run for all five models.
- **EmoBench GPT; NegotiationToM GPT and Gemini:** already measured on the full run.
- **MultiChallenge, Wonderbread:** their runners write the per-row fields, so WP7 computes these from
  the full-run rows. Keep generation cost and judge cost as separate numbers.
- **PlanBench, Multi-party Goal Tracking:** no runner and no results.
- **DocVQA Deepseek:** text-only model, cannot take an image.

Criteria 4 and 5 are out of scope. They need repeats and perturbations, which one sampled pass cannot
give ([criteria-runs.md](criteria-runs.md)).

## How the estimate works

- **Sample:** in each stratum take `ceil(0.1 × N)` items at random, seed `20261106`. Every model gets
  the same items, so comparisons are paired.
- **Scale by stratum, not by a flat ×10.** Rounding up makes the true factor differ: 3 of 28 EmoBench
  `emotional_cues` items is ×9.33. Estimated total cost = Σ over strata (sample cost × N / n).
- **Only totals scale.** Whole-run cost, tokens and wall-clock scale up. These are already rates and
  must **not** be multiplied: $ per correct, mean and p95 tokens, latency percentiles, parse and error
  rates, accuracy.
- **Everything is labelled an estimate**, with a bootstrap interval resampled within strata.
  Sub-benchmark numbers are weak where n is small: EmoBench has strata of 3–5 items and DocVQA a
  stratum of 3. Report the benchmark-level figure and do not rank models on a sub-benchmark.
- **Drift check.** For each cell, compare sample accuracy with the stored full-run rows on the *same
  items*. A gap means the model or config moved since the reported run, so the cost estimate may not
  describe that number. Report it, never overwrite the reported score.

## Decisions for the user (WP0)

Nothing past WP1 starts until D1 is answered: it changes the runners and could change the cost
substantially.

**D1. Which config do the reruns use?** Several reported numbers came from configs other than today's
settled ones:

| Cell | What produced the reported number (read from the runner) |
|---|---|
| MMLU Qwen, DocVQA Qwen | Together, `max_tokens=12000`, no reasoning switch, so thinking was on |
| MMLU Gemma, DocVQA Gemma | Together, `max_tokens` 12500 / 12000, no reasoning switch, so thinking was on |
| EmoBench Qwen | Together, `max_tokens=8192`, no reasoning switch |
| EmoBench Gemma | Together, **`temperature=0.6`**, no reasoning switch |
| MMLU Deepseek, EmoBench Deepseek | no output cap, no reasoning knob exists |
| BBH Qwen, BBH Gemma | runner has both Together and DeepInfra configs; which rows came from which must be checked |
| NegotiationToM Qwen, Gemma | Together, reasoning off |
| MMLU, DocVQA Gemini | native Google route; settled route is OpenRouter |

- **A: the reported config (recommended).** The estimate then describes the number in
  `Final_result.xlsx`, which matches the user's design of 2026-10-12 in [criteria-runs.md](criteria-runs.md).
  Cost: it breaks the "reasoning is always capped" rule for these runs, and thinking-on runs may be
  slow and expensive.
- **B: today's settled config.** Cheaper and more uniform, but it estimates a run that was never
  reported, so accuracy may not match the workbook.

**D2. Qwen's route.** The user's rule of 2026-11-04 sends new Qwen runs to DeepInfra, but the reported
numbers were mostly Together. Recommended: DeepInfra, labelled, and also priced at Together's rate from
the same tokens. Latency cannot be converted, so it describes DeepInfra only.

**D3. MultiChallenge and Wonderbread.** Recommended: compute from full-run rows with no sampling, once
each model's judging has finished.

**D4. Where estimates are recorded.** Recommended: `Criteria_results.xlsx` and `Results/results.db`,
flagged as a 10% estimate, never in `Tempo_results.xlsx` or `Final_result.xlsx`.

## Packages and order

```
WP0 decisions → WP1 samples → WP2 BBH ┐
                              WP3 MMLU │
                              WP4 DocVQA ├→ WP8 review → USER VERIFIES → WP9 Quest → WP10 audit
                              WP5 Emo  │                                              → USER CONFIRMS → WP11 record
                              WP6 NEG  ┘
                              WP7 report (parallel; needs smoke rows for its projection)
```

WP2–WP7 can run at the same time because they own different files.

| WP | Who | Owns | Summary |
|---|---|---|---|
| 0 | user + planner | this file | answer D1–D4 |
| 1 | any agent | sample files, builder script | draw the samples |
| 2 | any agent | `bbh_eval_core.py`, `BBH_*` | sampling and fields through the core |
| 3 | any agent | `mmlu_eval_core.py`, `MMLU_*` | same, plus 3 new runners |
| 4 | any agent | `docvqa_eval_core.py`, the four DocVQA runner folders | same, plus 2 new runners |
| 5 | any agent | `EMO_*` | 4 new runners |
| 6 | any agent | `neg_eval_core.py`, `NEG_{Qwen,Gemma,Deepseek}` | tag, opt-in, retries |
| 7 | any agent | `sample10_report.py`, `criteria_report.py` | estimates and cost projection |
| 8 | a **different** agent from the writer | nothing, reports only | adversarial review |
| 9 | **one** agent only | Quest | transfer, prove, submit |
| 10 | any agent | nothing, reports only | monitor, pull, audit |
| 11 | one agent, after the user confirms | pages, workbooks, db | record |

## Rules for every package

- **The planner session decides.** Stop and report rather than deciding a config, a threshold or a
  scope change. End with a `STATUS:` line from [handoffs.md](handoffs.md).
- **No Quest access before WP9.** Local real calls are for `--limit 3` smoke tests only.
- **Existing results stay valid.** With no `--sample`, a changed core must behave exactly as before.
  Output goes to `<model folder>/results_sample10/`, never `results/`: a shared checkpoint would make
  resume skip or mix rows.
- **Every row writes** the fields in [evaluation-criteria.md](evaluation-criteria.md) § Per-row fields.
  Build clients with `max_retries=0` and retry in our own loop, or `n_attempts` and `latency_s` are
  wrong ([script-skeleton.md](script-skeleton.md) §4).
- **Scoring is the benchmark's one lenient matcher**, imported, never copied into a runner.
- **Row keys come from the sample file**, so `idx` keeps indexing the whole task.
- **Git:** pull before work, stage explicit paths only, commit per finished change, push to `origin`
  and `backup`, never `upstream`. Run `python3 .claude/scripts/check_docs.py` first.
- **Done means evidence:** paste the smoke rows or the test output. Non-empty response, parsed,
  scored, usage fields filled.

## WP1 · Draw the samples

Write `.claude/scripts/build_sample10.py` to produce `sample10.json` beside each benchmark's data:
`{seed, fraction, created, strata: {name: {N, n, ids}}}`.

| Benchmark | Stratum | Id |
|---|---|---|
| BBH | file in `data/` | index into the task |
| MMLU | file in `data/` | index into the subject |
| DocVQA | `question_types[0]`; the workbook has no sub-benchmarks, so this is our choice | `questionId` |
| EmoBench | English only: EA `category`, EU `coarse_category` | `qid` |
| NegotiationToM | none needed; sample whole dialogues | `dialogue_id` |

**NegotiationToM:** use the format of `criteria_subsample.json` (`dialogue_ids`), taking the first 238
of the same `random.Random(42)` order. That contains the existing 150 and the 238-dialogue pilot set.
Record rows per task, because the scaling needs them.

**Done:** counts match § What runs, rerunning the script gives byte-identical files, every id exists in
the data. Commit before WP2–WP6 begin, as they all read these files.

## WP2 · BBH

All five runners import `bbh_eval_core.py`, so most of the work is in the core. Two things stop the
fields being captured today: `call(prompt)` returns only a string, and `core.retry` hides attempts.

- Add `--sample` to the core and the five current runners (`BBH_GPT_5.6_Luna`,
  `BBH_Gemini_Flash3.5lite_OpenRouter`, `BBH_Qwen`, `BBH_Gemma`, `BBH_Deepseek`).
- Follow `call_with_retries` and `openai_meta` in `aware_eval_core.py`: `once` returns text and meta,
  and the core adds `latency_s` and `n_attempts`. Metadata must stay with its own row under `--workers`.
- Establish what produced the BBH Qwen and Gemma numbers (D1) from their `config` column and report it.
- Remember both item loops in `run_tasks`, the threaded one and the serial one.

## WP3 · MMLU

- **Core-based:** `MMLU_GPT_5.6_Luna` and the Gemini runner. Same change as WP2. Which Gemini route
  follows D1.
- **Legacy, no core:** `MMLU_Qwen`, `MMLU_Gemma`, `MMLU_Deepseek`. Leave those scripts untouched and add
  a thin core-based `*_mmlu_sample.py` beside each, plus an sbatch script.
- **Prompt must match the reported run.** Qwen and Gemma use the text of `PROMPT_V2`; Deepseek's looks
  like `PROMPT_V1`. Compare rendered prompts byte for byte with the legacy script and show the result.

## WP4 · DocVQA

- **Core-based:** `DOCVQA_GPT_5.6_Luna`, `DOCVQA_Gemini_Flash3.5lite_Google`.
- **Legacy:** `qwen_DocVQA/qwen_eval.py`, `gemma_DocVQA/gemma_eval_half{1,2}.py`. Add thin core-based
  sample runners; the core's prompt is documented as byte-for-byte theirs, so verify it.
- Read [docvqa.md](benchmarks/tasks/docvqa.md) § traps first: shards share one daily quota, and empty
  rows must be counted.
- Local smoke needs the images; they are already in the local tree.

## WP5 · EmoBench

There is no shared core. Gemini, Qwen, Gemma and Deepseek are standalone scripts.

- Add a new sample runner per folder, on the shape of `EMO_GPT_5.6_Luna/gpt56luna_emo_eval.py`, which
  already writes every field. Swap in the client and parameters D1 selects. Legacy scripts stay as they
  are.
- Keep the upstream prompt and `reasoning_visibility.py`. Show the rendered prompts match the legacy
  script.
- **Check the scorer.** Legacy scripts compare with `==` on the parsed answer. Report whether that equals
  what produced the workbook's numbers before relying on it.
- EU is one item with two answers. Confirm how many calls an item costs; the 164-call figure above
  assumes one.

## WP6 · NegotiationToM

Mostly in place: `neg_eval_core.py` already has `--subsample`, `--tag` and `enable_row_fields()`.

- Add `sample10` to the tag allow-list in `parse_cli`.
- Opt in `NEG_Qwen`, `NEG_Gemma` and `NEG_Deepseek`: call `enable_row_fields()` and set `max_retries=0`
  with five attempts. Older runners stayed out precisely because SDK retries were on.
- Which Gemma folder runs: the reported number names Together, so `NEG_Gemma`, unless D1 says otherwise.
- `criteria_report.py` must not pick these tags up as criteria 4–5 runs.

## WP7 · Report and projection

Write `.claude/scripts/sample10_report.py` to read `results_sample10/` and output
`Results/sample10_estimates.csv` plus a summary. Reuse `usage_criteria()` and `PRICES` from
`criteria_report.py`.

- Add the missing prices with the date read: `qwen3.5-9b@deepinfra` ($0.10/$0.15 from
  [model-calls.md](model-calls.md)), Gemma on Together, Gemini native. Add the `FOLDER_PRICE` rows. An
  unpriced folder is refused, never guessed.
- Apply § How the estimate works: scaling by stratum, intervals, drift check. Legacy rows may have no
  `idx`, so match on question text and report any that do not match.
- Compare tokens only within a model family; compare across families in dollars and seconds.
- **Before submit:** use the smoke rows to project each cell's sample cost and full-run estimate, as a
  table for the user. Three rows is a rough guide, so flag it as such, especially for thinking-on cells.
- Compute D3's MultiChallenge and Wonderbread figures here, once judging has finished.
- Test on a small fixture where the correct answer is known by hand, including a stratum where N/n ≠ 10.

## WP8 · Review

Use the `reviewer` role, one pass per package, given the diff and not the writer's conclusion. Attack
these guarantees:

- No `--sample` means identical rows, filenames and prompts to before.
- A sample run cannot write into or resume from `results/`.
- `n_attempts` and `latency_s` are true under retries and under threads.
- The prompt and parameters match D1's choice for every cell.
- A failed row is counted as failed, not scored as a silent 0 that vanishes from the reliability rate.

**Done:** `STATUS: safe-to-run` per package.

## WP9 · Quest

**Blocked until the user says they have verified the scripts. Their call, not an agent's.** One agent
only, to avoid crossed syncs. Follow [quest-cluster.md](quest-cluster.md) and the `quest-sync` and
`submit-run` skills.

- Check `squeue` first. A judging run may be on the same keys.
- Transfer core, runners, sbatch scripts and `sample10.json` as one set, per benchmark, with `md5sum` on
  both sides. **The pre-submit hook only checks NegotiationToM**, so its pass proves nothing elsewhere,
  and it does not cover `*.json`.
- Submit one small benchmark and model first, confirm its first rows, then the rest. Per-key limits
  apply; do not stack several models on one key.

## WP10 · Monitor, pull, audit

Use `monitor-run` or `check-run`, then `pull-results` and `audit-results`. Judge by rows written, not job
state. Bad data falls under the standing kill-and-resync authorisation in `CLAUDE.md`, and the report
must say what happened to the rows. Then run WP7's report and **stop for the user**.

## WP11 · Record

Only once the user confirms, and as D4 decided. Rebuild from files on disk with
`build_criteria_workbook.py` and `build_results_db.py`. Move lasting facts into their references, such as
each benchmark's page and [criteria-runs.md](criteria-runs.md), then delete this file and its routing row.
