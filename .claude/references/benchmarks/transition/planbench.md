# PlanBench — benchmark card

<!-- size-budget: 12500 -->
<!-- One benchmark, the template's fields, and the traps found building its runners on 2026-11-09.
     Every count was measured from the local tree that day; the commands are in the code cited. -->

Strategy formulation. Upstream `karthikv792/LLMs-Planning`. **Scored by the VAL validator and
upstream's own text parsers — no LLM in our scoring.** Runners for the five selected models written
2026-11-09, smoke-tested locally. The 10% pilot was submitted to Quest 2026-11-11 on the user's
go-ahead (`## Run order`); not yet pulled or audited, so no recorded results.

## Paths

| | Path |
|---|---|
| Local | `Transition_processes_benchmarks/LLMs-Planning_bench` |
| Quest | `/gpfs/projects/p32983/Transition_processes_benchmarks/LLMs-Planning_bench` — created by the first sync, 2026-11-11 (it did not exist on 2026-11-09). Holds only what generation reads: `plan_eval_core.py`, the five `PLAN_*` folders, `plan-bench/prompts/`. `.env` was copied **on Quest** from `Awareness_in_LLM/.env`. `/projects/p32983/LLMs-Planning-main` is `cpz1698`'s copy: not ours, not touched |

## Layout — which of upstream's two trees is used

| Tree | What it is | Used? |
|---|---|---|
| `plan-bench/` | the PlanBench paper: 10 tasks, **prompts already generated** in `plan-bench/prompts/<domain>/<task>.json` | **yes** |
| `llm_planning_analysis/` | the README leaderboard: zero-shot task 1 | no — its prompt generator needs Fast Downward, its scorer defaults to a `gpt-4o` translator, and Randomized Mystery has no config |

The two trees share file names and differ in code, so a path alone does not say which ran.

Ours, at the benchmark root: `plan_eval_core.py` (generation: rows, messages, retry/halt harness,
checkpoint, shards, generation health — **never scores**), `plan_scorer.py` (offline scoring),
`setup_scorer.sh` (builds what the scorer needs), and one folder per model holding a ~50-line runner
and `run_plan.sh`. Keys in the root `.env` (gitignored).

| Slot | Folder | Model id | What the runner sets |
|---|---|---|---|
| OpenAI | `PLAN_GPT_5.6_Luna` | `gpt-5.6-luna` | `reasoning_effort="low"` (required), `max_completion_tokens=16384`, `seed=42` |
| Gemini | `PLAN_Gemini_Flash3.5lite_OpenRouter` | `google/gemini-3.5-flash-lite` | `reasoning.effort="minimal"` (fixed), `max_tokens=8192`, `seed=42` |
| Gemma | `PLAN_Gemma_DeepInfra` | `google/gemma-4-31B-it` | no reasoning parameter, `temperature=0`, `max_tokens=4096`, `seed=42` |
| Qwen | `PLAN_Qwen_DeepInfra` | `Qwen/Qwen3.5-9B` | `extra_body.reasoning.enabled=False` (fixed), `temperature=0`, `max_tokens=8192` (**was 4096 — it bound, see Run order**), `seed=42`; route `deepinfra` on every row; call timeout 900 s |
| Deepseek | `PLAN_Deepseek` | `deepseek-flash` | prompt ceiling as a second system message, `temperature=0`, `max_tokens=16384`, `seed=42` |

**Every output cap is chosen, not measured** — raised from AwareBench's short-answer caps because an
answer here is a multi-step plan. The pilot's `finish_length` column is the measurement.

Messages: upstream's chat system message (`utils/llm_utils.py`) plus the shipped `query` verbatim;
no `stop` (upstream sent none to chat models). Rows store `query_sha1`, not the prompt.

## Expected counts — the acceptance test

`plan_eval_core.EXPECTED`: 53 prompt files ("slices"), checked against the disk at every load.

| Domain | Tasks shipped | Rows per task | pilot10 per task |
|---|---|--:|--:|
| `blocksworld` | all 10 | 500 | 50 |
| `blocksworld_3` | all 10 + `task_3_plan_verification_with_llm_plans` (99) | 100 | 10 |
| `mystery_blocksworld` | all 10 | 500 | 50 |
| `mystery_blocksworld_3` | 9 (no task 5) | 100 | 10 |
| `logistics` | all 10 (task 5: 12 rows) | 285 | 29 (task 5: 2) |
| `depots` | task 1 only | 500 | 50 |
| `obfuscated_deceptive_logistics` | task 1 (285), task 2 (207) | — | 29, 21 |
| **Total** | 53 slices | **15,568** | **1,563** |

pilot10 = every 10th instance by sorted `instance_id` in every slice (`ceil(n/10)`). Per task,
pooled: t1 228, t2 170, t3 149 (+10), t4 149, t5 112, t6 149, t7 149, t8_1–t8_3 149 each.

## Scoring — `plan_scorer.py`, scorer `pb_lenient_v1`

| Tasks | Headline (`correct`) | How |
|---|---|---|
| 1, 4, 5, 6, 8_1–8_3 | plan valid | upstream text→PDDL parser, then VAL; tasks 4 and 6 against upstream's `new_instance`, task 5 against the generalized instances |
| 2 | valid **and** optimal length | optimal = length of the shipped ground truth (`astar(lmcut())`, unit-cost domains) — no Fast Downward run |
| 3 | verdict valid/invalid (binary) | ported `parse_output`; `acc_w_type` and `acc_w_expl` reported beside it |
| 7 | resulting state equal | upstream `text_to_state`, sorted compare |

`acc` keeps empty responses in the denominator; `acc_answered` is upstream's denominator. One lenient
extraction for every model (`extract()`): cut at the first `[STATEMENT]`, take the last
`[PLAN]…[PLAN END]` block, strip markdown, bullets and step numbers. An empty plan still goes to VAL.

**Self-test** (`plan_scorer.py --selftest`): upstream's stored GPT-4 responses, blocksworld and
blocksworld_3, all 10 tasks. In `upstream` mode it reproduces upstream's stored verdict on **every
row of 9 tasks**; task 7 differs on 2 of 500 and 2 of 100 (upstream's older state parser, both
directions). Lenient mode changes one verdict (task 4, in the model's favour). Mystery, logistics and
depots have no upstream fixture — the pilot is their first test.

Needs VAL and `tarski` 0.7.0, which Quest's shared env lacks: **score locally**, after
`setup_scorer.sh` (VAL built for this platform; Python 3.12 env), with
`.scorer-env/bin/python plan_scorer.py PLAN_<Model> --model <id> --sample pilot10`.

## Output and logs

- generation: `PLAN_<Model>/results_pilot10/<slug>_shard{k}of5.jsonl` (full set: `results/`; a
  `--limit` run: `smoke/`, gitignored), `<slug>_negotiated_params.json`, and per slice
  `<slug>_health.csv` + `<slug>_health_overall.csv` (`complete` 1 = every count met, no empty row,
  no fragment a backend cut off, one config). **`complete` does not cover `finish_length`**: a row
  that hit our output cap is a measurement of whether the cap binds, not a missing row — read that
  column before trusting a number. `<slug>` is the model id with `/`→`-`, `.`→`_`.
- scoring: `<slug>_scored.jsonl`, `<slug>_scores_by_slice.csv`, `<slug>_scores_by_task.csv`.
  **The by-task table leaves out the two `obfuscated_deceptive_logistics` slices** (they score 0 for
  every model, below); the by-slice table keeps them. A plan on which VAL **crashes or prints
  `Error in type-checking!`** is scored invalid with `val_crash` set (a column in both tables); VAL
  silent for any other reason is a `scorer_error` and makes the scorer exit 1, as does a crash share
  above 5% of the plans that reached VAL; a missing PDDL file or VAL aborts it.
- SLURM: `log_<k>.txt` / `.err` per shard in the model folder. Halt markers are tagged by shard
  (`shard2_FAILURE_HALT.txt`). **A shard exits 1 if any of its own rows is empty or interrupted**, so
  `COMPLETED 0:0` is a usable signal; resubmitting the shard retries exactly those rows.
- **A backend-interrupted reply is never an answer**: `finish_reason` `error` (OpenRouter, on HTTP
  200, with a fragment as content) or `insufficient_system_resource` (DeepSeek) is retried, and if
  every attempt ends so the row is written empty.
- **`--slices` is refused without `--limit`**: on a real run it would rewrite the shard file without
  the rows outside the selection. A repair is a plain resubmit of the shard. SIGTERM (SLURM's wall
  limit, `scancel`) is turned into a clean exit that saves the rows already paid for.
- **Upstream's `results/` and `responses/` under both trees carry no model of ours** — not our numbers.

## Run order

1. User verifies the scripts. 2. `quest-sync`: create the Quest folder; send `plan_eval_core.py`,
the five `PLAN_*` folders and `plan-bench/prompts/` (105 MB); copy `.env` on Quest. Generation reads
nothing else. 3. In each `PLAN_*`: `sbatch run_plan.sh` — five shards of the pilot at once (wall
time 4 h; the 15-row local smoke measured median latencies of 1–4 s for GPT, Gemini, Gemma, Deepseek,
about 20–30 min a shard). **Qwen is the long pole**: its replies are long (it writes visible
reasoning even with `reasoning` off) and this DeepInfra route's speed varies — see "Qwen's route speed" below. Its `run_plan.sh` asks for 48 h, the `normal`
partition's `MaxTime` (2 days). More shards for Qwen alone would cut that, but 5 is the standing
ceiling (`quest-cluster.md`) and raising it is the user's call.
4. `monitor-run`. 5. `pull-results`. 6. Score locally (above); `audit-results`. The full set is
15,568 rows, ~3,100 a shard: size its wall clock from the pilot's `latency_s_median` first.

**Pilot submitted 2026-11-11 14:44** (arrays 8157873 GPT, 8157875 Gemini, 8157876 Gemma, 8157877
Deepseek, 8157878 Qwen; 64 files md5-matched, Quest `load_rows` gave 1,563 / 15,568 unique uids).
Observed in the first 40 minutes: **Gemini finished** all five shards in 15–17 min (1,563 rows,
0 empty, `finish_length` 0, `complete` 1); GPT's shards 0 and 1 within 36 min. **Shard cost is not
even.** The mystery_blocksworld shards (shard 3 = mystery only) write long answers — Gemma 692
tokens, 44 s a call, on a prompt where the 15-row smoke had medians of 1.6 s — and DeepInfra hung
calls on Gemma and Qwen (`TIMEOUT 1/5`, 240 s / 600 s) that the retry absorbs. A shard that does not
fit its wall time exits on SIGTERM with its rows saved; resubmit only that index with a longer
limit (`sbatch --array=3 --time=… run_plan.sh`, command-line options override the `#SBATCH` lines).

**Where it ended up, 2026-11-11 22:30.** GPT and Gemini: 1,563 rows each, 0 empty. Gemma: shard 4
reached its 4 h wall at 280/311 and was resumed unchanged (job 8215431, 1 h 09 min): 1,563 rows, 0
empty, `complete` 1. Deepseek: 1,563 rows, **11
empty** — all `task_2` (logistics 1, mystery_blocksworld 4, obfuscated 6), each five attempts that
spent all 16,384 tokens on reasoning; shards 2–4 exit 1 for that reason by design. At
`temperature=0, seed=42` a resubmit reproduces them, so the user **accepted them as no-answer rows**
(they stay in `acc`'s denominator). Cap hits elsewhere: Deepseek 17 rows (1.1%), Gemma 28 (1.8%).

**Qwen's 4,096 cap bound, and the run was replaced.** Stopped at 7 h 42 min with 760 rows, 0 empty:
**227 (29.9%) `finish=length`, only 13 of them with `[PLAN END]`** — mystery_blocksworld 132 of 180,
logistics 44 of 66, depots 12 of 37, blocksworld 36 of 367. The truncated replies are long chain of
thought cut mid-sentence (no repetition loops; at 480 rows the other replies' completion tokens had
median 1,168, p90 2,940). Per-shard latency median 59 s (easy blocksworld shard) to 276 s (mystery shard), 5
shards. The user raised the cap to 8,192 on 2026-11-11 (timeout 600→900 s, wall time 16→48 h) and
the 760 rows (with their logs) were archived on Quest as `results_archive_max_tokens4096_*`, not
mixed in, and Qwen was resubmitted as job 8223099 (5 shards, 4 h 19 min to 17 h 30 min, all
`COMPLETED 0:0`). Local smoke at 8,192 on two mystery_blocksworld rows: both ended `stop` with
`[PLAN END]`, at 2,612 and 5,293 tokens (185 s, 315 s) — the second is one a 4,096 cap would have cut.

**The 8,192 cap still binds, audited 2026-11-12.** 1,563 rows, 0 empty, one config, but 394
`finish=length` — 349 of the 1,513 rows in scorable slices (23.1%), **252 of 500 in
mystery_blocksworld**, 40 of 263 in logistics, 15 of 500 in blocksworld. Those rows score 4.9% against
54.6% for the rows that finished, and only 53 of the 349 reached `[PLAN END]`. They are long thought
cut short, not loops (mean share of the most repeated line 1.6%). Raising 4,096→8,192 rescued 104 of
the 227 rows the old run had truncated (36 of them correct); 123 still hit the cap. So **Qwen's
PlanBench number is a lower bound set by the cap**, and a cap that does not bind (≥16 k, as GPT and
Deepseek have) is 5–23 min for a reply that fills it (16,384 tokens at 52 down to 12 tokens/s). At
`temperature=0, seed=42` the route is not reproducible: of the 533 rows that had finished under the
old cap, only 60 came back with identical text.

**Qwen's route speed is not a constant.** Tokens/s over rows of at least 300 tokens: median 13.9 in the
4,096 run (p10 7.8, p90 57), 30.9 in the 8,192 run (p10 16.9, p90 56), hourly medians 16–88. The same
prompt (idx 0) streamed 11 tokens/s while five of our shards ran, and 52 tokens/s twice afterwards
with none. Replies under 60 tokens took a median 1.8 s, which is why shorter benchmarks felt fast. For
scale, medians on the other routes: Gemma on DeepInfra 23, GPT 98, Deepseek 164, Gemini 293. How much
is our own five streams and how much is DeepInfra's load at the hour is **not separated**. The
pilot's shards took 4.3, 7.3, 13.5, 16.0 and 17.5 h, so the full set (10× the rows) is 43–175 h a
shard at 5 shards.

## Showing reasoning

`plan-bench/README.md` is silent: no line asks for or forbids reasoning, and `plan-bench` has no
reasoning mode to switch (chain of thought exists only as `llm_planning_analysis`'s separate
`t1_cot` task, out of scope). Upstream's mode is the shipped prompt, unchanged, so there is nothing
for a runner to resolve at run time. Hidden reasoning is still capped per `model-parameters.md`.

## Traps

- **Upstream's obfuscated parsers are broken.** `text_to_plan_obfuscated` subscripts `dict_keys` and
  reads `data['actions']['action']`; upstream's bare `except` scores every resulting crash as a wrong
  plan. We import it unchanged, so both `obfuscated_deceptive_logistics` slices (task 1 and 2) score
  **0 for every model by construction**, with `parse_error` `TypeError` filled — seen on all five
  models' smoke rows, where GPT, Deepseek and Gemma wrote exactly the ground-truth plan and scored 0.
  Fixing the first line alone would hit a `KeyError` on the next (the yaml has no `'action'` key).
  The by-task tables leave these slices out; the by-slice tables show them. Whether to repair the
  parser or drop the slice is the user's call. `text_to_state_obfuscated` has a second bug
  (`obj.strip.isdigit()`).
- **A missing VAL scores everything 0 upstream** (`validate_plan` checks for "Plan valid" in empty
  output). `plan_scorer.validate` raises instead.
- **VAL crashes on malformed plans — upstream's Linux binary too.** The first scoring pass of the
  pilot stopped on 42 of 6,252 plans (0.7%) where VAL died with SIGSEGV (empty output) or printed
  `Error in type-checking!`. The shipped Linux `validate`, run on Quest on the same 42 inputs, gave the
  same two outcomes (exit 139 on 21, 255 with the message on 21), so it is not our macOS build. All 42
  plans had a defect of their own: unbalanced parentheses, wrong arity, or an object the problem does
  not have (Qwen's 21 were truncated rows that invented objects `f`–`i`). Upstream scores them wrong
  too. Treated as an invalid plan with `val_crash`, not as a `scorer_error`.
- **Upstream's stored results are not a reference.** Some prompt files were regenerated after them:
  task 3's ground truth differs on 381 of 500 rows. The self-test compares only matching rows.
- **Upstream scores only answered rows**; ours keeps empties in `acc`.
- **An empty plan can be right** (task 6: the event already achieved the goal) — never short-circuit
  it to wrong.
- `utils/__init__.py` reads `OPENAI_API_KEY` at import, and `evaluate_plan` runs Fast Downward on
  every instance; the scorer avoids both.
- The shipped `planner_tools/VAL/validate` is Linux-only; building in place would overwrite it.
- `tarski` 0.7.0 needs Python ≤3.12 (`typing.io`).
- Upstream's `.gitignore` has `*/*.sh`; `!PLAN_*/run_plan.sh` re-includes ours.
- The README leaderboard (zero-shot, `gpt-4o` extraction) is not comparable with these numbers.
