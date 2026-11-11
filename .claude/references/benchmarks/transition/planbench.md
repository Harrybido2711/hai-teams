# PlanBench — benchmark card

<!-- size-budget: 12500 -->
<!-- One benchmark, the template's fields, and the traps found building its runners on 2026-11-09.
     Every count was measured from the local tree that day; the commands are in the code cited. -->

Strategy formulation. Upstream `karthikv792/LLMs-Planning`. **Scored by the VAL validator and
upstream's own text parsers — no LLM in our scoring.** Runners for the five selected models written
2026-11-09, smoke-tested locally; the 10% pilot waits for the user's verification. No results.

## Paths

| | Path |
|---|---|
| Local | `Transition_processes_benchmarks/LLMs-Planning_bench` |
| Quest | **not present for this account** — checked 2026-11-09: this account's `Transition_processes_benchmarks/` holds only `Awareness_in_LLM`. `/projects/p32983/LLMs-Planning-main` is `cpz1698`'s copy: not ours, not touched. The first sync creates `/gpfs/projects/p32983/Transition_processes_benchmarks/LLMs-Planning_bench` |

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
| Qwen | `PLAN_Qwen_DeepInfra` | `Qwen/Qwen3.5-9B` | `extra_body.reasoning.enabled=False` (fixed), `temperature=0`, `max_tokens=4096`, `seed=42`; route `deepinfra` on every row |
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
  every model, below); the by-slice table keeps them. A plan VAL gave no verdict for is a
  `scorer_error` and makes the scorer exit 1; a missing PDDL file or VAL aborts it.
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
about 20–30 min a shard). **Qwen is the long pole**: its 15 smoke rows took 22 min — median 59.5 s a
call, max 349 s (4,096 tokens, the one `length` row, on the unscorable obfuscated slice), ~12
tokens/s on this DeepInfra route — so ~8 h a shard at 5 shards; its `run_plan.sh` keeps the 16 h wall
time. More shards for Qwen alone would cut that, but 5 is the standing ceiling
(`quest-cluster.md`) and raising it is the user's call. Qwen writes long visible reasoning even with
`reasoning` off, so watch `finish_length` in its pilot health file.
4. `monitor-run`. 5. `pull-results`. 6. Score locally (above); `audit-results`. The full set is
15,568 rows, ~3,100 a shard: size its wall clock from the pilot's `latency_s_median` first.

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
