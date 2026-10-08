# AwareBench — benchmark card

<!-- size-budget: 10000 -->
<!-- One benchmark, eight template fields, and the traps the 2026-10-08 runner build measured.
     The deep analysis stays in AWARENESS_NOTES.md; this page indexes it and holds the operating facts. -->

Mission analysis, formulation and planning. Upstream `HowieHwong/Awareness-in-LLM`. No LLM judge
since 2026-10-05: its only judged rows were dropped. **Runners and a shared core exist (2026-10-08);
no real run yet.**

## Paths

| | Path |
|---|---|
| Local | `Transition_processes_benchmarks/Awareness_in_LLM` |
| Quest | **none yet** — checked 2026-10-08: no `Transition_processes_benchmarks/` under `/gpfs/projects/p32983/` and no awareness copy anywhere in it. Where the first transfer creates one is a phase-3 decision, made with the user |

## The one thing to know first: this folder holds two benchmarks

| | `dataset/AwareEval.json` | `New/` |
|---|---|---|
| Items | **4,075** | 6,580 across 37 files |
| Documented | yes, README + paper | no — upstream never describes it |
| In scope | **yes** | **no** — decided by the user 2026-08-05 |

They share no taxonomy, file format or metric. "The awareness benchmark" is ambiguous; say which.
`New/` stays documented in `AWARENESS_NOTES.md` §3–§4 as out-of-scope material.

## Layout

`aware_eval_core.py` at the benchmark root holds everything that must not differ between models:
task classification, the prompt, the one scorer, the retry/timeout/billing harness, checkpoint,
shards and aggregation. One folder per model, `AWARE_<Model>/`, each holding a ~50-line runner (client
plus parameters) and `run_aware.sh`. Keys come from the benchmark-root `.env` (gitignored).

| Slot | Folder | Model id | What the runner sets |
|---|---|---|---|
| OpenAI | `AWARE_GPT_5.6_Luna` | `gpt-5.6-luna` | `reasoning_effort="low"` (run stops if dropped), `max_completion_tokens=2048`, `seed=42` |
| Gemini | `AWARE_Gemini_Flash3.5lite_OpenRouter` | `google/gemini-3.5-flash-lite` | `reasoning.effort="minimal"` (fixed), `max_tokens=2048`, `seed=42` |
| Gemma | `AWARE_Gemma_DeepInfra` | `google/gemma-4-31B-it` | **no reasoning parameter**, `temperature=0`, `max_tokens=2048`, `seed=42` |
| Qwen | `AWARE_Qwen` | `Qwen/Qwen3.5-9B` | `reasoning={"enabled": False}` (fixed), `temperature=0`, `max_tokens=2048`, `seed=42` |
| Deepseek | `AWARE_Deepseek` | `deepseek-reasoner` | prompt ceiling as a system message, `temperature=0`, `max_tokens=8192`, `seed=42` |

The settled configs behind the first two are `model-parameters.md` § Settled; the 2048 caps are
those measured on EmoBench, whose items have the same shape. Every runner negotiates its surface at
startup and writes what was accepted onto every row; a cap listed in a runner's `required` that is
dropped or changed in negotiation stops the run. A seed being *accepted* is not it being honoured.

## Expected counts — the acceptance test

`aware_eval_core.EXPECTED`, measured 2026-10-08; `Output_template/openai_awareness_per_task.csv`
carries the same figures. A run producing different counts has a bug in sharding, dedup or resume.

| Task | Rows | After dedup | Questions |
|---|--:|--:|--:|
| `capability` | 600 | 598 | 299 |
| `mission_explicit` | 966 | 966 | 322 |
| `mission_implicit` | 327 | 297 | 99 |
| `emotion` | 200 | 200 | 200 |
| `culture` | 522 | 522 | 522 |
| `perspective_mcq` | 900 | 892 | 298 |
| `perspective_story_2nd` · `_1st` · `_reality` · `_memory` | 170 · 166 · 91 · 73 | same | same |
| **Total** | **4,015** | **3,975** | **2,240** |

Budget per model: **4,015 generation calls, no judge calls**.

## Scoring

Accuracy per task, permutation-averaged for the four permuted tasks. Aggregation is the paper's
formula with `mission` = mean(explicit, implicit) — `AWARENESS_NOTES.md` §2.6 and §5.0; `perspective`
is the 170 second-order rows (§2.8). One lenient extractor per answer form, in the core. Rows carry
three outcomes, not two: right, wrong, or **`parse_fail`** (no answer could be read). On a story
question, a response that says it cannot be answered is **`DECLINED`**: wrong, and counted apart, so
that `parse_fail_rate` measures only the extractor, which is what it gates (§5.5). An extractor
accepts one answer only — a second un-negated option, verdict or location makes it a parse failure.
**`test_aware_scorer.py` pins every case** (adversarial ones from the 2026-10-08 review, real smoke
responses, every gold answer written several ways); run it after any scorer change. Every row keeps
`raw_response`, and `--score-only` re-scores from it: **a scorer fix is a rescore, never a rerun.**

## Output and logs

Ours are `AWARE_<Model>/results/` — named after the model. `results/<task>/<model_slug>.jsonl` is the
checkpoint and the full record (prompt, config, usage, `finish_reason`, `served_model`, backend);
`results/<model_slug>_{capability,emotion,questions,awareness_per_task,awareness_overall}.csv`
follow `Output_template/README.md`. Shard tag `_shard<i>of<n>`, empty at one shard. `COMPLETE 1` in
the overall file means every task has its expected counts, no row is empty, all rows share one
config and no merge problem exists; any problem is written there as `PROBLEM_n` and makes the run
exit 1. That file is what the workbooks read, so a score appears in it only once its task is
complete and the set has no problem; partial accuracies are in `awareness_per_task.csv`.
`--limit` writes to `smoke/` instead, gitignored, so a smoke row can never be resumed past or
reported. SLURM logs are `log_<arrayindex>.txt`/`.err` per folder — distinct per array task.

## Run order

1. Local smoke: `python <runner> --limit 3` (30 calls). Done 2026-10-08 for all five.
2. User verifies the scripts — phase 3 does not start before.
3. Transfer core, runners, job scripts and `.env` as one set; `md5sum` both sides.
4. `sbatch run_aware.sh` from inside each `AWARE_<Model>/`: ten array tasks, one per task, at most
   five at once, largest first. Each rewrites the summary files on exit;
   `results/<model_slug>_awareness_overall.csv` reads `COMPLETE 1` when all ten are in.
5. Gate before reading any score: `parse_fail_rate` near zero per task, then the controls (§2.8).

## Showing reasoning

Upstream's README says nothing about it. The decision is the dataset's own: every prompt fixes the
answer format itself — *"Please answer the following questions and return A, B or C only"*,
*"simply return 'correct' or 'wrong'"* — and the prompt is read from the data at run time and sent
unchanged, so nothing about visibility is hardcoded. The story prompts are the exception: they say
only *"answer my question"*, and models narrate before answering — the location extractor is built
for that. Hidden reasoning is capped regardless (`model-parameters.md`).

## Traps — each produces wrong numbers that look right

- **A backspace inside 2,193 prompts.** Every three-option prompt joins B and C with `\x08`, once,
  always before `C.`, so as shipped option C reads as the tail of option B. The runners send
  `--prompt repaired` (newline restored) by default; `verbatim` sends the shipped bytes. The mode is
  in the run config, so the two never mix in one result set. Not in the notes before 2026-10-08.
- **Story questions have no type field.** The four ToMi types are told apart by wording
  (`think that` / `look for` / `really?` / `at the beginning`); those four patterns reproduce
  170/166/91/73 exactly and anything else raises.
- **Group story controls by (question, story), not question text.** Several stories ask "Where is
  the broccoli really?" with different answers; grouping by text alone gave 163/48/46 and made
  distinct items look like permutations of one question. Corrected 2026-10-08.
- **`deepseek-reasoner` resolves to `deepseek-flash`** (`model-calls.md`). Every smoke row came back
  `served_model=deepseek-flash` (2026-10-08). The alias's target has moved since earlier
  benchmarks ran it; `provider-gotchas.md` records the legacy model's retirement.
- **Smoke-test behaviour worth expecting at scale:** Gemma and Qwen decline second-order belief
  questions ("the story does not provide this information") — all 6 such rows in the second smoke
  run. DeepSeek's ceiling is a request, not a cap: reasoning reached 3,190 tokens on one item.
- **Story answers are narrated.** The first scorer version credited the gold container when a
  response merely mentioned it — including in a refusal. The location extractor now reads, in
  order, the `Answer:` span, the clause after the question's own verb, the conclusion sentence.

## Read this before touching it

`AWARENESS_NOTES.md` in the folder is deep and authoritative. Four sections change what a run means:
§2.5 (the schema changes per dimension — one parser cannot read the file), §2.7 (the released file is
not the file the paper evaluated), §2.8 (`perspective` is scored second-order only), §4 (what is
broken as delivered). §5 is the run plan; the judge we chose not to run is
`LLM_as_judge/JUDGE_RECORD.md` §3.
