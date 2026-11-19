# MMLU — benchmark card

<!-- size-budget: 12500 -->
<!-- One job — the card for one benchmark — and it runs long for reasons bbh's does: nine slots,
     two coexisting layouts, two prompts that are not interchangeable, two scorers still in play,
     and a broken run kept as a record beside the folder that finished it. Raised 2026-09-07 when
     the results table landed; the split that would shrink it is a scoring page like bbh's, and
     that is worth doing when the uniform rescore happens, not before. Raised to 12500 on
     2026-11-15 for the 10% sample section (moves out when the sample is recorded and
     sample10-plan.md is retired) and the two open scorer gaps (go when the user decides them). -->

General task ability, multiple choice. Accuracy. No LLM judge.

## Paths

| | Path |
|---|---|
| Local | `Tasks_benchmarks/mmlu` |
| Quest | **`/projects/p32983/Tasks_benchmarks/mmlu`** — this account's own copy, synced 2026-08-30, 29 files verified byte-identical by `md5sum`. Code and data only |
| Quest — not ours | `/projects/p32983/mmlu` is **`cpz1698`'s**, flat, last touched 2026-06-09. Do not write into it |

## Layout — rebuilt 2026-08-29 on the EmoBench convention

`MMLU_<Slot>/` per model, holding `<vendor>_mmlu_eval.py`, `run_mmlu.sh` and
`results/<Subject>/<model-slug>.{jsonl,csv}` + `_overall.csv`. The 13 subject JSONs live in `data/`
and a shared `mmlu_eval_core.py` owns the scorer, both prompts and the write path — and, in its
last section, the 10% sample harness (§ The 10% sample rerun).

**Two layouts coexist, and only the newest slots use the one described above.** The seven older
slots are flat — `<vendor>_<Subject>.csv` beside a `<vendor>_overall_results.csv` roll-up — while
`MMLU_GPT_5.6_Luna`, `MMLU_Gemini_Flash3.5lite_OpenRouter` and `MMLU_Gemini_Flash3.5lite_Google`
use `results/<Subject>/`. Count the subject CSVs, never the roll-up: **Gemma's covers 8 of 13
subjects and Qwen's is an empty file, yet both have all 13 on disk** — the same trap bbh's
slot-level roll-up carries.

**`MMLU_Gemini_Flash3.5lite_Google`, added 2026-09-07, is where the Gemini slot is being finished.**
The OpenRouter run of 2026-08-30 wrote all 3,943 rows but 2,188 of them empty: its balance ran out
mid-run (HTTP 402, `limit_source: openrouter_credits`) and the four subjects after `Miscellaneous`
are 100% empty. `MMLU_Gemini_Flash3.5lite_OpenRouter/` is kept **untouched as the record of that
run**; this folder began as a byte-identical copy of its `results/` and is being completed on the
native Google AI Studio route. It carried the OpenRouter rows for one day and then redid them too,
so it is now a single configuration: `route=google_aistudio`, `thinking_budget=128`, `seed=42`,
`max_tokens=8192`, prompt v2, on every one of its 3,943 rows. **`prune_non_google_rows.py` is how
that was done** — it drops any row whose `config` lacks `route=google_aistudio` and deletes the
merged files, after which the ordinary 5-shard resume re-asks exactly what it removed. It reports
and changes nothing without `--apply`.

`MMLU_Llama` has a runner and no subject CSVs at all.

## Expected counts

**13 subjects, 3,943 items**, counted 2026-08-29. Sizes are very uneven (100 to 895), so the
macro-average over subjects the workbook reports is **not** the same as pooling all items.
`answer` in the data is an **index as a string** — `"2"` means `choices[2]`.

## Results — every model that has a runner has run, verified 2026-09-07

All 13 subjects / 3,943 items each. Macro is the mean of the 13 per-subject scores, read from the
per-subject files — **never from `<vendor>_overall_results.csv`**, which is stale for two slots.

| Slot | Folder | Macro | empty | Scorer | Among the six |
|---|---|---|---|---|---|
| OpenAI | `MMLU_GPT_5.6_Luna` | **0.9176** | 0 | `mmlu_lenient_v1` | yes |
| Deepseek | `MMLU_Deepseek` | **0.9174** | 0 | per-runner `==` | yes |
| Gemini | `MMLU_Gemini_Flash3.5lite_Google` | **0.9135** | 0 | `mmlu_lenient_v1` | yes |
| Gemma | `MMLU_Gemma` | **0.9071** | 3 | per-runner `==` | yes |
| XAI | `MMLU_XAI` | **0.9018** | 0 | per-runner `==` | yes |
| Qwen | `MMLU_Qwen` | **0.8802** | 20 | per-runner `==` | yes |
| — | `MMLU_Gemini_Flash2.5` | 0.8811 | 0 | per-runner `==` | no — superseded |
| — | `MMLU_GPT_4o_mini` | 0.8011 | 0 | per-runner `==` | no — superseded |
| — | `MMLU_Llama` | — | — | — | no — never produced rows |
| — | `MMLU_Gemini_Flash3.5lite_OpenRouter` | 0.607 (unusable) | 2,188 | `mmlu_lenient_v1` | no — the broken run, kept as a record |

**Gemini's column is one route and is reported.** The folder began as a copy of the OpenRouter run
so its 1,755 answered rows would not be paid for twice, which made the column two configurations;
on 2026-09-07 the user decided to redo those rows here as well. They were pruned by
`prune_non_google_rows.py` and re-asked — 1,755 calls, exactly the number pruned — so all 3,943 rows
now carry `route=google_aistudio`. **The macro landed on 0.9135 both before and after, which is a
coincidence at four decimals and not a sign the rerun did nothing**: `Formal_logic` moved
0.9365 → 0.9762, `Jurisprudence` 0.9352 → 0.9074, `Marketing` 0.9701 → 0.9573.

**Two scorers are still in play here, and that is the open item.** Only the two newest slots import
`mmlu_eval_core`; the other seven carry the score their own runner wrote with `==`. Until a uniform
rescore is run, a small gap between two columns may be the scorer rather than the model — the exact
condition the one-scorer rule exists to remove, and bbh moved by 0.19–0.64 when it was fixed there.
The rescore is offline work: the responses are all on disk.

Per-subject cells live in the workbooks, `Tempo_results.xlsx` for all nine columns and `Final_result.xlsx`
for the selected models, each with its sources on the `Provenance` sheet.

## Two prompts, and they are not interchangeable

v1 (Deepseek, GPT-4o-mini, Gemini-2.5) shows the choices as a raw list and asks for the answer
**text**; v2 (Gemma, Llama, Qwen, XAI) labels them `A.`–`D.` and asks for a **letter**. Four runners
were asked for a letter and three for the text — a materially different task, not a formatting
difference. **New runs default to v2**, which leaves Deepseek alone on v1 among the six. The prompt
version is part of the run config and a resume across a change of it is refused. **The 10% sample
runs v2 for all five, Deepseek included** (the user, 2026-11-15) — its Deepseek accuracy is not the
reported v1 number's like for like.

## Scoring — one matcher, imported, for every model

`mmlu_eval_core.py::score_response`, `mmlu_lenient_v1`. **MMLU had exactly bbh's split**: the
`*_eval.py` runners compared text with `==` while a separate `*_rescore.py`, written for only three
of seven providers, accepted a letter, an index or letter-plus-text. Seven branches now, all naming
the same choice a different way; the two loosest fire only when the model emitted `Final Answer:`.
Built on the lessons bbh paid for across five scorer versions — [bbh-scoring.md](bbh-scoring.md).

## The 10% sample rerun — written 2026-11-15, not submitted

The plan, its decisions and each model's settled config: [sample10-plan.md](../../sample10-plan.md),
WP3. What is MMLU's own:

| Model | Folder | Runner |
|---|---|---|
| GPT | `MMLU_GPT_5.6_Luna` | `gpt56luna_mmlu_eval.py --sample sample10` |
| Gemini | `MMLU_Gemini_Flash3.5lite_OpenRouter` — OpenRouter, not the route of the reported number; its `results/` stays the broken run's record | `gemini35lite_mmlu_eval.py --sample sample10` |
| Qwen | `MMLU_Qwen_DeepInfra` (new; `MMLU_Qwen/` untouched) | `qwen_di_mmlu_sample.py` — prompt ceiling, `CutRetry` |
| Gemma | `MMLU_Gemma_DeepInfra` (new; `MMLU_Gemma/` untouched) | `gemma_di_mmlu_sample.py` |
| Deepseek | `MMLU_Deepseek`, beside the legacy script | `deepseek_mmlu_sample.py` — prompt ceiling, `CutRetry` |

- **401 items in 13 strata**, from `sample10.json`; `idx` indexes the whole subject.
- **Output** `results_sample10/<Subject>/<slug>[_shardNof5].{jsonl,csv}` + `_overall.csv`, roll-up
  `<slug>_mmlu_overall.csv`; a `--limit` smoke goes to `smoke_sample10/`. Never `results/`. Each
  folder's `run_mmlu_sample10.sh` runs 5 shards; `merge_mmlu_shards.py --sample sample10` merges
  them and exits 1 unless the merged ids are exactly the sample's. Logs: absolute paths in the
  model folder, appended. Halt markers `sample10_[shard<N>_]*_HALT.txt`, a smoke's
  `smoke_sample10_*`; each run clears only its own prefix.
- **Calls run serially in the main thread** (SIGALRM watchdog, re-armed per `CutRetry` stage);
  `--workers` above 1 is refused. Exit 1: failed or cut rows (resubmit); 2: an odd forced answer;
  3: thinking came back on — reasoning text, tokens or inline `<think>` (Qwen, Gemma: the shard stops
  at that row, writes `*_THINKING_HALT.txt`; a resubmit finding such a row stops before any call).
- **Rows add** `pred_letter`, `parse_ok`, `failed`, the per-row usage fields, `served_model`,
  `backend`, `reasoning_chars`, `think_tag_calls`, `halted`, `CutRetry`'s marks, `sample`, `ts`. `config` is a JSON object here, a
  flat string on legacy rows. **`parse_ok`** = the one scorer, asked about each option in turn,
  accepts at least one — so a row scored 1 is always `parse_ok`.
- **What a row cost:** the billed ledger, [evaluation-criteria.md](../../evaluation-criteria.md)
  § Per-row fields.
- **Time.** `latency_s` is the answering attempt's alone, without a stage kept from an earlier one (that is in
  `first_`/`retry_latency_s`). A row failed after a cut first stage has no `first_*`.
- **A resubmit keeps a row's history.** A stored failed or cut row stays in the file until a new row
  replaces it; the new row's `billed_*` and `attempts_without_usage` are cumulative over every run,
  `n_attempts` is this run's, and `prior_n_attempts`, `prior_failed_runs` (earlier runs that ended
  failed *or* cut), `resumed_after_failure` record the earlier runs. A stored cut row whose re-call
  fails is replaced by the failed row — its bill intact, and still loud (failed, exit 1).
- **A stopped run still writes the row it was on** — SIGTERM, a billing, quota or malformed-request
  halt, FAILURE_HALT. Mid-call: a failed row marked `halted`, with every call it was billed; after
  the call returned: the answered row. The file's bill equals what was billed.
  **Known limits:** a second SIGTERM during recovery or the final save can lose unsaved rows; a stop
  after a call returned but before `call_with_retries` returns is written failed and `halted`, then
  re-called; a kill in `create` before the request leaves may overcount `attempts_without_usage` by 1.
- **Without `--sample` nothing changed**: `test_sample10.py` checks the legacy code, files and
  requests byte for byte against `1cc5bbd5`.
- **Smoke, 2026-11-15** (`--limit 3`, 39 rows each, all exit 0): Qwen's and Gemma's rows are from
  the final code; the others' predate the last field renames, with identical requests.

## Its own traps

- **The rescore scripts are the cheap path.** A scoring change does not need a rerun: `*_rescore.py`
  re-derives the numbers from stored responses. Check for one before spending a provider call.
- **A `dotenv/` directory sits among the provider subdirectories** — it is not a provider.
- **Two scorer gaps in `mmlu_lenient_v1`, open since 2026-11-15 — the user decides.** A fix to
  either is a new scorer version and a rescore of every model.
  - **`Final Answer: <D>` scores 0 even when D is right.** v2's template ends `"Final Answer: <A, B,
    C, or D>"` and some models keep the brackets, which are not stripped, so the row is also a parse
    failure. Gemma on DeepInfra: 3 to 6 of 39 rows in each of three smokes (`<A>`, `<D>`), most of
    them right — all 3 in the last. Stored rows: 4 of Gemma's, 1 of Qwen's.
  - **A reply with no marker that starts "A " scores as choosing A.** Branch 4 (letter then text)
    runs before the `has_marker` guard, on the whole response — so "A company's duty…" names A.
