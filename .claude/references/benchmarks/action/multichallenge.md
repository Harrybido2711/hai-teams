<!-- size-budget: 11000 -->
<!-- One job: MultiChallenge's operating facts. Larger than the default because it carries the
     judge pilot and the full run side by side, which share every path but the set, and since
     2026-11-01 the full run's per-model job record. -->
# MultiChallenge — benchmark card

Coordination, over multi-turn conversations. Upstream `ekwinox117/multi-challenge`. **LLM-judged, on
every item.** The **judge pilot** (2026-10-19) graded `gpt-5.6-luna` on a 10% sample with four
candidate judges; the user chose **Claude Haiku 5.5** on 2026-10-31. Since then all five selected
models have a runner for the **full run** (all 273 items), and Haiku 5.5 grades each. Generated on
Quest 2026-11-01 (run order below), graded by Haiku 5.5, and **recorded 2026-11-11** in both
workbooks and `results.db` (§ Results, below the run order).

## Paths

| | Path |
|---|---|
| Local | `Action_processes_benchmarks/Multi-challenge_bench` |
| Quest | `/projects/p32983/Action_processes_benchmarks/Multi-challenge_bench` — created 2026-10-19 by the pilot's first transfer (the whole vendored tree plus our files, md5-verified); `.env` copied there from our NegotiationToM folder on Quest |
| Shared | `Action_processes_benchmarks/` itself — `judge_core.py`, `test_judge_pilot.py`, `run_judge_full.sh`, `compare_judges.py`. **A sync md5s these too**: a check of this folder alone passes with the core stale |

## Layout

```
Multi-challenge_bench/
├── main.py · src/                upstream harness — src/evaluator.py holds JUDGE_PROMPT
├── data/benchmark_questions.jsonl  the 273 items; data/final_model_responses/ is upstream's
├── mc_eval_core.py               ours: sets, the generation loop (run_cli), verbatim prompt + schema, scorer
├── sample_uids.json              the pilot sample, drawn once (seed 20261019), committed
├── MC_GPT_5.6_Luna/              gpt56luna_mc_eval.py, run_mc.sh — one folder per model, each the
├── MC_Gemini_Flash3.5lite_OpenRouter/  client and parameters only (a judge_core.Generator):
├── MC_Qwen/ · MC_Gemma_DeepInfra/      gemini35lite_ / qwen_ / gemma_di_ / deepseek_mc_eval.py
├── MC_Deepseek/                  + run_mc.sh (short 4 h; Deepseek normal 12 h — it queues and reasons)
└── MC_Judges/                    mc_judge.py --judge <key> [--set full --producer MC_<Model>]
```

The judge configurations, `Generator`, call infrastructure and checkpoint are shared with
Wonderbread in `Action_processes_benchmarks/judge_core.py`. Pilot: `../run_judges.sh` (array over
the four judges) and `../compare_judges.py`. Full run: `../run_judge_full.sh` (array over the five
models, Haiku 5.5). Pinned checks: `../test_judge_pilot.py`.

## Expected counts

273 items — `INFERENCE_MEMORY` 113, `INSTRUCTION_RETENTION` 69, `SELF_COHERENCE` 50,
`RELIABLE_VERSION_EDITING` 41; `PASS_CRITERIA` is `"YES"` on all 273. `mc_eval_core.load_items`
refuses any other count. **Pilot sample: 27** — 11 / 7 / 5 / 4. Judge rows: 27 × 3 replicates = 81
per judge. **Full run:** 273 generation rows per model; 273 × 3 = 819 judge rows per model (Gemini: 272 × 3 = 816 — one item is not scored, see below).

## Scoring

Upstream's, unchanged: an item passes on `verdict == PASS_CRITERIA`; per-axis pass rate; headline
the **macro** mean over the four axes (micro reported beside it). The judge sees the final response
and the `TARGET_QUESTION` rubric only. What our runner repairs, and why, is
`LLM_as_judge/JUDGE_SELECTION.md` §7.2 D1–D2; how it is posed through OpenRouter is the docstring of
`mc_eval_core.py`. A judge row is `ok`, `parse_fail`, `refused` or `failed`; only `ok` carries a
verdict, and a unit's label is the majority of its replicates (a tie is unjudged, never NO).

## Output and logs

Ours, each directory named after the model that wrote it, one row per item (generation) or per item
× replicate (judge):

- pilot: `MC_GPT_5.6_Luna/results/gpt-5.6-luna.jsonl`, `MC_Judges/results/<judge>/gpt-5.6-luna.jsonl`;
- full run: `MC_<Model>/results/full/<model id, "/" → "-">.jsonl` (e.g. `Qwen-Qwen3.5-9B.jsonl`) and
  `MC_Judges/results/full/haiku-5.5/<same name>.jsonl`. A judge refuses rows generated for another set.

`--limit` writes `smoke/` instead, gitignored. Rows are appended and flushed per call; resume skips
finished uids and refuses a different config. SLURM logs `log_<jobid>.txt` per folder; the judge
arrays log to `../judge_log_<job>_<task>.txt` (pilot) and `../judge_full_log_<job>_<task>.txt`.

## Run order

1. Local smoke — done 2026-10-19, after the review fixes: generation `--limit 3` (3 ok,
   `finish=stop`), each judge `--limit 2` (24 calls, all ok, every one on its pinned backend).
2. **The user verifies the scripts.** Phase 3 does not start before.
3. Transfer as one set with Wonderbread and the shared files; `md5sum -c` against a local manifest.
   Done 2026-10-19 for the reviewed code: 1,347 files, exit 0 (Quest also holds 13 `__pycache__`
   files from running `test_judge_pilot.py` there, and nothing else). No job submitted.
4. `sbatch run_mc.sh` in `MC_GPT_5.6_Luna/` — **done 2026-10-19, job 4480406, 4m39s:** 27/27 rows
   `ok`, all `finish=stop`, 0 empty, 0 retries, one config; pulled down md5-identical. Then
   `../run_judges.sh` — job 4482838; this benchmark's stage done 2026-10-19, 81/81 rows per judge,
   Sonnet 5.5 refusing 11 (`LLM_as_judge/JUDGE_SELECTION.md` §6.1).
5. `python ../compare_judges.py` on the pulled rows — done; `JUDGE_SELECTION.md` §6.4.

**Full run** (the five selected models, then Haiku 5.5):

1. Local smoke — done 2026-10-31: each model `--limit 3` (3/3 `ok`, `finish=stop`, exit 0, no
   `<think>` text); Haiku 5.5 on each model's smoke rows, `--set full --limit 3` (45/45 `ok`, 0
   parse failures). The pilot's judge files still resume with 0 calls under the changed code.
2. **The user verifies the scripts.** Phase 3 does not start before.
3. Transferred with Wonderbread and the shared files as one set, 95 files md5-identical both sides
   (2026-10-31); `TOGETHER_API_KEY`, `DEEPINFRA_API_KEY`, `DEEPSEEK_API_KEY` added to Quest's `.env`.
4. `sbatch run_mc.sh` in each `MC_<Model>/` — done 2026-11-01, 273/273 `ok` for every model:

   | Model | Job | Wall | Exit | Not `finish=stop` |
   |---|---|---|---|---|
   | GPT-5.6-Luna | 6252750 | 40m51s | 0 | — |
   | Gemini | 6252751 | 16m44s | 0 | 1 `error`: a fragment; job 6648993 retried it, 5/5 cut off — `failed` |
   | Qwen | 6252835 | 30m54s | 2 | 1 `length` |
   | Gemma | 6252837 | 1h45m | 0 | — |
   | Deepseek | 6252838 | 47m51s | 2 | 1 `length` |

   Qwen's `length` (`6765f489…`) is a repetition loop run to the cap. Deepseek's (`6765fc5e…`, a
   six-week schedule of quotes) spent 7,849 of its 8,192 tokens reasoning; Gemma answered that item
   in 4,416, Qwen in 6,607. Both were then **judged as the answers the cap produced** (`usable`
   counts `length`); the user never ruled on it separately — a confirmation is still owed. Gemini's `error` on `6765fc5e…` was a 19 KB
   fragment recorded `ok` ([provider-gotchas.md](../../provider-gotchas.md) § OpenRouter): Google's
   `RECITATION` filter, deterministic, so the regenerated row is `failed` and `mc_judge` refuses the
   set. **The user decided on 2026-11-07 that it is not scored** ("这个题目所有的benchmark都不记录评分"): `mc_judge.py --skip-qid 6765fc5e0af93a24e8fbe4ee` leaves it out, accepts only an item that has no usable response, and writes `skipped_qids` into the rows' config. Gemini is judged on 272 items. **All five are scored on the same 272** — read from that sentence as "for every model", and recorded so 2026-11-11; that is Deepseek's `length` item too, so Qwen's `6765f489…` is the one capped answer left in. All five files and their
   logs pulled down md5-identical 2026-11-04.
5. `sbatch --array=<i> run_judge_full.sh` from `Action_processes_benchmarks/`, per model — done, all
   `ok`: GPT 7177159 (index 0), Gemma 7221201 (3), Qwen 7470998_2, Deepseek 7470998_4; Gemini's was
   judged on its own with `--skip-qid` (`run_judge_full.sh` does not pass it; 7470998_1's log holds
   only Wonderbread). Rows per model: 819, Gemini 816.

## Results — Haiku 5.5, recorded 2026-11-11

The user asked for the record on 2026-11-11, **before an audit**; recomputed from the judge rows on
disk by `Results/record_haiku_judge.py` and kept in `Final_result.xlsx` / `Tempo_results.xlsx` ›
`MultiChallenge`, with `results.db` rebuilt from them. Fractions; three replicates per item, an item's
label the strict majority. **All five on the same 272 items**; the macro mean of the four axes is
upstream's headline, the micro (all items as one pool) is stored beside it and never averaged in.

| | Gemini | OpenAI | Qwen | Gemma | Deepseek |
|---|--:|--:|--:|--:|--:|
| INFERENCE_MEMORY (113) | 0.5044 | 0.3982 | 0.3982 | 0.5575 | 0.5664 |
| INSTRUCTION_RETENTION (69) | 0.6957 | 0.6667 | 0.2754 | 0.6667 | 0.7536 |
| RELIABLE_VERSION_EDITING (41) | 0.2439 | 0.3415 | 0.1951 | 0.3659 | 0.4878 |
| SELF_COHERENCE (49 scored of 50) | 0.3673 | 0.3673 | 0.2653 | 0.3673 | 0.5102 |
| **Overall (macro)** | 0.4528 | 0.4434 | 0.2835 | 0.4893 | **0.5795** |
| Overall (micro) | 0.4890 | 0.4522 | 0.3125 | 0.5221 | 0.5919 |

Haiku 5.5's error rate against humans is **unmeasured** (`JUDGE_SELECTION.md` §8), so these are the
judge's grades. Qwen's column is the Together route; Deepseek's is `deepseek-flash` under a prompt
ceiling, not upstream's no-system-prompt conversation.

## What its README says about showing reasoning

Nothing. The tested model gets the stored conversation verbatim, no system prompt, so nothing is
added to ask for or suppress visible reasoning. Hidden reasoning is capped per model — each runner's
docstring; for Deepseek, which has no knob, by a prompt ceiling (its trap below).

## Its own traps

- **The vendored harness does not run** (`TypeError` before the first call) — `JUDGE_SELECTION.md` §7.2 D1. Our
  runners do not use `main.py`.
- **Deepseek's conversation is not upstream's.** It has no thinking knob, so a system message asks it
  to reason briefly (`model-parameters.md` rule 5), ahead of a conversation upstream sends with no
  system prompt — the user's decision, 2026-10-31, and on every row's config. It is a request: one
  smoke item still spent 3,143 reasoning tokens. Read it from `reasoning_tokens`, not from answer length.
- **An unpinned OpenRouter model is several models.** In the first smoke run Kimi K2.6 flipped one
  verdict across three replicates at `temperature=0` — while its calls were landing on two backends
  with different quantizations. Pinned to one (bf16), it gave three identical answers in three.
  Stability is still measured per judge from the replicates, not assumed from the setting.
