# MultiChallenge — benchmark card

Coordination, over multi-turn conversations. Upstream `ekwinox117/multi-challenge`. **LLM-judged, on
every item.** Our runners exist for the **judge pilot** only (2026-10-19): `gpt-5.6-luna` on a 10%
sample, graded by four candidate judges. No full run, no recorded result.

## Paths

| | Path |
|---|---|
| Local | `Action_processes_benchmarks/Multi-challenge_bench` |
| Quest | `/projects/p32983/Action_processes_benchmarks/Multi-challenge_bench` — created 2026-10-19 by the pilot's first transfer (the whole vendored tree plus our files, md5-verified); `.env` copied there from our NegotiationToM folder on Quest |

## Layout

```
Multi-challenge_bench/
├── main.py · src/                upstream harness — src/evaluator.py holds JUDGE_PROMPT
├── data/benchmark_questions.jsonl  the 273 items; data/final_model_responses/ is upstream's
├── mc_eval_core.py               ours: sample, verbatim prompt + schema, scorer (imports ../judge_core.py)
├── sample_uids.json              the pilot sample, drawn once (seed 20261019), committed
├── MC_GPT_5.6_Luna/              generation: gpt56luna_mc_eval.py, run_mc.sh
└── MC_Judges/                    mc_judge.py --judge <key>; one file for all four judges
```

The judge configurations, call infrastructure and checkpoint are shared with Wonderbread in
`Action_processes_benchmarks/judge_core.py`; the array job is `../run_judges.sh` and the comparison
`../compare_judges.py`. Pinned checks: `../test_judge_pilot.py`.

## Expected counts

273 items — `INFERENCE_MEMORY` 113, `INSTRUCTION_RETENTION` 69, `SELF_COHERENCE` 50,
`RELIABLE_VERSION_EDITING` 41; `PASS_CRITERIA` is `"YES"` on all 273. `mc_eval_core.load_items`
refuses any other count. **Pilot sample: 27** — 11 / 7 / 5 / 4. Judge rows: 27 × 3 replicates = 81
per judge.

## Scoring

Upstream's, unchanged: an item passes on `verdict == PASS_CRITERIA`; per-axis pass rate; headline
the **macro** mean over the four axes (micro reported beside it). The judge sees the final response
and the `TARGET_QUESTION` rubric only. What our runner repairs, and why, is
`LLM_as_judge/JUDGE_SELECTION.md` §7.2 D1–D2; how it is posed through OpenRouter is the docstring of
`mc_eval_core.py`. A judge row is `ok`, `parse_fail`, `refused` or `failed`; only `ok` carries a
verdict, and a unit's label is the majority of its replicates (a tie is unjudged, never NO).

## Output and logs

Ours: `MC_GPT_5.6_Luna/results/gpt-5.6-luna.jsonl` (one row per item) and
`MC_Judges/results/<judge>/gpt-5.6-luna.jsonl` (one row per item × replicate) — each directory named
after the model that wrote it. `--limit` writes `smoke/` instead, gitignored. Rows are appended and
flushed per call; resume skips finished uids and refuses a different config. SLURM logs
`log_<jobid>.txt` per folder; the judge array logs to `../judge_log_<job>_<task>.txt`.

## Run order

1. Local smoke — done 2026-10-19, after the review fixes: generation `--limit 3` (3 ok,
   `finish=stop`), each judge `--limit 2` (24 calls, all ok, every one on its pinned backend).
2. **The user verifies the scripts.** Phase 3 does not start before.
3. Transfer as one set with Wonderbread and the shared files; `md5sum -c` against a local manifest.
   Done 2026-10-19 for the reviewed code: 1,347 files, exit 0 (Quest also holds 13 `__pycache__`
   files from running `test_judge_pilot.py` there, and nothing else). No job submitted.
4. `sbatch run_mc.sh` in `MC_GPT_5.6_Luna/` — **done 2026-10-19, job 4480406, 4m39s:** 27/27 rows
   `ok`, all `finish=stop`, 0 empty, 0 retries, one config; pulled down md5-identical. Then
   `../run_judges.sh` (after the Wonderbread generation job too) — not yet submitted.
5. `python ../compare_judges.py` on the pulled rows.

## What its README says about showing reasoning

Nothing. The tested model gets the stored conversation verbatim, no system prompt, so nothing is
added to ask for or suppress visible reasoning. Hidden reasoning is capped at the settled
`reasoning_effort="low"`.

## Its own traps

- **The vendored harness does not run** (`TypeError` before the first call) — `JUDGE_SELECTION.md` §7.2 D1. Our
  runners do not use `main.py`.
- **An unpinned OpenRouter model is several models.** In the first smoke run Kimi K2.6 flipped one
  verdict across three replicates at `temperature=0` — while its calls were landing on two backends
  with different quantizations. Pinned to one (bf16), it gave three identical answers in three.
  Stability is still measured per judge from the replicates, not assumed from the setting.
