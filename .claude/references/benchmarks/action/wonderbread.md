# Wonderbread — benchmark card

<!-- size-budget: 7000 -->
<!-- One job: Wonderbread's operating facts. Larger than the default because its data has a fetch
     path of its own (the 33 GB question) and its judged scope covers two subtasks. -->

Monitoring progress toward goals. Upstream `HazyResearch/wonderbread`. **LLM-judged.** Our runners
exist for the **judge pilot** only (2026-10-19): `gpt-5.6-luna` on a 10% sample of the two judged
subtasks, graded by four candidate judges, plus upstream's human-labelled samples graded by the same
four. No full run, no recorded result.

## Paths

| | Path |
|---|---|
| Local | `Action_processes_benchmarks/Wonderbread_bench` |
| Quest | `/projects/p32983/Action_processes_benchmarks/Wonderbread_bench` — created 2026-10-19 by the pilot's first transfer (the whole vendored tree plus our files, md5-verified); `.env` copied there from our NegotiationToM folder on Quest; `data/demos/` is fetched there by `fetch_sample_demos.py`, never transferred |

## Layout

```
Wonderbread_bench/
├── wonderbread/              upstream package — helpers.py, the task folders, benchmark/webarena/*.json
├── data/                     qa_dataset.csv, df_rankings.csv; experimental_results/ is upstream's
│   └── demos/                OURS, gitignored: only the sampled demos, from fetch_sample_demos.py
├── wb_eval_core.py           ours: samples, anchor sets, upstream builders/prompts/scorer imported
├── fetch_sample_demos.py     byte-range extraction of the sampled demos from Zenodo
├── sample_uids.json          the pilot sample, drawn once (seed 20261019), committed
├── WB_GPT_5.6_Luna/          generation: gpt56luna_wb_eval.py --task qa|sop, run_wb.sh
└── WB_Judges/                wb_judge.py --judge <key> --task qa|sop --set sample|anchor
```

Shared with MultiChallenge in `Action_processes_benchmarks/`: `judge_core.py`, `run_judges.sh`,
`compare_judges.py`, `test_judge_pilot.py`.

## What the judge actually covers — larger than "QA"

Recorded in `LLM_as_judge/JUDGE_SELECTION.md` §7.3–7.5, and found by reading code rather than the paper:

- **SOP Generation is judged too.** Its "semantic" Precision/Recall are tallies of GPT-4
  line-entailment decisions, on a **different GPT-4 snapshot** than QA uses. The call budget
  therefore scales with SOP length, not with item count.
- **The SOP-Improvement rubric scorer does not execute** as vendored — it fails on four independent
  counts.
- **The same 1–5 rubric ships with both polarities**: "1 (best) to 5 (worst)" in one file, "1 (worse)
  to 5 (best)" in another. Which direction a number uses cannot be inferred from the number.

## The data — the 33 GB question, for the pilot

`gold_demos.zip` is 33 GB because of the videos. Neither judged subtask reads a video: both read
each demo's SOP, trace `.json` and screenshots, which Zenodo also ships as
`gold_demos_sop_and_trace_and_screenshots_only.zip` (5.7 GB). Zenodo serves byte ranges, so
`fetch_sample_demos.py` reads that zip's central directory and only the sampled members: **31
demos, 383 files, 291 MB**, CRC-checked. Whether a full run is in scope is still the advisor's open
question; the pilot does not settle it.

## Expected counts

QA 120 items, 11 question templates (one of 20, ten of 10), 126 distinct demos — all inside the 724
gold demos. SOP Generation runs on the 162 rank-1 demos of `df_rankings.csv` (gitlab 46,
shopping_admin 49, shopping 44, reddit 23) under three ablations: `td`, `td_kf`, `td_kf_act`.
**Pilot sample:** QA 12 (one per template, two from the 20); SOP 16 demos (5 / 5 / 4 / 2) × 3
ablations = 48 SOPs. **Anchor sets:** 30 QA answers with human 1–3 scores and 30 SOP pairs with
human precision/recall, both upstream's, under `data/experimental_results/*/human_correlation/`.
Judge calls: QA 4 per answer per replicate; SOP one per line of each SOP in both directions.

## Scoring

QA: four 1–3 scores, **1 = best**; completeness and soundness see the Human Label, clarity and
compactness do not; an answer under 5 characters is not judged (`skipped`, upstream's "NA"). SOP:
upstream's `PairwiseComparison` precision / recall / ordering, F1 as `run_sop_generation.py`, fed our
parsed indices. Every deviation from upstream is declared in the docstring of `wb_eval_core.py`.

## Output and logs

Ours: `WB_GPT_5.6_Luna/results/{qa,sop}/gpt-5.6-luna.jsonl`; `WB_Judges/results/<judge>/{qa,sop}/
{gpt-5.6-luna,anchor-human-sample}.jsonl` — named after the model that wrote them. The SOP text
files upstream's `run()` writes go to `results/sop_upstream_files/`, gitignored (the `.jsonl` holds
the same text). `--limit` writes `smoke/`, gitignored. SLURM logs `log_<jobid>.txt`.

## Run order

1. `python fetch_sample_demos.py` (local and on the Quest login node; `--check` verifies).
2. Local smoke — done 2026-10-19: `--dry-run` for every request; generation `--limit 6` (24 calls,
   all ok, largest request 35.6 MB / 30 screenshots); after the review fixes, every judge on every
   set — 0 failed, 0 parse failures; Sonnet 5.5 refused every SOP call and exited 1, as designed.
   Demos fetched on Quest too, md5-identical to the local 383 files.
3. **The user verifies the scripts.** Phase 3 does not start before. The reviewed code is on Quest
   (transferred with MultiChallenge's, `md5sum -c` exit 0); no job submitted.
4. `sbatch run_wb.sh` in `WB_GPT_5.6_Luna/` (QA then SOP, one job) — **done 2026-10-19, job
   4480407, 8m13s:** QA 12/12 and SOP 48/48 `ok`, all `finish=stop`, 0 empty, 0 retries; pulled
   down md5-identical. The SOPs come to 1,054 lines over both directions — 3,162 judge calls per
   judge at 3 replicates. Then `../run_judges.sh` — not yet submitted.
5. `python ../compare_judges.py` on the pulled rows.

## What its README says about showing reasoning

Nothing. Upstream's prompts ask for a 2–3-sentence answer (QA) or an SOP; nothing is added. Hidden
reasoning is capped at the settled `reasoning_effort="low"`.

## Its own traps

- **`wonderbread/helpers.py` imports four client libraries at module level** that the pilot never
  calls; Quest lacks anthropic and pydrive2. `wb_eval_core.upstream()` substitutes stand-ins that
  raise if touched — do not "fix" it by installing them.
- **Sonnet 5.5 refuses SOP Generation's line-matching prompt** — content filter, HTTP 200,
  `content` null. `LLM_as_judge/JUDGE_SELECTION.md` §6.1.
- **One anchor pair's text is not the one upstream scored.** `anchor20`'s stored generated SOP has
  10 steps; upstream recorded 11 lines for it. Its human labels are kept; the test pins the gap.
- **`preprocess_sop` strips everything up to the first "."** of each line (`JUDGE_SELECTION.md` §7.4, B4) — kept for
  comparability, so a step like "Click Save. Then confirm." is matched as "Then confirm."
