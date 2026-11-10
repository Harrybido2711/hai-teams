# Wonderbread — benchmark card

<!-- size-budget: 13000 -->
<!-- One job: Wonderbread's operating facts. Larger than the default because its data has a fetch
     path of its own, its judged scope covers two subtasks, and since 2026-10-31 it carries the
     judge pilot and the full run side by side, with the per-model image caps the full run needs
     and, since 2026-11-01, the full run's Quest record. -->

Monitoring progress toward goals. Upstream `HazyResearch/wonderbread`. **LLM-judged.** The **judge
pilot** (2026-10-19) graded `gpt-5.6-luna` on a 10% sample of the two judged subtasks with four
candidate judges, plus upstream's human-labelled samples; the user chose **Claude Haiku 5.5** on
2026-10-31. The **full run** has four selected models — GPT, Gemini, Qwen, Gemma — answer every
item, and Haiku 5.5 grade each; Deepseek is text-only and does not run here (the user's decision,
2026-10-31). Generation submitted on Quest 2026-11-01 (run order below); no recorded result.

## Paths

| | Path |
|---|---|
| Local | `Action_processes_benchmarks/Wonderbread_bench` |
| Quest | `/projects/p32983/Action_processes_benchmarks/Wonderbread_bench` — created 2026-10-19 by the pilot's first transfer (the whole vendored tree plus our files, md5-verified); `.env` copied there from our NegotiationToM folder on Quest; `data/demos/` is fetched there by `fetch_sample_demos.py`, never transferred |
| Shared | `Action_processes_benchmarks/` itself (`judge_core.py` and the judge scripts) — md5 it too on every sync; [multichallenge.md](multichallenge.md) § Paths |

## Layout

```
Wonderbread_bench/
├── wonderbread/              upstream package — helpers.py, the task folders, benchmark/webarena/*.json
├── data/                     qa_dataset.csv, df_rankings.csv; experimental_results/ is upstream's
│   └── demos/                OURS, gitignored: the demos a set needs, from fetch_sample_demos.py
├── wb_eval_core.py           ours: sets, anchors, image policy, the generation loop (run_cli), upstream imported
├── fetch_sample_demos.py     byte-range extraction from Zenodo; --set sample (default) | full
├── sample_uids.json          the pilot sample, drawn once (seed 20261019), committed
├── WB_GPT_5.6_Luna/          gpt56luna_wb_eval.py --task qa|sop, run_wb.sh — one folder per model,
├── WB_Gemini_Flash3.5lite_OpenRouter/   each the client and parameters only (a judge_core.Generator):
├── WB_Qwen/ · WB_Gemma_DeepInfra/       gemini35lite_ / qwen_ / gemma_di_wb_eval.py + run_wb.sh
├── WB_Qwen_DeepInfra/        qwen_di_wb_eval.py — Qwen's rerun at a 30-image cap (2026-11-09); WB_Qwen/ is the Together record
└── WB_Judges/                wb_judge.py [--judge <key>] --task qa|sop --set sample|anchor|full [--producer WB_<Model>]
```

Shared with MultiChallenge in `Action_processes_benchmarks/`: `judge_core.py`, `run_judges.sh`
(pilot), `run_judge_full.sh` (full run), `compare_judges.py`, `test_judge_pilot.py`.

## What the judge actually covers — larger than "QA"

Recorded in `LLM_as_judge/JUDGE_SELECTION.md` §7.3–7.5, and found by reading code rather than the paper:

- **SOP Generation is judged too.** Its "semantic" Precision/Recall are tallies of GPT-4
  line-entailment decisions, on a **different GPT-4 snapshot** than QA uses. The call budget
  therefore scales with SOP length, not with item count.
- **The SOP-Improvement rubric scorer does not execute** as vendored — it fails on four independent
  counts.
- **The same 1–5 rubric ships with both polarities**: "1 (best) to 5 (worst)" in one file, "1 (worse)
  to 5 (best)" in another. Which direction a number uses cannot be inferred from the number.

## The data — 33 GB upstream, 2.7 GB read

`gold_demos.zip` is 33 GB because of the videos. Neither judged subtask reads a video: both read
each demo's SOP, trace `.json` and screenshots, which Zenodo also ships as
`gold_demos_sop_and_trace_and_screenshots_only.zip` (5.7 GB). Zenodo serves byte ranges, so
`fetch_sample_demos.py` reads that zip's central directory and only the members a set needs,
CRC-checked: the pilot's sample **31 demos, 383 files, 291 MB**; the full set (`--set full`) **255
demos, 3,522 files, 2.7 GB** — fetched locally 2026-10-31, every needed demo present.

## Images — downscaled for every model, and two per-request caps

Upstream's screenshots are 3456×2234-class PNGs. At that size three of the four providers refuse or
hang on the larger requests ([provider-gotchas.md](../../provider-gotchas.md) § Image limits).
**Every model gets every screenshot at a 1188-px long side as JPEG q85** (`wb_eval_core.IMAGE_POLICY`,
written on every row's config) — the user's decision, 2026-10-31, a declared deviation from
upstream. Largest full-set request after it: QA 55 images / 7.8 MB, SOP 71 / 9.6 MB.

Two models still have an image-count cap, `Generator.max_images`. A request over it is never sent:
the unit is written `over_image_limit`, is not judged, and does not fail the run. **That model is
reported on the units within its cap**, and every row carries `n_images`, so every model can also be
scored on the same subset for a like-for-like comparison (the user's rule for Qwen, 2026-10-31,
applied to Gemma the same way):

| Model | Cap | QA over / 120 | SOP over / 486 (td_kf + td_kf_act) |
|---|---|---|---|
| Qwen3.5-9B, Together — the run judged so far | 10 | 36 | 102 (51 demos × 2) |
| Qwen3.5-9B, DeepInfra — the rerun, `WB_Qwen_DeepInfra/` | 30 | 7 | 4 (2 demos × 2) |
| Gemma 4 31B, DeepInfra | 32 | 5 | 4 (2 demos × 2) |
| GPT-5.6-Luna, Gemini 3.5 Flash-Lite | none hit | 0 | 0 |

The counts are from the full-set dry run (`--dry-run`, 2026-10-31). The `td` ablation and the 54
SOP-evidence QA items send no images.

## Expected counts

QA 120 items, 11 question templates (one of 20, ten of 10), 126 distinct demos — all inside the 724
gold demos. SOP Generation runs on the 162 rank-1 demos of `df_rankings.csv` (gitlab 46,
shopping_admin 49, shopping 44, reddit 23) under three ablations: `td`, `td_kf`, `td_kf_act`.
**Pilot sample:** QA 12 (one per template, two from the 20); SOP 16 demos (5 / 5 / 4 / 2) × 3
ablations = 48 SOPs. **Full run, per model:** 120 QA rows and 486 SOP rows, `over_image_limit`
included (table above). **Anchor sets:** 30 QA answers with human 1–3 scores and 30 SOP pairs with
human precision/recall, both upstream's, under `data/experimental_results/*/human_correlation/`.
Judge calls: QA 4 per answer per replicate; SOP one per line of each SOP in both directions, × 3
replicates. Counted from the full run's SOPs with `sop_lines` (2026-11-04, the method reproduces the
pilot's 3,162): SOP GPT 33,840 · Gemma 27,786 · Gemini 26,241 · Qwen 22,338; QA 1,440 · 1,380 ·
1,440 · 1,008. At Haiku 5.5's pilot rates (~$0.00018 and ~2.1 s per SOP call, plus the 0.5 s sleep)
that is ≈ $4–6.5 and 17–25 h per model, one stream each.

## Scoring

QA: four 1–3 scores, **1 = best**; completeness and soundness see the Human Label, clarity and
compactness do not; an answer under 5 characters is not judged (`skipped`, upstream's "NA"). SOP:
upstream's `PairwiseComparison` precision / recall / ordering, F1 as `run_sop_generation.py`, fed our
parsed indices — **one unreadable line drops its SOP pair from that replicate**, which is why the
index parser reads the final `"index"` of a judge answer whose scratchpad broke the JSON. Every
deviation from upstream is declared in the docstring of `wb_eval_core.py`.

## Output and logs

Ours, named after the model that wrote them:

- pilot: `WB_GPT_5.6_Luna/results/{qa,sop}/gpt-5.6-luna.jsonl`;
  `WB_Judges/results/<judge>/{qa,sop}/{gpt-5.6-luna,anchor-human-sample}.jsonl`;
- full run: `WB_<Model>/results/full/{qa,sop}/<model id, "/" → "-">.jsonl`;
  `WB_Judges/results/full/haiku-5.5/{qa,sop}/<same name>.jsonl`. A judge refuses rows of another set.

The SOP text files upstream's `run()` writes go to `results/[full/]sop_upstream_files/`, gitignored
(the `.jsonl` holds the same text). `--limit` writes `smoke/`, gitignored. SLURM logs
`log_<jobid>.txt`; the full-run judge array `../judge_full_log_<job>_<task>.txt`.

## Run order

**Pilot** (2026-10-19, done). Demos fetched locally and on Quest (md5-identical, 383 files); local
smoke; the user verified; code on Quest with MultiChallenge's (`md5sum -c` exit 0). `run_wb.sh` in
`WB_GPT_5.6_Luna/`, job 4480407, 8m13s: QA 12/12 and SOP 48/48 `ok`, 0 empty, 0 retries; the SOPs
came to 1,054 lines, 3,162 judge calls per judge. `../run_judges.sh`, job 4482838, **halted on
OpenRouter billing** 13:34–13:56 with every SOP sample unfinished; resumed as
`sbatch --array=0-2 ../run_judges.sh`, job 4506836 (15:18–19:06), every set complete, pulled down
md5-identical. `python ../compare_judges.py` — `JUDGE_SELECTION.md` §6.4.

**Full run:**

1. Local smoke — done 2026-10-31. `--set full` dry run for every request. Each model QA
   `--limit 7` and SOP `--limit 1` (40/40 `ok`; Qwen and Gemma at a 10-image request). The largest
   requests probed on Gemini (55 and 71 images, ok) and Gemma (32, ok; 55 and 71 refused — its cap).
   Haiku 5.5 on each model's smoke rows: 96/96 QA `ok`; SOP 7 parse failures in 720, all
   recovered by the parser fix (the same gold line, quoted text copied unescaped).
2. **The user verifies the scripts.** Phase 3 does not start before.
3. On the Quest login node: `python fetch_sample_demos.py --set full`, then `--set full --check`
   — done 2026-11-01: the 224 demos the sample lacked, 2,431 MB in 1,937 range requests; `--check`
   "all 255 needed demos present". Transferred with MultiChallenge and the shared files as one
   set, 95 files md5-identical both sides (2026-10-31); `TOGETHER_API_KEY` and `DEEPINFRA_API_KEY`
   added to Quest's `.env` here.
4. `sbatch run_wb.sh` in each `WB_<Model>/` (QA then SOP, one job, `normal` 12 h) — done
   2026-11-01 on the interrupted-finish guard (multichallenge.md § Run order). GPT 6648994 (1h10m),
   Gemini 6648995 (42m), Gemma 6648997 (1h56m): exit 0, every unit `ok` or over its cap exactly as
   the dry run counted. Qwen 6648996 (48m): exit 1, SOP exit 2 on 3 rows at `length` — the user's
   call, with MultiChallenge's. 0 empty rows. Pulled down md5-identical 2026-11-04.
5. `sbatch --array=<i> run_judge_full.sh` from `Action_processes_benchmarks/`: GPT (index 0) job
   7177159, 2026-11-04. The rest wait on the OpenRouter balance and the user's decisions.
6. **Qwen's rerun on DeepInfra** (the user's decision, 2026-11-09, to lift the cap from 10 to 30 and
   keep every unit on one route): `sbatch run_wb.sh` in `WB_Qwen_DeepInfra/` (24 h — the route ran
   at 17–111 s a call, so ~9–10 h; per-call ceiling 600 s), then **archive** the Together judge
   files — `WB_Judges/{results,smoke}/full/haiku-5.5/{qa,sop}/Qwen-Qwen3.5-9B.jsonl`, moved to
   `WB_Judges/archive/qwen_together_2026-11/` with their paths kept, never renamed in place — since
   both Qwen producers write that name and `load_done` refuses to resume across them. Then
   `sbatch --array=5 run_judge_full.sh` — Wonderbread only, ~18 h; **index 2 no longer judges
   Wonderbread**, so the two can never write one file. Estimate ≈ $5.3, almost all of it the judge.
   Then point `Results/build_judge_summary.py`'s Qwen row at `WB_Qwen_DeepInfra`; until then it
   reads `WB_Qwen`, and its `same_producer` check refuses judge rows from the other folder.

## What its README says about showing reasoning

Nothing. Upstream's prompts ask for a 2–3-sentence answer (QA) or an SOP; nothing is added. Hidden
reasoning is capped per model — each runner's docstring.

## Its own traps

- **`wonderbread/helpers.py` imports four client libraries at module level** that we never call;
  Quest lacks anthropic and pydrive2. `wb_eval_core.upstream()` substitutes stand-ins that raise if
  touched — do not "fix" it by installing them.
- **Sonnet 5.5 refuses SOP Generation's line-matching prompt** — content filter, HTTP 200,
  `content` null. `LLM_as_judge/JUDGE_SELECTION.md` §6.1.
- **One anchor pair's text is not the one upstream scored.** `anchor20`'s stored generated SOP has
  10 steps; upstream recorded 11 lines for it. Its human labels are kept; the test pins the gap.
- **`preprocess_sop` strips everything up to the first "."** of each line (`JUDGE_SELECTION.md` §7.4, B4) — kept for
  comparability, so a step like "Click Save. Then confirm." is matched as "Then confirm."
- **The pilot's rows ran at full-size screenshots; the full run's do not.** GPT's `--set sample`
  refuses to resume the pilot files rather than mix the two.
