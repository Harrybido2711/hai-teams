<!-- size-budget: 7500 -->
<!-- One job: how criteria 4 and 5 are run and analysed. What the criteria are is evaluation-criteria.md. -->
# Criteria 4 and 5 — running and analysing repeats and perturbations

Stability (4) and robustness (5) from [evaluation-criteria.md](evaluation-criteria.md) need runs of
their own. This is how they are designed, run and turned into numbers.

## The design (user's decisions, 2026-10-12)

- **Config:** each benchmark's repeats use exactly the config that produced the number reported on
  its sheet, so stability describes that number. Qwen, Gemma and Deepseek have no single config
  across benchmarks; this sidesteps defining one.
- **Runs per model per benchmark:** `rep1`…`rep5` unperturbed, plus `perturb-shuffle` and
  `perturb-paraphrase`, each to its own tagged output on the shared subsample.
- **Stability (4):** share of items whose answer *content* is identical in all five; pass@1 (mean
  accuracy over reps), pass^5 (correct in all five), and the min–max accuracy spread. The seed is
  pinned in every rep, so this measures how reproducible the config is *with its seed fixed*.
- **Robustness (5):** flip rate of each perturbation against `rep1`, **minus the rep1-vs-rep2 flip
  rate** — a model that varies on its own would otherwise be charged for that variance as
  fragility. Paired test for both the excess flips and the accuracy change.
- **What gets perturbed** is the instruction wrapper, never item text, and answer-format
  instructions stay verbatim so a flip is not a parse failure. A wrapper counts whoever wrote it, as
  long as it is separate from every item: NegotiationToM's is ours, EmoBench's is upstream's `sys.en`
  (the user's ruling, 2026-10-12). Where a benchmark has no options (DocVQA) or no
  wrapper apart from item text (AwareBench), that perturbation is not defined. AwareBench's data
  already holds 2–3 option orders per MCQ question, so its shuffle flip rate comes from the normal
  run at no extra cost.
- **Phase 1:** EmoBench and NegotiationToM, `gpt-5.6-luna` and `gemini-3.5-flash-lite`. Phase 2:
  the other three models there, then MMLU, bbh and DocVQA, whose Qwen/Gemma/Deepseek runners are
  legacy scripts with no tag, no subsample and no per-row fields.
- **Recorded in** `Criteria_results.xlsx` (the user's decision 2026-10-19, replacing a `Criteria`
  sheet in both workbooks): a sheet per benchmark, a row per selected model, accuracy beside the six
  criteria. An unconfirmed run is shown flagged; it reaches `Final_result.xlsx` once confirmed.

## Running it

| | NegotiationToM | EmoBench | AwareBench |
|---|---|---|---|
| Subsample | `criteria_subsample.json` — first 150 of the pilot's seed-42 order; 300/300/295 rows (294/294/295 scored), 126 distinct negotiations | all 400 English items | `subsample_uids.json` — seed 20261012, 30 rows per task, 300; whole questions for the permuted tasks |
| Flags | `--subsample criteria_subsample.json --tag rep<N>` or `--tag perturb-X --perturb X` | `--tag rep<N>` or `--tag perturb-X --perturb X` | `--rep K` (1–5); no perturbation runs — shuffle comes from the full run's own orderings, paraphrase is undefined |
| Output | `NEG_<Model>/results/<tag>/<task>/<slug>.jsonl`, `run_config.json`; params `results/negotiated_params_<tag>.json` | `EMO_<Model>/results_<tag>/<EU\|EA>/`, `run_config.json` | `AWARE_<Model>/results_rep<K>/<task>/<slug>.jsonl`; params `results_rep<K>/<slug>_negotiated_params.json`, refused before the first call unless equal to the full run's |
| Perturbation code | `neg_eval_core.py` (`PERTURB`, `PARAPHRASE`) | `criteria_perturb.py`, imported by each runner | — |
| Job | `sbatch run_criteria.sh` in the model folder: array 0–6 = rep1–5, shuffle, paraphrase | same | `sbatch run_aware_reps.sh`: array 1–5 |

- **Guards.** NegotiationToM tags are an allow-list (`rep<N>`, `perturb-shuffle`,
  `perturb-paraphrase`, `smoke-<name>`) and need `--subsample`; the command line, including the
  subsample's ids, is validated in `parse_cli` before the probe call, so a rejected command spends
  nothing and writes nothing. A tag resumes only under the configuration its `run_config.json`
  recorded (model, perturbation, seed, subsample md5, and a fingerprint of the prompts it renders);
  otherwise archive it. A tag holding rows but no `run_config.json` is refused for the same reason,
  and a tagged run is never sharded. NegotiationToM checks all of this in `parse_cli`, before the
  probe; EmoBench after it, since its config includes the negotiated parameters.
- **Halt markers carry the tag in front** (`rep1_BILLING_HALT.txt`), so a criteria run never clears
  or fakes the main run's marker in the same folder, and every marker still matches `*_HALT.txt`.
- **Sequence:** a model's criteria array goes after its main run in that folder has finished — the
  same key's rate limit — and only once its config is settled: a model with no pilot yet has no
  config the reported number will use.
- **Sync:** the pre-submit hook does not cover `*.json` or anything in EmoBench; md5 those by hand.

## Analysing it

`python3 .claude/scripts/criteria_report.py neg|emo <model folder>` applies every rule below and
writes `<model folder>/criteria_summary.json` — 1, 2, 3, 6 from the main run (`main_run`), 4 and 5
(`tasks`) once all seven tags exist; it refuses to compute when the seven tags differ. `neg` needs
pandas and scikit-learn. AwareBench has its own, `Awareness_in_LLM/aware_criteria.py` →
`criteria_awareness.csv`, under the same rules and taking 1, 2, 3, 6 from `criteria_report.py`; its
excess flips pair each question's orderings ≥1 against ordering 0, clustered by question.

`build_criteria_workbook.py` rebuilds `Criteria_results.xlsx` from those files, every blank with its
reason; never edit it by hand. It never writes `Tempo_results.xlsx` or `Final_result.xlsx`.

**A model that honours the seed has a noise floor of 0.** `gemini-3.5-flash-lite` on NegotiationToM
returned byte-identical responses in all five repeats (separate calls), so every flip under a
perturbation counts as excess; `gpt-5.6-luna`'s repeats vary with the seed pinned and its noise
floor absorbs part of its flips. Compare raw flips beside the excess before ranking the two.

- **Pair rows by uid** (NegotiationToM) or by (task, qid) (EmoBench). Before computing anything,
  assert all seven tags hold the same key set and the counts above. Nothing in the runners asserts it.
- **Compare content, never letters.** NegotiationToM desire/belief: the tuple of
  `core.pred_item(pred, k)` for high, medium, low — reuse the core function, `pred` is the raw dict.
  Intention: `pred_bitmask`, not `pred_intents`, whose order follows the presented label order.
  EmoBench: `answer_text`, and for EU the pair (`emo_answer_text`, `cause_answer_text`).
- **A failure is not an answer.** Empty `raw_response`, `pred` None or a None `*_answer_text` is a
  failed row; None == None never counts as stable.
- **Denominators** come from `scorable()` for NegotiationToM; correctness from the row's own
  `desire_em` / `belief_em` / `score`.
- **Cluster paired tests by negotiation** (`dialogue_id.split('-')[0]`): up to three cutoffs of
  one negotiation are in the subsample, and they are not independent.
- **Intention's per-row "correct"** — its reported metric is F1 over the set, so pass^5 needs one:
  the predicted label set equals the gold set exactly (bitmask match), the user's choice 2026-10-12.
