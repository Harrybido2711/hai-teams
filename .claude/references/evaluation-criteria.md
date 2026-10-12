<!-- size-budget: 6000 -->
<!-- One job: the criteria, the per-row fields that feed them, and their traps. -->
# Evaluation criteria beyond accuracy

Six criteria every model is judged on alongside its accuracy, on every benchmark. Adopted by the
user 2026-10-08. They exist because accuracy alone cannot separate two models that score alike but
differ tenfold in cost, or one that is right on average but gives a different answer each time it
is asked.

**State on 2026-10-12: three runners, none yet run on Quest.** `EMO_GPT_5.6_Luna` writes every
field below in its own runner. NegotiationToM's core writes them for any runner that calls
`enable_row_fields()` — today `NEG_GPT_5.6_Luna` and `NEG_Gemini_Flash3.5lite_OpenRouter`; the
older NEG runners do not opt in, because they leave the SDK's own retries on and their `n_attempts`
would undercount. No other runner writes them. Rows already on disk therefore **cannot** produce
criteria 1–3 per item. A run made before a runner was updated says so; it does not get an estimate.

## The six

| # | Criterion | Metric | Computed as | Needs |
|---|---|---|---|---|
| 1 | **Cost** | $ per correct answer | Σ(`prompt_tokens`·p_in + `completion_tokens`·p_out) / 1e6 ÷ #correct | the normal run |
| 2 | **Token efficiency** | mean and p95 output tokens per item, reasoning included | over `completion_tokens`; also the mean on wrong items ÷ the mean on right ones (overthinking) | the normal run |
| 3 | **Latency** | p50 / p95 seconds per call | over `latency_s` | the normal run |
| 4 | **Stability** | share of items whose answer is identical across k = 5 repeats | per item, then averaged; report pass^k beside pass@1 | extra runs, subsample |
| 5 | **Robustness** | flip rate — share of items whose answer changes under a perturbation | paraphrased prompt, shuffled option order | extra runs, subsample |
| 6 | **Reliability** | parse-success rate; empty, error and truncation rates | from `parse_ok`, `finish_reason`, `n_attempts` | the normal run |

## Per-row fields

**Every runner persists these on every row**, next to the raw response. A run-level sum cannot give
a per-correct cost or a per-item comparison, and a field not written during the run cannot be
recovered afterwards.

| Field | Source |
|---|---|
| `prompt_tokens`, `completion_tokens` | the provider's usage object, through the shared core's `usage_from()` — it already folds each SDK's naming, and Gemini's separate `thoughts_token_count`, into one count |
| `reasoning_tokens` | the provider's reasoning breakdown where it reports one; empty, not 0, where it does not |
| `latency_s` | the successful call alone, back-off excluded |
| `finish_reason` | as returned; `length` is a truncation |
| `n_attempts` | calls spent on this row, retries included |
| `parse_ok` | whether the parser produced a prediction |

## Traps

- **Never count tokens from `raw_response`.** Hidden reasoning tokens are billed and are not in it,
  so a tokenizer count undercounts a reasoning model's cost several-fold.
- **`n_attempts` and `latency_s` are wrong while the SDK retries by itself** — the values look
  plausible and nothing is logged. Client setting: [script-skeleton.md](script-skeleton.md) §4.
- **`parse_ok` needs a non-empty string**, not just a present key: `{"answer": null}` is no prediction.
- **The response's model id is not proof of identity.** `gpt-5.6-luna` echoes the alias in
  `resp.model` and returns no `system_fingerprint`, so neither can show the model behind it changed
  between two halves of one score. Record both anyway; do not cite them as that proof.
- **Token counts do not compare across families** — each uses its own tokenizer. Compare tokens
  within a family; compare across families in dollars and seconds.
- **Throughput comes from `wall_seconds`, never from summed call latencies** — the reason is in the
  comment on `STATS` in `neg_eval_core.py`.
- **Prices are dated.** Record the price per million tokens and the date it was read; reasoning
  tokens are billed as output.
- **Repeats and perturbations use the model's settled config** from
  [model-parameters.md](model-parameters.md). Changing temperature to provoke variance measures a
  different configuration. `temperature=0` is not deterministic on hosted APIs, and several reasoning
  models refuse the parameter outright.
- **Repeats write to their own tagged files** (`_rep{k}`, `_perturb-{name}`). Written into the main
  checkpoint, resume skips every uid already present and the repeat never runs.
- **The subsample is fixed and shared.** About 300 items **per benchmark** (the user's choice,
  2026-10-12 — per task made bbh and MMLU near-full runs ×7), stratified across its tasks, drawn
  once with a recorded seed, committed as a uid list, the same items for every model — the
  comparison is paired. EmoBench uses all 400 items; NegotiationToM about 300 rows per task.
- **A perturbation must not change the answer.** Shuffling options moves the gold *letter*; match on
  option content. Gold labels are still never rewritten.
- **Scored with the benchmark's one lenient matcher**, the same as accuracy
  ([script-skeleton.md](script-skeleton.md) rule 7).

## Using them to choose models

The criteria inform the user's choice of selected models; they do not make it. The current
selection, and who decided it, is in `PLAN.md`.

1. **Gate on reliability (6).** A model that fails it has an accuracy nobody can believe.
2. **Accuracy against cost (1).** Normalise accuracy within each benchmark (rank, or ÷ the best
   score) before averaging, so no benchmark dominates by scale. A model that another beats on both
   accuracy and cost is dominated.
3. **Separate the rest on 2–5.**

A difference counts only if it survives a paired test on the same items — paired bootstrap, or
McNemar for right/wrong. Two models whose intervals overlap are not ranked.

## Stability and robustness — the design (user's decisions, 2026-10-12)

- **Config:** each benchmark's repeats use exactly the config that produced the number reported on
  its sheet, so stability describes that number. Qwen, Gemma and Deepseek have no single config
  across benchmarks; this sidesteps defining one.
- **Runs per model per benchmark:** `rep1`…`rep5` unperturbed, plus `perturb-shuffle` and
  `perturb-paraphrase`, each to its own tagged output on the shared subsample.
- **Stability (4):** share of items whose answer *content* is identical in all five; pass@1 (mean
  accuracy over reps), pass^5 (correct in all five), and the min–max accuracy spread.
- **Robustness (5):** flip rate of each perturbation against `rep1`, **minus the rep1-vs-rep2 flip
  rate** — a model that varies on its own would otherwise be charged for that variance as
  fragility. Paired test for both the excess flips and the accuracy change.
- **What gets perturbed** is only text this project owns: the instruction wrapper, never upstream
  item text; answer-format instructions stay verbatim so a flip is not a parse failure. Where a
  benchmark has no options (DocVQA) or no instruction of ours (AwareBench), that perturbation is
  not defined. AwareBench's data already holds 2–3 option orders per MCQ question, so its shuffle
  flip rate comes from the normal run at no extra cost.
- **Phase 1:** EmoBench and NegotiationToM, `gpt-5.6-luna` and `gemini-3.5-flash-lite`. Phase 2:
  the other three models there, then MMLU, bbh and DocVQA, whose Qwen/Gemma/Deepseek runners are
  legacy scripts with no tag, no subsample and no per-row fields.
- **Recorded in** a new `Criteria` sheet in both workbooks — rows benchmark × model, columns the six
  criteria — under the same rule as accuracy: after the user confirms the run.

**Not yet decided — ask before acting:** the gate thresholds and the weights for any composite score.
