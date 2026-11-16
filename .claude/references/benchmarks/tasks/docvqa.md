# DocVQA — benchmark card

<!-- size-budget: 10500 -->
<!-- One job — the operational card: paths, layout, counts, results, traps. Method went to
     docvqa-method.md and the matcher to docvqa-scoring.md. Over the 5 KB nudge because six runners
     in two generations share one folder, and the paragraph saying which is which is what stops a
     reader copying the wrong one. Raised to 10500 on 2026-11-15 for the sample path's section: its
     traps are about this folder's files and belong on this card. If the sample section outgrows
     that, split it into docvqa-sample10.md with a row in benchmarks/README.md. -->

Document visual question answering. Upstream docvqa.org. Scored by **ANLS**, no LLM judge.

**Four of the six have results, and four is the ceiling** — exactly the models that accept an
image ([model-calls.md](../../model-calls.md)). **XAI and Deepseek are blank here because they are
text-only, not because the run is outstanding**; nothing is waiting to be scheduled for them. Six
runners produced those four columns: the current Gemini and OpenAI pair, added 2026-09-10, plus
Qwen and Gemma, and the two superseded models whose columns are kept in `Tempo_results.xlsx` only.

## Paths

| | Path |
|---|---|
| Local | `Tasks_benchmarks/DocVQA` |
| Quest | `/gpfs/projects/p32983/Tasks_benchmarks/DocVQA` |

**Where upstream is, what the challenge defines the task as, and why this project sends the image
rather than the OCR the dataset ships: [docvqa-method.md](docvqa-method.md).** Read it before
proposing a change of method — the alternatives were costed on 2026-09-09 and rejected with
reasons.

## Layout

```
DocVQA/
├── openai_eval.py · openai_eval_submit_array.sh · openai_eval_run_merge.sh
├── merge_openai_results.py · cleanup_shards.py       sharding support
├── gemini_eval.py · gemini_eval_script.sh
├── qwen_DocVQA/qwen_eval.py · gemma_DocVQA/gemma_eval_half{1,2}.py
├── docvqa_eval_core.py · merge_docvqa_shards.py      the shared core, added 2026-09-10
├── DOCVQA_GPT_5.6_Luna/ · DOCVQA_Gemini_Flash3.5lite_Google/
│   └── <vendor>_docvqa_eval.py · run_docvqa.sh · results/
├── DOCVQA_{Gemini_Flash3.5lite_OpenRouter,Qwen_DeepInfra,Gemma_DeepInfra}/   10% sample only
│   └── <vendor>_docvqa_eval.py · run_docvqa_sample10.sh · results_sample10/
├── sample10.json · test_docvqa_sample10.py           the 10% sample and its offline test
├── docvqa_output/docvqa_validation.json · images/    the data — 3.5 GB, mostly page images
├── openai_partial_results/                           shard state from the interrupted run
├── OpenAI_tesing/                                    retry probes (sic — the folder is misspelled)
└── OPENAI_EVAL_NOTES.md
```

**Two generations of runner coexist here.** The four that produced results are standalone scripts
with their own copy of the scorer, their own resume and their own sharding. The two added
2026-09-10 — `DOCVQA_GPT_5.6_Luna` and `DOCVQA_Gemini_Flash3.5lite_Google`, the current pair —
import **`docvqa_eval_core.py`** instead, on the shape bbh and MMLU already use. **The core's
scorer and prompt are copied verbatim from the four**, deliberately: those results stay valid only
if the new models are judged by the same matcher and asked the same question. Changing anything in
the core means rescoring all six, not just scoring the new two.

Gemini runs on the **native Google AI Studio route** (`GEMINI_FLASH_LITE_API_KEY`), matching the
route MMLU's Gemini column was finished on, so that model's config is the same across both
benchmarks. Two things about that key are worth knowing: Quest's `DocVQA/.env` did not carry it
until 2026-09-10, and the SDK will silently prefer an ambient `GOOGLE_API_KEY` or `GEMINI_API_KEY`
over the one you pass — on Quest that would have picked up the 2.5 run's `AIzaSy` key, a different
quota, so the runner deletes both from the environment before building the client.

## The 10% sample path (`--sample sample10`, 2026-11-15)

For criteria 1·2·3·6 ([sample10-plan.md](../../sample10-plan.md) WP4): **539 of the 5,349 questions**,
9 strata by `question_types[0]`, the same `questionId`s for GPT, Gemini, Qwen and Gemma.
`core.run_sample` writes `<model folder>/results_sample10/` (`smoke_sample10/` under `--limit`) and
**never reads or writes `results/`**; every row carries the per-row usage fields, plus `billed_*`
(tokens, `billed_cost`, `billed_estimated_cost`) and `attempts_without_usage` — every HTTP call's
cost, not just the answer's, and cumulative over a row's earlier failed runs (`prior_n_attempts`,
`prior_failed_runs`, `resumed_after_failure`). Runners: `--sample`
added to `DOCVQA_GPT_5.6_Luna`, and three folders that exist only for it. `run_docvqa_sample10.sh` in
each is one job, no array. **Each runs today's settled config (D1 = B), not the condition behind its
reported number** — Gemini on OpenRouter, Qwen and Gemma on DeepInfra; each runner's docstring says
what differs. Without `--sample` nothing changed: `test_docvqa_sample10.py` runs the path against
commit `1cc5bbd5` and compares every file byte for byte. **`parse_ok`** means the extractor gave
a non-empty answer; it falls back to the whole reply without `Final Answer:`, as every DocVQA score
did, so `has_marker` is the strict reading. `parse_ok` false means score 0 and ANLS 0: `load_sample`
refuses an empty gold answer, the one case where ANLS credits an empty reply.

- **Never smoke the full-run path with `--limit` alone.** Without `--sample`, `--limit` writes the
  untagged filename into `results/`, over the merged full-run file — kept, as that path must not change.
- **One message layout for all four**: page, then text, as GPT's full run; Qwen's and Gemma's
  reported runs sent the text first.
- **Question 59607 is in the sample**: one of Gemini's five deterministic `MALFORMED_RESPONSE`
  rows on the native route. If it reproduces, its row is written empty and the job exits 1.
- **Gemini sends no `provider.require_parameters`**, so OpenRouter could silently drop a setting no
  backend supports. Kept: it is the settled config every benchmark uses (2026-11-15).
- **Qwen and Gemma halt at the first row showing thinking** (reasoning text, a think tag inline in
  `content`, or reasoning tokens): `sample10_THINKING_HALT.txt`, exit 3; the marker names
  `results_sample10/` (`smoke_sample10/`) as the directory to archive. A resubmit on the same config
  halts before calling any row; only the one-call negotiation probe runs, as the config comes from it.
- **A halt or SIGTERM writes the row in flight as failed** (`error` starts `interrupted:`), so its
  billed calls stay in the file and fold into the next run of it.
- **A row's HTTP calls in one run: `n_attempts + cut_retry + forced_answer`** (true = 1). One short
  if no attempt returned after a cut first call (no `cut_retry` then; `billed_*` holds it).
  `attempts_without_usage` counts calls, not attempts.

## Expected counts

**5,349 validation questions.** The interrupted run left ~3,021 of them with empty responses, so a
result set here is only meaningful alongside its empty count.

## Its own traps — all four from one incident

Recorded in `OPENAI_EVAL_NOTES.md`, which is authoritative on this benchmark:

1. **Shards share one API key's daily quota.** Five parallel shards drew on the same 10,000 RPD cap
   and exhausted it; each then logged the same refusal ~210 times. Shard count was cut 10 → 5.
2. **An RPD refusal must not be retried.** Retrying a daily-cap error burns the retry budget against
   a wall that only clears at midnight.
3. **Resume skips already-answered questions**, so a partial shard is worth keeping — but only if the
   empty rows are pruned first, or resume treats them as answered.
4. **`timeout=30` was added to the API calls** after hanging requests; the notes' §"Problem 3" ties
   the hang directly to the shard design.

The notes also carry a cost estimate and a written resume procedure — read them before restarting,
not after.

## Results — all four image-capable models, verified 2026-09-10

5,349 validation questions each. **ANLS is the reported metric**; accuracy is the binary column,
lenient since `docvqa_lenient_v2` on 2026-09-11, and is shown because a gap between the two says
the model was right but phrased it differently.

| Slot | Model | ANLS | Accuracy | Empty | Among the six |
|---|---|---|---|---|---|
| Qwen | `Qwen/Qwen3.5-9B` | **0.9568** | 0.9705 | 3 | yes |
| Gemini | `gemini-3.5-flash-lite` | **0.9394** | 0.9686 | 5 | yes |
| Gemma | `google/gemma-4-31B-it` | **0.9293** | 0.9405 | 16 | yes |
| OpenAI | `gpt-5.6-luna` | **0.8635** | 0.9288 | 0 | yes |
| — | `gemini-2.5-flash` | 0.9357 | 0.9587 | 3 | no — superseded |
| — | `gpt-4o-mini-2024-07-18` | 0.8583 | 0.8841 | 0 | no — superseded |
| XAI · Deepseek | — | — | — | — | **cannot run — text-only** |

Both replacements beat the model they replaced, narrowly: Gemini 0.9394 against 0.9357, OpenAI
0.8635 against 0.8583. **Qwen still leads this benchmark.**

**Gemini's five empty rows are permanent, not a retry away.** All five return `finish_reason
MALFORMED_RESPONSE` with thought tokens spent and no output part, and re-asking them at the same
seed reproduces it exactly. They are scored 0. 5 of 5,349 is 0.09%, and it is recorded here rather
than rounded away because a future run that hits the same thing should recognise it.

**Do not compare these with the paper's 0.665.** That baseline reads serialised OCR; these read the
page. Different input, different era.

Per-model cells live in the workbooks — `Final_result.xlsx` for the selected models that can run
it, `Tempo_results.xlsx` for those plus the superseded pair — each with its sources on the `Provenance`
sheet.

## Scoring

**ANLS is the reported metric and it is graded** — a near-miss scores its similarity rather than
zero, taking the **maximum over each question's answer list** with the official **0.5 threshold**.
The binary `score` column is a project addition and is at `docvqa_lenient_v2`.

**Both the version history and the fuzzy branch that was measured and refused:
[docvqa-scoring.md](docvqa-scoring.md).** Read it before proposing that the matcher be loosened —
three fuzzy variants were measured over all 32,094 stored rows and each one credited misread names,
phone numbers or dollar amounts.
