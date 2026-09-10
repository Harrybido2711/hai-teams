# DocVQA — benchmark card

<!-- size-budget: 7000 -->
<!-- One job — the operational card for one benchmark: paths, layout, counts, results, traps. The
     other job, "why is the method what it is", moved to docvqa-method.md on 2026-09-10 after this
     file crossed its budget a third time. It is still over the 5 KB nudge because six runners in
     two generations live in one folder and the paragraph saying which is which is what stops a
     reader copying the wrong one. -->

Document visual question answering. Upstream docvqa.org. Scored by **ANLS**, no LLM judge.

**Four of the six have results, and four is the ceiling** — exactly the models that accept an
image ([model-calls.md](../../model-calls.md)). **XAI and Deepseek are blank here because they are
text-only, not because the run is outstanding**; nothing is waiting to be scheduled for them. Six
runners produced those four columns: the current Gemini and OpenAI pair, added 2026-09-10, plus
Qwen and Gemma, and the two superseded models whose columns are kept in `Results.xlsx` only.

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

5,349 validation questions each. **ANLS is the reported metric**; accuracy is exact match with the
four tolerances the scorer allows, and is shown because a gap between them says the model was
right but phrased it differently.

| Slot | Model | ANLS | Accuracy | Empty | Among the six |
|---|---|---|---|---|---|
| Qwen | `Qwen/Qwen3.5-9B` | **0.9568** | 0.9611 | 3 | yes |
| Gemini | `gemini-3.5-flash-lite` | **0.9394** | 0.9602 | 5 | yes |
| Gemma | `google/gemma-4-31B-it` | **0.9293** | 0.9306 | 16 | yes |
| OpenAI | `gpt-5.6-luna` | **0.8635** | 0.9151 | 0 | yes |
| — | `gemini-2.5-flash` | 0.9357 | 0.9491 | 3 | no — superseded |
| — | `gpt-4o-mini-2024-07-18` | 0.8583 | 0.8746 | 0 | no — superseded |
| XAI · Deepseek | — | — | — | — | **cannot run — text-only** |

Both replacements beat the model they replaced, narrowly: Gemini 0.9394 against 0.9357, OpenAI
0.8635 against 0.8583. **Qwen still leads this benchmark.**

**Gemini's five empty rows are permanent, not a retry away.** All five return `finish_reason
MALFORMED_RESPONSE` with thought tokens spent and no output part, and re-asking them at the same
seed reproduces it exactly. They are scored 0. 5 of 5,349 is 0.09%, and it is recorded here rather
than rounded away because a future run that hits the same thing should recognise it.

**Do not compare these with the paper's 0.665.** That baseline reads serialised OCR; these read the
page. Different input, different era.

Per-model cells live in the workbooks — `Final_Result.xlsx` for the four of the six that can run
it, `Results.xlsx` for those plus the superseded pair — each with its sources on the `Provenance`
sheet.

## Scoring

**ANLS, and it must take the maximum over the answer list.** Every question carries several
acceptable answers, so scoring against one of them systematically under-reports. `gemini_eval.py`
does this correctly: normalised Levenshtein, the standard **0.5 threshold** below which the score is
zero, and `max` across the gold list.
