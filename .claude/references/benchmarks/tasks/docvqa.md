# DocVQA — benchmark card

<!-- size-budget: 8000 -->
<!-- One job — the card for one benchmark — and it is over the nudge because two of its sections
     exist to stop questions being re-asked: where upstream is (there is no vendored repo, and that
     is correct), and why the method is image-only (the challenge's own task definition, plus a
     decision the user made on 2026-09-09 after the alternative was costed). Deleting either is how
     the same afternoon gets spent twice. Raised again on 2026-09-10 when the benchmark gained
     a shared core and two runners on it, which put two generations of runner in one folder — the
     paragraph explaining which is which is what stops a reader copying the wrong one. -->

Document visual question answering. Upstream docvqa.org. Scored by **ANLS**, no LLM judge.

**Four of the six have results, and four is the ceiling.** `gemini_eval.py`, `openai_eval.py`,
`qwen_DocVQA/` and `gemma_DocVQA/` — exactly the four models that accept an image
([model-calls.md](../../model-calls.md)). **XAI and Deepseek are blank here because they are
text-only, not because the run is outstanding.** Nothing is waiting to be scheduled for them.

## Upstream, and why nothing is vendored here

| | |
|---|---|
| Paper | Mathew, Karatzas & Jawahar, *DocVQA: A Dataset for VQA on Document Images*, WACV 2021 · [arXiv 2007.00398](https://arxiv.org/abs/2007.00398) |
| Official code | [github.com/mineshmathew/DocVQA](https://github.com/mineshmathew/DocVQA) — the authors' **baselines only**, 10 files. `M4C_baseline/README.md` is a 0-byte placeholder |
| Data | [rrc.cvc.uab.es/?ch=17](https://rrc.cvc.uab.es/?ch=17) § Downloads → **Task 1 · Single Page Document VQA**. Registration-gated: "You will need to register to get access to the download section" |

**This is the one benchmark with no vendored upstream repo, and that is correct rather than an
omission.** The data cannot be cloned — it is a registered download — and the baseline repo is BERT
and M4C implementations we do not use. Do not go looking for a missing checkout.

## The task is defined on the image, and OCR is optional — settled 2026-09-09

Checked against the challenge's own Task 1 page after the question was raised:

> "The objective of this task is to answer questions asked on **a document image**." · "The answers
> to questions are short text spans **taken verbatim from the document**." · Task 3's download note:
> "OCR outputs are **auxiliary data**; participants are free to use **any OCR**."

Task 1's definition mentions OCR exactly once, and only to explain the metric — ANLS penalises
smoothly so as to tolerate "OCR recognition errors". **So sending the page image to a multimodal
model is compliant and is the reading closest to the definition**; the 2021 baselines used OCR text
because no model could read the page, not because the rules asked for it.

**The user decided on 2026-09-09 to keep the image-only method.** The alternative was considered and
rejected with the reasons on record, so do not re-open it without a new one:

- **OCR-only** would let the two text-only models run and close DocVQA's blank columns, but it
  measures answering from serialised text, drops the layout the paper says models struggle with
  most, and its rows would not be comparable with the four already collected.
- **Image + OCR** stays within the definition but still leaves those two columns blank, since they
  cannot take the image at all.

**We hold no OCR anyway** — neither locally nor on Quest. The Microsoft OCR file ships separately
from the images on RRC and was never downloaded, and Quest has no OCR tooling either (`tesseract`
absent, `pytesseract` / `paddleocr` / `easyocr` / `cv2` all missing from the project env). Any future
OCR experiment starts with that download.

**Why our numbers sit above the paper's.** The BERT baseline reads serialised OCR and extracts a
span — 0.665 ANLS on test. Our four models read the page and score 0.93–0.96. Different input,
different era: **it is not a like-for-like comparison and must not be reported as beating the
baseline.**

## Scoring

**ANLS, and it must take the maximum over the answer list.** Every question carries several
acceptable answers, so scoring against one of them systematically under-reports. `gemini_eval.py`
does this correctly: normalised Levenshtein, the standard **0.5 threshold** below which the score is
zero, and `max` across the gold list.

## Paths

| | Path |
|---|---|
| Local | `Tasks_benchmarks/DocVQA` |
| Quest | `/gpfs/projects/p32983/Tasks_benchmarks/DocVQA` |

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
