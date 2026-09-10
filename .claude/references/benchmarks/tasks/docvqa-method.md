# DocVQA — upstream, and why the method is what it is

<!-- size-budget: 6000 -->
<!-- Split out of docvqa.md on 2026-09-10, when that card crossed its budget a third time. Two
     jobs, not one: the card is operational — paths, layout, counts, results, traps — while this
     page answers "why is it done this way", which is asked once every few months and must not be
     re-derived from scratch each time. -->

The card itself is [docvqa.md](docvqa.md). This page holds what settles arguments about the method:
where upstream actually is, what the challenge defines the task as, and the decision taken against
that definition.

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
