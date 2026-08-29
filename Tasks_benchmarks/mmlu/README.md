# MMLU — what it runs, how it is scored, how this folder is laid out

General task ability, multiple choice. Accuracy. No LLM judge.

## What the data is

**13 subject files, 3,943 items** (counted 2026-08-29). Sizes are very uneven — `Business_ethics`
100, `Miscellaneous` 783, `Moral_scenarios` 895 — so **a macro-average over subjects is not the same
as pooling all 3,943 items**. The workbook reports the macro-average.

Each `data/<Subject>.json` is a **list** of `{question, subject, choices[4], answer}`. **`answer` is
an index as a string**: `"2"` means `choices[2]`, not "choice number 2". Gold text is
`choices[int(answer)]`; gold letter is `"ABCD"[int(answer)]`.

## Two prompts, and they are not interchangeable

MMLU arrived with **two materially different prompts**, not a whitespace difference:

| | Used by | Choices shown as | Asks for |
|---|---|---|---|
| **v1** | Deepseek, GPT-4o-mini, Gemini-2.5 | a raw Python list | the answer **text** |
| **v2** | Gemma, Llama, Qwen, XAI | labelled `A.`–`D.` | a **letter** |

Four runners were asked for a letter and three for the text — different tasks. **New runs default to
v2**: it is the majority, and a single letter is unambiguous to score. Of the six reported models
that leaves **Deepseek alone on v1**; re-pointing it means re-running it, not editing a header.

The prompt version is part of the run config and is written onto every row, so a resume across a
change of it is refused rather than silently mixing two prompts in one result set.

## Scoring — one matcher, imported, for every model

`mmlu_eval_core.py::score_response`, at `mmlu_lenient_v1`. **MMLU had exactly the split bbh had**:
`<provider>_eval.py` compared the model's text to the gold choice with `==`, while a separate
`<provider>_rescore.py` — written for only three of seven providers — accepted a letter, an index,
or letter-plus-text. A model that answered `C` scored 0 under one and 1 under the other.

Seven branches, all naming the *same* choice a different way — the text; the bare letter; the index;
letter-then-text; comma-vs-space; the letter at the end; and `the answer is C`. The last two fire
only when the model actually emitted `Final Answer:`, because without it the extracted answer is a
scrape of the whole response and a letter appearing in the reasoning is not a choice. Quotes,
markdown bold and LaTeX wrappers are stripped first. These are the lessons bbh paid for over five
scorer versions — [`bbh-scoring.md`](../../.claude/references/benchmarks/tasks/bbh-scoring.md).

## Layout

```
mmlu/
├── mmlu_eval_core.py          the one scorer, the subject list, both prompts, the write path
├── data/<Subject>.json × 13   shared by every runner
├── .env                       provider keys (gitignored)
└── MMLU_<Slot>/               Deepseek · Gemma · Qwen · XAI · Llama
    │                          · Gemini_Flash2.5 · GPT_4o_mini   (superseded, hold the old numbers)
    │                          · Gemini_Flash3.5lite_OpenRouter · GPT_5.6_Luna  (current, NOT RUN)
    ├── <vendor>_mmlu_eval.py  a client and a `call`; no scorer
    ├── run_mmlu.sh
    └── results/<Subject>/<model-slug>.{jsonl,csv} + _overall.csv
```

Reorganised 2026-08-29 from a flat directory. Result files are named after the **model**, not the
folder, and `--model` sets both what is called and what is written, so a copied folder cannot
relabel another model's numbers.

**Run it:** `cd MMLU_GPT_5.6_Luna && sbatch run_mmlu.sh`, or
`python gpt56luna_mmlu_eval.py --subject all --prompt v2 --workers 5`. `--limit N` truncates each
subject for a smoke test — a partial run, never a number to report.

## What to know before reading a number here

- **The two current models have runners and no results.** `MMLU_GPT_5.6_Luna` and
  `MMLU_Gemini_Flash3.5lite_OpenRouter` are two of the six; the workbook's Gemini and OpenAI columns
  are already re-pointed to them and blank. Their caps (`max_completion_tokens=16384`,
  `max_tokens=8192`) are **chosen, not measured** — run `--limit 20` and check `no_marker` first.
- **The stored rows predate the shared core** and were produced by the per-provider runners under
  whichever of the two prompts that runner used, and scored by whichever of the two scorers. Rescore
  them through the core before comparing models.
- **`Llama` has a runner but no subject CSVs**, and is not one of the six.
