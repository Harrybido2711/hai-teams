# MMLU — benchmark card

General task ability, multiple choice. Accuracy. No LLM judge.

## Paths

| | Path |
|---|---|
| Local | `Tasks_benchmarks/mmlu` |
| Quest | not verified — nothing points to a remote copy, but Quest was not checked (out of scope 2026-08-22). Confirm before assuming a remote path |

## Layout — rebuilt 2026-08-29 on the EmoBench convention

`MMLU_<Slot>/` per model, holding `<vendor>_mmlu_eval.py`, `run_mmlu.sh` and
`results/<Subject>/<model-slug>.{jsonl,csv}` + `_overall.csv`. The 13 subject JSONs live in `data/`
and a shared `mmlu_eval_core.py` owns the scorer, both prompts and the write path.

Slots **with results**: `MMLU_Deepseek` · `MMLU_Gemma` · `MMLU_Qwen` · `MMLU_XAI` ·
`MMLU_Gemini_Flash2.5` · `MMLU_GPT_4o_mini` (the last two superseded, named for what they call).
`MMLU_Llama` has a runner and no subject CSVs. Slots **with a runner and no results**:
`MMLU_GPT_5.6_Luna` and `MMLU_Gemini_Flash3.5lite_OpenRouter` — two of the six, added 2026-08-29,
unpiloted.

## Expected counts

**13 subjects, 3,943 items**, counted 2026-08-29. Sizes are very uneven (100 to 895), so the
macro-average over subjects the workbook reports is **not** the same as pooling all items.
`answer` in the data is an **index as a string** — `"2"` means `choices[2]`.

## Two prompts, and they are not interchangeable

v1 (Deepseek, GPT-4o-mini, Gemini-2.5) shows the choices as a raw list and asks for the answer
**text**; v2 (Gemma, Llama, Qwen, XAI) labels them `A.`–`D.` and asks for a **letter**. Four runners
were asked for a letter and three for the text — a materially different task, not a formatting
difference. **New runs default to v2**, which leaves Deepseek alone on v1 among the six. The prompt
version is part of the run config and a resume across a change of it is refused.

## Scoring — one matcher, imported, for every model

`mmlu_eval_core.py::score_response`, `mmlu_lenient_v1`. **MMLU had exactly bbh's split**: the
`*_eval.py` runners compared text with `==` while a separate `*_rescore.py`, written for only three
of seven providers, accepted a letter, an index or letter-plus-text. Seven branches now, all naming
the same choice a different way; the two loosest fire only when the model emitted `Final Answer:`.
Built on the lessons bbh paid for across five scorer versions — [bbh-scoring.md](bbh-scoring.md).

## Its own traps

- **The rescore scripts are the cheap path.** A scoring change does not need a rerun: `*_rescore.py`
  re-derives the numbers from stored responses. Check for one before spending a provider call.
- **A `dotenv/` directory sits among the provider subdirectories** — it is not a provider.
