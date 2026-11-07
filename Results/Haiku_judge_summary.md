# Haiku 5.5 judge summary — MultiChallenge and Wonderbread

Judge: Claude Haiku 5.5, 3 replicates per unit. Every number is rebuilt from the row files on disk by `Results/build_judge_summary.py`. **Preliminary: not yet audited, and not yet in `Tempo_results.xlsx` / `Final_result.xlsx`.** N/A = the model cannot take that benchmark.

## MultiChallenge (higher is better, %)

| Model | Macro (4 axes) | Micro | Items scored | Inference Memory | Instruction Retention | Reliable Version Editing | Self Coherence |
|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | 44.3 | 45.2 | 272 / 273 | 39.8 | 66.7 | 34.1 | 36.7 |
| gemini-3.5-flash-lite | 45.3 | 48.9 | 272 / 273 | 50.4 | 69.6 | 24.4 | 36.7 |
| Qwen3.5-9B | 28.4 | 31.2 | 272 / 273 | 39.8 | 27.5 | 19.5 | 26.5 |
| gemma-4-31B-it | 48.9 | 52.2 | 272 / 273 | 55.8 | 66.7 | 36.6 | 36.7 |
| deepseek-flash | 58.0 | 59.2 | 272 / 273 | 56.6 | 75.4 | 48.8 | 51.0 |

All five models are scored on the same 272 items: item `6765fc5e0af93a24e8fbe4ee` is left out for every model, because Gemini's answer to it was blocked by Google's RECITATION filter (5 of 5 attempts) and the user decided it is not scored (2026-11-07). Macro = mean of the four axes; an item's label is the majority of its 3 replicates.

## Wonderbread (N/A = text-only model)

QA: judge scores 1–3, **1 is best**, mean over answers and 3 replicates. SOP: precision / recall / F1 / ordering, higher is better.

| Model | Completeness | Soundness | Clarity | Compactness | SOP precision | SOP recall | SOP F1 | SOP ordering |
|---|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | 1.82 | 1.59 | 1.16 | 1.36 | 0.757 | 0.876 | 0.799 | 0.696 |
| gemini-3.5-flash-lite | 1.91 | 1.77 | 1.12 | 1.67 | 0.870 | 0.865 | 0.855 | 0.810 |
| Qwen3.5-9B | 1.81 | 1.75 | 1.16 | 1.74 | 0.743 | 0.774 | 0.736 | 0.712 |
| gemma-4-31B-it | 1.94 | 1.69 | 1.20 | 1.28 | 0.832 | 0.851 | 0.828 | 0.635 |
| deepseek-flash | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Coverage — how many units each model was scored on

| Model | MC items | WB QA answers | WB SOPs | WB QA not run (image cap) | WB SOP not run (image cap) |
|---|---|---|---|---|---|
| gpt-5.6-luna | 272 / 273 |  120 / 120 | 486 / 486 | 0 | 0 |
| gemini-3.5-flash-lite | 272 / 273 |  120 / 120 | 486 / 486 | 0 | 0 |
| Qwen3.5-9B | 272 / 273 |  84 / 120 | 384 / 486 | 36 | 102 |
| gemma-4-31B-it | 272 / 273 |  115 / 120 | 482 / 486 | 5 | 4 |
| deepseek-flash | 272 / 273 |  N/A | N/A | N/A | N/A |

## Two things to read the table with

**(a) A model's own limit means some questions could not be run.** Wonderbread questions carry many screenshots, and Qwen3.5-9B and gemma-4-31B-it accept only a fixed number of images per request. A question that needs more is never sent — it is recorded as `over_image_limit`, not judged and not counted as a wrong answer. Qwen is scored on 84 of 120 QA and 384 of 486 SOP units; Gemma on 115 of 120 and 482 of 486. The other models are scored on every unit, so a Wonderbread number is not computed over the same questions for every model; the Coverage table shows the denominator.

**(b) A text-only model cannot complete every benchmark.** deepseek-flash takes no images, so it cannot run Wonderbread at all and shows N/A there. It is compared with the others only on MultiChallenge, and its row is not comparable across the whole suite. The same applies to any text-only model.
