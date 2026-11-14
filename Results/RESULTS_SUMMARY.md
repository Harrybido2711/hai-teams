# Results summary — five models on ten team-process benchmarks

Status: 14 November 2026. Nine benchmarks are complete runs. **PlanBench is a pilot on one tenth of
its questions**; its numbers will be replaced when the full run is in.

This note goes with the workbook `Final_result.xlsx`. It says what each benchmark's result is, what
to keep in mind before comparing models on it, and how the benchmarks are averaged.

## What was measured

Ten benchmarks, chosen to cover what working in a team requires: *transition* processes (analysing
the mission, planning, setting goals), *action* processes (monitoring progress, coordinating) and
*interpersonal* processes (managing conflict and affect), with three benchmarks of general task
ability beside them as a baseline. Five models were run on every benchmark they can take.

| Column | Model |
|---|---|
| Gemini | Google `gemini-3.5-flash-lite` |
| OpenAI | OpenAI `gpt-5.6-luna` |
| Qwen | Alibaba `Qwen3.5-9B` |
| Gemma | Google `gemma-4-31B-it` |
| Deepseek | DeepSeek `deepseek-reasoner` — text only, so it has no score on the two benchmarks that send images |

## How the workbook is laid out

- **`Summary`**, the first sheet, lists every benchmark's result: the numbers in the table below.
- **One sheet per benchmark**: a row for each sub-benchmark, a column for each model, the model's
  exact name in row 2. **One row on every sheet is highlighted in yellow. That row is the
  benchmark's result** — the single number used whenever benchmarks are compared or averaged.
- **`Provenance`**, the last sheet, says for every sheet and column which run the numbers come
  from, how they were scored, and what in them is unusable.

A blank cell means there is no usable number. It never means zero.

## Results

<!-- generated:results -->
| Benchmark | Team process | Gemini | OpenAI | Qwen | Gemma | Deepseek | Best of each sub-benchmark |
|---|---|--:|--:|--:|--:|--:|--:|
| BIG-Bench Hard | general task ability | 0.9375 | 0.9349 | 0.9339 | **0.9684** | 0.9616 | 0.9696 |
| MMLU | general task ability | 0.9135 | **0.9176** | 0.8802 | 0.9071 | 0.9174 | 0.9277 |
| DocVQA | general task ability | 0.9686 | 0.9288 | **0.9705** | 0.9405 | N/A | 0.9705 |
| AwareBench | transition — mission analysis | 0.7936 | 0.7909 | 0.6912 | 0.7897 | **0.9159** | 0.9274 |
| PlanBench (pilot) | transition — strategy formulation | 0.4316 | 0.7859 | 0.4541 | 0.5605 | **0.9200** | 0.9200 |
| Multi-party Goal Tracking | transition — goal specification | 0.2040 | 0.2640 | 0.1177 | 0.2583 | **0.2900** | 0.2958 |
| Wonderbread | action — monitoring progress | **0.7707** | 0.7699 | 0.7167 | 0.7614 | N/A | 0.8135 |
| MultiChallenge | action — coordination | 0.4528 | 0.4434 | 0.2835 | 0.4893 | **0.5795** | 0.5795 |
| EmoBench | interpersonal — affect management | 0.6975 | 0.6875 | 0.6350 | **0.7450** | 0.6725 | 0.7550 |
| NegotiationToM | interpersonal — conflict management | 0.5570 | **0.5788** | 0.5029 | 0.5477 | 0.5538 | 0.5834 |
<!-- /generated:results -->

Every result is on a 0–1 scale and higher is better. Bold is the best model on that benchmark. N/A
means the model cannot take the benchmark's input. The last column is the score reached when the
best model is chosen separately on every sub-benchmark; it is the benchmark's share of Oracle 2,
defined at the end.

## What each result is

On most sheets the result is the benchmark's usual overall score. Where the benchmark's authors
report several numbers and no total, the highlighted row is **ours**: a composite defined for this
project and labelled on the sheet with its formula. That is the case for four benchmarks.

| Benchmark | Sheet | What it asks | Its result |
|---|---|---|---|
| BIG-Bench Hard | `Big Bench Hard` | 20 reasoning tasks, 4,833 questions, scored by exact match | mean of the 20 task accuracies |
| MMLU | `MMLU` | 13 of MMLU's 57 subjects, 3,943 multiple-choice questions | mean of the 13 subject accuracies |
| DocVQA | `DocVQA` | 5,349 questions about images of documents | accuracy, the share of answers counted right — a row added for this project. ANLS, the benchmark's own graded metric, is the row beside it |
| AwareBench | `Awareness` | 4,015 prompts on whether a model knows its own capability and mission and reads emotion, culture and another person's perspective | mean of the five dimension accuracies |
| PlanBench (pilot) | `PlanBench (pilot)` | writing, checking, reusing and repairing plans, 10 task types; a plan validator decides | **ours**: correct ÷ questions with all tasks pooled; the authors report each task on its own |
| Multi-party Goal Tracking | `Multi-party Goal Tracking` | filling in the dialogue state and each speaker's goal in conversations between several people, under six prompt styles | **ours**: mean of the two tasks, each the mean of its six styles |
| Wonderbread | `Wonderbread` | answering questions about recorded web workflows, and writing the step-by-step procedure for them; an LLM judge grades | **ours**: mean of the eight judged criteria, the four question-answering ones first converted to 0–1 |
| MultiChallenge | `MultiChallenge` | 272 multi-turn conversations of four challenge types; an LLM judge decides pass or fail | mean of the four pass rates |
| EmoBench | `Emo` | 400 multiple-choice questions on understanding emotions and on acting on them | mean of the two halves, understanding and application |
| NegotiationToM | `NegotiationToM` | inferring what the parties to a negotiation want, believe and intend, 14,138 questions | **ours**: mean of four metrics — exact match on desires, exact match on beliefs, micro F1 and macro F1 on intentions |

## Before comparing models

**On every sheet.** A column is one model but not always one configuration. Where a model has a
reasoning setting it was kept short on most benchmarks — Gemini at minimal effort, OpenAI at low,
Qwen and Gemma with thinking off — but the setting is not identical on all ten. The Deepseek column
is `deepseek-reasoner` on the older runs and `deepseek-flash`, the model that name now points to, on
AwareBench, PlanBench, Multi-party Goal Tracking and MultiChallenge. `Provenance` records each case.

**BIG-Bench Hard** is near its ceiling. On 15 of the 20 tasks at least one model scores 1.000, so
the models differ by little. Qwen's `dyck_languages` (0.696) counts 17 unusable answers as wrong.

**MMLU** is 13 subjects, not the whole benchmark, and two answer-matching rules sit behind one
sheet: Gemini and OpenAI are scored by a lenient matcher, Qwen, Gemma and Deepseek by strict
equality, and Deepseek was asked for the answer's text where the others were asked for a letter.
OpenAI's lead over Deepseek is 0.0002 — a tie.

**DocVQA** has two rows of different kinds. Accuracy, the result, counts an answer right or wrong;
ANLS gives partial credit for a near miss. Do not average them.

**AwareBench**'s result is the mean of five dimensions, not of the sheet's rows: `mission` already
combines the two rows above it, and `mission_open-ended` is empty for every model because those
questions need a human or LLM grader and were left out. `perspective (2nd-order)` is 170 questions
and a fifth of the result. Qwen's 0.0 there is real — it declined every one — and Gemma's 0.129
includes 124 it declined, so that row reflects refusals more than ability.

**PlanBench** is a pilot: every tenth question, 1,513 scored per model, so a task row rests on
between 10 and 199 questions. Qwen's column is a lower bound — 320 of its 1,513 answers were cut off
at the output limit. `task_3_plan_verification_with_llm_plans` cannot pass 0.7, because 3 of its 10
pilot questions have no reference answer. Two question sets, 50 questions a model between them, are
left out for every model because the benchmark's own parser cannot score them.

**Multi-party Goal Tracking** is scored by an automatic matcher written for this project; the
benchmark's authors reviewed answers by hand. An answer the matcher cannot place counts as wrong:
Qwen's `dst_reasoning` 0 is answers in a shape it cannot read, not answers it read and found wrong.
Scores are low for every model, goal
0.06–0.24, because models name a goal in their own words rather than the scheme's. OpenAI, Gemini
and Qwen are lower bounds: an answer list with the wrong number of items earns nothing, and matched
by position OpenAI would lead on `dst`.

**Wonderbread**: every cell is the grade of an LLM judge (Claude Haiku 5.5, three gradings each),
and the judge's agreement with human graders has not been measured. The four `qa_` rows are graded
1–3 where **1 is best**, so lower is better there — the opposite of every other row in the
workbook. The result row converts them with (3 − x) / 2 before averaging. The models are not scored
on identical sets: an item with more screenshots than a model accepts was left out for that model
(questions / procedures scored: Gemini and OpenAI 120 / 486, Qwen 113 / 482, Gemma 115 / 482).

**MultiChallenge** is graded by the same judge, with the same caveat. `Overall (micro)`, under the
result, pools the same conversations into one rate; it is a second reading, not a second score, and
is never averaged in. 272 of 273 conversations are scored, and a challenge type holds 41 to 113 of
them, so single rows are noisy.

**EmoBench** is small. With 400 questions, one question moves a row by 0.015–0.036. The result is
the mean of its two halves, not of the eight rows. Gemma's column was run at temperature 0.6.

**NegotiationToM**: `All_EM` is not the total. It is a separate, stricter metric — a dialogue counts
only if every question about it is answered right — so 0.04–0.09 is its normal range. In the result,
intentions count twice, since micro and macro F1 are two readings of the same answers. Gemma's
`Belief_EM` was measured with thinking off: 0.49, against 0.62 with it on over the 463 questions
measured both ways.

## Averages across benchmarks

Every benchmark weighs the same, whatever its number of questions.

- **A model's mean** is the average of its results. A model with no score on one of the benchmarks
  counted has no mean there: it is never given a zero, and never averaged over fewer benchmarks than
  the others. That is why Deepseek has a mean only where the two image benchmarks are left out.
- **Oracle 1** takes, on each benchmark, the result of whichever model is best, then averages. It is
  the ceiling for choosing one model per benchmark.
- **Oracle 2** takes the best model on every sub-benchmark — every row of a sheet that is not itself
  an aggregate — combines those the way the benchmark's result combines its rows, then averages. It
  is the ceiling for choosing a model per sub-benchmark. On a Wonderbread `qa_` row the best model
  is the one with the lowest grade.

<!-- generated:averages -->
| Benchmarks counted | How many | Gemini | OpenAI | Qwen | Gemma | Deepseek | Oracle 1 | Oracle 2 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| All, the pilot included | 10 | 0.6727 | **0.7102** | 0.6186 | 0.6968 | — | 0.7656 | 0.7742 |
| Those every model has a score on, the pilot included | 8 | 0.6234 | 0.6754 | 0.5623 | 0.6583 | **0.7263** | 0.7394 | 0.7448 |
| Complete runs only | 9 | 0.6995 | 0.7018 | 0.6368 | **0.7119** | — | 0.7485 | 0.7580 |
| Complete runs every model has a score on | 7 | 0.6508 | 0.6596 | 0.5778 | 0.6722 | **0.6987** | 0.7136 | 0.7198 |
<!-- /generated:averages -->

Bold is the best single model. Both oracles choose with the scores already known, so they are upper
bounds on what choosing a model in advance could reach, not estimates of it. The rows that include
the pilot count PlanBench at its pilot result.
