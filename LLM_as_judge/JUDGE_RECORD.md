# Judge record — every LLM-as-a-judge in this benchmark suite

The filled record required by [JUDGE_DOCUMENTATION_RULE.md](JUDGE_DOCUMENTATION_RULE.md), covering
every judge in the repo in one place. Written 2026-08-19, from the vendored code rather than from the
papers — rule 1 puts code first, and on all three benchmarks the code turned out to say something the
paper did not.

Three benchmarks are documented here, in the order they are usable:
[Wonderbread](#1--wonderbread) · [MultiChallenge](#2--multichallenge) · [AwareBench](#3--awarebench).
Verbatim prompts are in the appendices so the fields stay readable.

---

## Which benchmarks need no record at all

**Seven of the ten do not, because no model's opinion enters their score.** They are listed once,
here, and nowhere else — there is no per-benchmark file for them, because there is nothing about a
judge to record.

The rule is one sentence: **a judge record documents what cannot be re-derived.** Accuracy against a
gold label can be recomputed by anyone holding the dataset and the responses; a GPT score cannot be
recomputed from anything except that exact model, that exact prompt and that exact aggregation. So a
record is required precisely where a model's opinion enters the number.

| Benchmark | Scored by | Scoring code |
|---|---|---|
| Multi-party Goal Tracking | gold `[MASK]` strings — Exact Match / Correct / Partial; the residue goes to **a human**, not a model | none vendored — the three-level scorer is described in the paper only |
| PlanBench | **VAL/PDDL symbolic validator** — plan validity is decidable, not a matter of opinion | `plan-bench/response_evaluation.py:74, 104-107, 139, 229` |
| NegotiationToM | grouped Exact Match (Desire/Belief) + `sklearn` micro/macro F1 (Intention) | `neg_eval_core.py:14, 899-912` |
| EmoBench | MCQ accuracy on gold letters, majority vote across four answer-order permutations | `OpenAI_result/openai_emo_eval.py:116-153` |
| DocVQA | ANLS — normalised Levenshtein against the gold answer list | `openai_eval.py:103, 116, 225` |
| BIG-Bench Hard | exact match on the extracted final answer | `openai_eval.py:42` |
| MMLU | accuracy on the parsed letter | `openai_eval.py:43`; `*_rescore.py` re-scores offline |

**Two things in the tracker look like a judge and are not.** mpgt's *"manual review"* is a person —
substituting GPT there redefines the metric rather than automating it. And a `--model GPT4` flag
inside a benchmark's task runner names the **model under test**, not a grader; four of Wonderbread's
six subtasks are exactly that case, and it is the easiest way to mis-scope this suite.

## Substituting a different judge model

Nothing here forces GPT-4 specifically — and for Wonderbread nothing can, since all three of its
snapshots are retired (§1 D4). But **a substitution has to be declared, because judge-based scores
are not portable across judge models.** Whatever is used, fix and log the judge model version,
temperature, prompt version, retry rules and the raw judge output, keep a manually reviewed sample
for reliability auditing, and never silently score a parse failure as zero. A substituted judge's
numbers are not comparable to the paper's and must not be tabulated as if they were.

`GPT_LLM_AS_JUDGE_GUIDE.md` holds this project's judge-implementation notes — structured output,
position bias, pairwise setup.

---

**One caveat on the "no judge" verdict for keyed benchmarks.** It is only safe because the keys were
actually opened and counted. AwareBench's `mission_implicit` stores its key under `Label`/`Choices`
with capital letters while every other dimension uses lowercase; a loop written against the majority
spelling returns `None` for all 327 rows, scores every one wrong, and drops the headline by about 8
points. Our own notes fell for exactly this. "Has a gold label" is a claim to verify, not to assume.

---

## 1 · Wonderbread

Scope: six subtasks, **two judged by a model**, three scored deterministically, one with a rubric
scorer that does not execute.
Sources read: `HazyResearch/wonderbread@ed052c67aeada04167cdfe92ff8de454aa94627a` (2025-07-30),
vendored at `Action_processes_benchmarks/Wonderbread_bench` with `.git` stripped, plus the committed
results under `data/experimental_results/`. **Code read, not executed.**

| Subtask | Reported metric | Model in the loop? | Documented below |
|---|---|---|---|
| **Question Answering** | soundness / completeness / clarity / compactness, 1–3 each | **judge** — GPT-4 scores the answer | **Judge 1** |
| **SOP Generation** | "semantic" Precision / Recall / Ordering | **judge** — GPT-4 decides line-level entailment | **Judge 2** |
| SOP Improvement | 1–5 rubric score | **judge, but broken** — see D1 | **Judge 3** |
| Demo Segmentation | Adjusted Rand Index | no — `sklearn.metrics.cluster.adjusted_rand_score` | — |
| Demo Validation | F1 / Accuracy | no — labels compared directly | — |
| SOP Ranking | Kendall τ / Spearman ρ | no — `scipy.stats` against human rankings | — |

The bottom three call `_fetch_completion(messages, model)` to **perform** the task; the prediction is
then scored by ARI, F1 or a rank correlation. Their `--model GPT4` is the subject, not the grader.
· Source: `.../documentation/demo_segmentation/main.py:248, 328`;
`.../knowledge_transfer/demo_validation/main.py:69, 155`; `.../improvement/sop_ranking/main.py:42, 59-66`

This **corrects our own guide**, which recorded Wonderbread as *"Yes — explicitly for QA; verify two
more subtasks"* on the basis of the paper alone. With the repo on disk the answer is settled.

### Judge 1 — Question Answering

120 items in `data/qa_dataset.csv`. **Four judge calls per item = 480 calls per model evaluated.**

**A1 Grading model.** The code passes the literal label `'GPT4'`, which the router resolves to
`gpt-4-0125-preview` for text-only prompts (`gpt-4-vision-preview` if any message carries an image;
QA judging is text-only). `temperature=0.0`, `max_tokens=4096`, system message `SYSTEM_PROMPT`
prepended. Recording the resolved id matters — the string "GPT-4" appears nowhere in the request.
· Source: label at `.../question_answering/eval.py:33, 42, 51, 60`; resolution at
`wonderbread/helpers.py:487-494`; request at `wonderbread/helpers.py:454-462`

**A2 Material shown to the model.** Differs by criterion, and the asymmetry is the point.

| Criterion | Question | Tested answer | Human Label (gold) | Screenshots / trace / SOP |
|---|:--:|:--:|:--:|:--:|
| completeness | ✅ | ✅ | ✅ | ❌ |
| soundness | ✅ | ✅ | ✅ | ❌ |
| clarity | ✅ | ✅ | **❌** | ❌ |
| compactness | ✅ | ✅ | **❌** | ❌ |

**Withheld from every criterion:** the screenshots, keyframes, action trace, SOP, and the web
application itself. The prompt says so — *"You won't be provided with information about the web
application"* — so the judge cannot check a claim against the demonstration; it can only compare
against the human label, and for clarity and compactness not even that. **Two of the four criteria
are judgements about prose made with no access to ground truth.**

A copy-paste artifact worth knowing: the clarity and compactness prompts both keep the sentence
*"…but only the question, **human label**, and response"* although neither interpolates a human
label, and compactness's third few-shot example prints a `Human Label:` line anyway. The judge is
told a field exists that it is not given, and shown one example where it was.
· Source: `.../question_answering/eval.py:27-61`; `.../question_answering/prompts.py:102-107, 149-155, 173`

**A3 Human comparison. Yes, and it is on disk** — the only judge in this suite with one. A 30-item
sample was scored by humans and by the GPT-4 judge. Recomputed from the committed CSVs on 2026-08-19
(Spearman ρ on ranks, plus exact-agreement rate, since the scale has only three points):

| Criterion | Spearman ρ | Pearson r | Exact agreement | human mean | judge mean |
|---|---:|---:|---:|---:|---:|
| completeness | 0.860 | 0.842 | 86.7% | 1.60 | 1.73 |
| soundness | 0.876 | 0.915 | 93.3% | 1.33 | 1.40 |
| clarity | 0.802 | 0.802 | 96.7% | 1.10 | 1.07 |
| compactness | 0.886 | 0.927 | 96.7% | 1.17 | 1.20 |

n = 30. **Who the humans were is not stated** — no annotator description, no instructions issued to
them, no inter-human agreement, so there is no way to tell whether 0.86 sits near the human-human
ceiling or far below it. The judge is mildly harsher than the humans on the two criteria that see the
gold label and indistinguishable on the two that do not. The prompts carry few-shot examples with
scores; nothing states they were drawn from this sample.
· Source: `data/experimental_results/question_answering/human_correlation/{qna-human-sample-final.csv,
qna-GPT4-sample-final.csv}`; the authors' own `get_corrs.py` (Spearman + Pearson, no agreement stat)

**A4 Grading instructions.** Four variants, one per criterion, no shared system prompt beyond
`SYSTEM_PROMPT`. Verbatim in [Appendix A](#appendix-a--wonderbread-qa-prompts-verbatim-4-variants).
· Source: `.../question_answering/prompts.py:1-191`

**A5 Scoring criteria.** Four, scored by **four separate API calls**, never jointly — so no priority
is expressed and the judge never trades one against another.
*completeness* (fully answers, covers relevant information, all aspects) · *soundness* (accuracy, no
unsupported assumptions, logical) · *clarity* (unambiguous, needs no clarification, one
interpretation) · *compactness* (short, no unnecessary information).

Each function's docstring adds *"Please score this metric harshly"* — but that lives in the Python
docstring and **is never sent to the model**. Easy to mistake for an instruction.
· Source: `.../question_answering/eval.py:27-61` (docstrings) vs `prompts.py` (actual prompt text)

**A6 Score meanings.** Integer 1–3. **Direction: 1 is best, 3 is worst — lower is better.** Every
point carries a written definition in every prompt. This is inverted relative to the intuitive
reading of a results table, and inverted relative to Judge 3's scale in the same repository (D2). The
authors' own radar plot fixes the radial axis to `range=[0, 3]` without noting it, so a **smaller
polygon is the better model**.
· Source: `prompts.py:14-17, 65-68, 114-117, 160-163`; radar axis at
`wonderbread/benchmark/eval/run_question_answering.py:38-40`

**B1 Type of judgment.** Absolute score of a single answer on a 3-point ordinal scale, four times
independently. No pairwise comparison, no ranking. The judge is told to return the number and nothing
else, so **no explanation is captured** — the few-shot examples show explanations, but the output
format forbids them.
· Source: `.../question_answering/eval.py:63-75`; `prompts.py:48, 99, 145, 190`

**B2 Repeated grading.** One call per (item, criterion) — 4 per item, 480 per model. No repeats, no
self-consistency check, no tie rule, because nothing is graded twice. At `temperature=0` this makes
judge variance invisible rather than absent.
· Source: `.../question_answering/eval.py:63-75`

**B3 Use of a correct answer.** The `Human Label` column of `data/qa_dataset.csv` is the reference,
shown to completeness and soundness and **withheld from clarity and compactness** (A2). The repo does
not document who wrote the labels or how they were validated.
· Source: `.../question_answering/eval.py:65-68`; `data/qa_dataset.csv` columns
`Task ID(s), Question Instantiation, Evidence, Human Label, Question Template`

**B4 Final score calculation.** The four scores are appended as four columns and **never combined
into a composite**. The reported figure is a per-criterion **micro** mean over items, per model:

```
per item   : four independent 1-3 scores
per model  : mean of each criterion over all 120 items   # micro, equal weight per item
reported   : four numbers per model; no single headline score
```

Ties: not applicable. **Failed or skipped judgments: the string `"NA"` is written into an otherwise
integer column** whenever the response is NaN or shorter than 5 characters — a silent skip.
`df[cols].mean()` will either drop those rows from the denominator or raise, depending on how pandas
types the column, and nothing counts them. On the shipped 480-row result set no `"NA"` survives (all
values are 1/2/3), so the published numbers are unaffected; a run against a model that emits empty
answers would not be.
· Source: `.../question_answering/eval.py:29-30, 85-95`; aggregation at
`wonderbread/benchmark/eval/run_question_answering.py:22-33`; distribution counted from
`data/experimental_results/question_answering_all_results.csv` 2026-08-19

**C1 Generation settings.** *Judge:* `gpt-4-0125-preview`, `temperature=0.0`, `max_tokens=4096`,
`SYSTEM_PROMPT` prepended, **sequential** (a plain `for` loop, no concurrency), **no seed**. Retry:
`RateLimitError` sleeps 60 s and retries by **unbounded recursion** — no attempt cap, no backoff
growth, so a sustained 429 recurses until the stack limit. Any other `APIError` is re-raised and
kills the run mid-file, and because the CSV is written only after the loop completes, **a failure at
row 119 of 120 loses all judge output.** Our run must checkpoint per row.
*Tested model:* answers are generated separately and read from the input CSV's `Response` column; the
answer prompt caps length with *"Please limit your response to 2-3 sentences or less"*, which
directly shapes the compactness score.
· Source: `wonderbread/helpers.py:454-471, 487-494`; `.../question_answering/eval.py:77-95`;
`prompts.py:275-280`

**C2 Output handling.** Required format is the bare number — *"Return only the number corresponding
to the rating, nothing else."* **Extraction rule: none.** The raw completion string is stored
directly as the score — no regex, no cast, no range check, no malformed-output branch. The functions
are annotated `-> int` and return `str`. A judge replying `"Score: 2"` puts that text into the score
column, silently turning it into an object column and corrupting every downstream `.mean()`. The only
guard is the pre-call length skip that emits `"NA"` (B4).
`our choice: strict parse to {1,2,3} + parse_fail counter (2026-08-19, planner)`.
· Source: `.../question_answering/eval.py:27-61`; `prompts.py:48, 99, 145, 190`;
project rule in `JUDGE_DOCUMENTATION_RULE.md` §C2

**C3 Software and access.** `HazyResearch/wonderbread@ed052c6` (2025-07-30, "Update README.md"),
vendored 2026-08-19; package name `wonderbread` (`setup.py`). Access via the `openai` client with the
default `OPENAI_API_KEY`. **Read by this project on 2026-08-19; never executed here.** The committed
`question_answering_all_results.csv` (480 rows: 120 items × GPT4 / GeminiPro / Claude3 / Human) is
the authors' output, not ours.
· Source: `setup.py`, `wonderbread/helpers.py:454-456`, `VENDORED_SOURCES.md`

### Judge 2 — SOP Generation (the "semantic" Precision / Recall)

Not obvious from the metric name, and the most consequential finding in this record: **Precision,
Recall and Ordering for SOP Generation are not string metrics. Each is a tally of GPT-4 entailment
decisions.**

**A1 Grading model.** `gpt-4-1106-preview` — a **different snapshot from Judge 1's
`gpt-4-0125-preview`** — hard-coded as a module constant and reached through the JSON-mode helper.
Two judges in one repository on two GPT-4 snapshots; "Wonderbread uses GPT-4" would hide that.
· Source: `.../sop_generation/eval_completion.py:15, 92`; `wonderbread/helpers.py:512-527`

**A2 Material shown.** One line of one SOP (the "Query"), and the *entire other* SOP as an indexed
list of lines. Nothing else — no screenshots, no action trace, no task intent. The predicted SOP and
the gold SOP each take a turn as Query and as List.
· Source: `.../sop_generation/metrics.py:64-96`; `.../sop_generation/eval_prompts.py:73-104`

**A3 Human comparison.** `not reported` — no human-agreement study for the entailment decision,
unlike Judge 1.
· Source: `not stated upstream`

**A4 Grading instructions.** One variant, no few-shot examples, no system message beyond
`SYSTEM_PROMPT`. Verbatim in [Appendix B](#appendix-b--wonderbread-sop-generation-entailment-prompt).
· Source: `.../sop_generation/eval_prompts.py:44-104`

**A5 Scoring criteria.** One — semantic entailment of a step's *primary objective*. Not correctness,
and not ordering (ordering is computed from the returned indices afterwards, not judged).

**A6 Score meanings.** An index, not a scale. `index != -1` means "this line is covered by the other
SOP"; `-1` means not covered. No partial credit, no confidence.
· Source: `.../sop_generation/metrics.py:97-152`

**B1 / B2 Judgment and repeats.** A mapping/entailment decision, **one call per line, in both
directions**. For a predicted SOP of *p* lines against a gold SOP of *g* lines that is **p + g calls
per demonstration** — an order of magnitude more judge traffic than QA, scaling with SOP length, so a
verbose model costs more to score than a terse one. No repeats.

**B3 Use of a correct answer.** The human-written Gold SOP, one per demonstration, shipped as
`SOP - <demo>.txt`.

**B4 Final score calculation.**

```
precision = |{pred lines whose index != -1}| / len(pred_sop)
recall    = |{gold lines whose index != -1}| / len(gold_sop)
ordering  = |{pred lines mapped to a non-decreasing gold index}| / len(pred_sop)
```

Both denominators are line counts **after `preprocess_sop`**, which splits on newlines, drops empties,
and **strips everything up to and including the first `.` in each line** — intended to remove the
`"1."` numbering, but it also truncates any step containing an earlier period. A model writing
`"Click Save. Then confirm."` contributes the line `"Then confirm."`. That reshapes both the
denominator and what the judge is asked to match.
· Source: `.../sop_generation/eval.py:19-45`; `.../sop_generation/metrics.py:97-178`

**C1 / C2 Settings and output handling.** JSON mode (`response_format={"type":"json_object"}`),
`max_tokens=4096`, **no temperature passed — so the API default of 1.0 applies, and Judge 2 is not
deterministic while Judge 1 is** (D3). Malformed output is handled properly for once: a completion
that does not parse as JSON with an `index` key is regenerated **once**, and a second failure raises
rather than defaulting. `completion["index"]` is then cast with `int()`, which raises on a non-numeric
index.
**Caching changes what a "run" means.** Completions are keyed by
`sop_cache/<cache_id>/<prompt_name>.json` and reused unless `force_fetch=True`; `cache_id` is built
from experiment/demo/ablation names, so re-running after editing a prompt returns the **old**
completions.
`our choice: pass temperature=0 and clear sop_cache per configuration (2026-08-19, planner)`.
· Source: `.../sop_generation/eval_completion.py:14-15, 43-81, 92`; `wonderbread/helpers.py:512-527, 457-460`

**C3 Software and access.** As Judge 1. Never executed here.

### Judge 3 — SOP Improvement rubric (present, does not run)

Our guide flagged this as the open question: the PDF says SOP Improvement uses a 1–5 rubric but never
says who applies it. **Answer: `gpt-4-turbo` applies it, in `wonderbread/benchmark/eval/eval_rubric.py`
— and that file cannot execute.** Recorded in full because "the scorer exists" and "the scorer runs"
are different facts, and only the second licenses a number.

**A1** `gpt-4-turbo`, `temperature=0`, `max_tokens=4096`, `response_format={"type":"json_object"}`,
called through a bare `openai.OpenAI()` client rather than the shared helper. A **third** GPT-4
snapshot in the same repo. · Source: `eval_rubric.py:165-171`

**A2** The SOP text only. The signature is `rubric_evaluation_prompt(sop, gold_sop)` and the call
site passes `row["gold_sop"]`, but **the parameter is never interpolated into the prompt** — the gold
SOP is accepted and discarded. The judge scores an SOP against a written rubric with no reference and
no demonstration. · Source: `eval_rubric.py:24-71, 148`

**A3** `not reported`.

**A4** System message *"You are a steps of procedure (SOP) evaluator. Please evaluate the following
SOP based on the rubric provided. Do not add quotes around the explanation text"*, then a rubric of
six named criteria (Element Specification, Action Type, Edge Case Coverage, Discrete Action, Action
Relevance, Generality), one worked high-quality SOP example with commentary, and a JSON output block.
· Source: `eval_rubric.py:24-71, 149-164`

**A5** The six above. The prompt asks for a 0/1 sub-score per rubric item inside the free-text
`explanation`, so the sub-scores are generated but **never parsed and never stored**.
· Source: `eval_rubric.py:40-41`

**A6** 1–5. **Direction is contradictory across the repository — see D2.** This copy says *"1 (worse)
to 5 (best)"*; the near-identical prompt at `.../sop_generation/eval_prompts.py:22` says *"1 (best)
to 5 (worst)"*. No per-point definitions in either.

**B1–B4** Absolute score, one call per SOP, run for two SOPs per demonstration (`base_sop` and the
self-improved `sop_1`) so the improvement is their difference. No repeats, no reference (A2). No
aggregation code — scores are written per demo to a CSV and combined by hand.
· Source: `eval_rubric.py:134-194`

**C2** ` ```json ` fences are stripped, then the string is passed to Python's **`eval()`** — not
`json.loads()` — and a bare `except:` prints `"Error in response"` and moves on, leaving that
demonstration with **no score key at all** while still appending the row. Malformed judgments vanish
from the denominator without being counted, exactly the failure mode this project's rule forbids.
`eval()` on model output is also arbitrary code execution.
`our choice: json.loads + parse_fail counter, never eval() (2026-08-19, planner)`.
· Source: `eval_rubric.py:172-185`

**C3** As above. Never executed here — and, per D1, not executable.

### Wonderbread — discrepancies and decisions

**D1 · `eval_rubric.py` cannot run, in four independent ways.** Any one of these stops it:
1. `from eval_prompts import rubric_evaluation_prompt` (line 10) — `eval_prompts.py` does not exist in
   `wonderbread/benchmark/eval/`; it lives in `.../documentation/sop_generation/`. `ImportError`.
2. `from workflows.helpers import ...` (line 17) — there is no `workflows` package anywhere in the
   repo; the package is `wonderbread`. `ImportError`.
3. `evaluate_sops_all` is defined with three required parameters (lines 74-78) and called with two
   (line 241). `TypeError`.
4. Line 24 redefines `rubric_evaluation_prompt`, shadowing the import — so even with the imports
   repaired, the six-criterion / "1 (worse) to 5 (best)" version wins and the seven-criterion /
   "1 (best) to 5 (worst)" version including *Faithfulness to Demonstration* is dead code.

It also hard-codes `model_name = "Claude3"` (line 95) with the other model filters commented out, so
evaluating a different model means editing source. **No SOP-Improvement rubric number can be
reproduced from this commit without rewriting the file.**
`our choice: treat SOP Improvement as not runnable and exclude it from scope (2026-08-19, planner)`.
· Source: `wonderbread/benchmark/eval/eval_rubric.py:10, 17, 24, 74-78, 95, 241`

**D2 · The 1–5 rubric's direction is inverted between two copies of the same prompt.**
`.../sop_generation/eval_prompts.py:22` — *"a score from 1 (best) to 5 (worst)"*.
`.../eval/eval_rubric.py:34` — *"a score from 1 (worse) to 5 (best)"*.
Same rubric, same benchmark, opposite polarity; the two also disagree on the criteria list (the first
includes *Faithfulness to Demonstration*, the second drops it). A number produced under one and read
under the other inverts every conclusion. The live path is `eval_rubric.py`'s — higher is better —
but since that path does not execute (D1), **no direction can be treated as established**; whichever
scorer we write must state its own and say so in `Results.xlsx`.

**D3 · Judge 1 is deterministic, Judge 2 is not.** QA passes `temperature=0.0` explicitly;
`fetch_openai_json_completion` passes no temperature, so SOP-Generation entailment runs at the API
default of 1.0. Precision/Recall/Ordering therefore carry run-to-run variance the QA numbers do not,
and nothing in the repo reports it.
`our choice: pass temperature=0 to Judge 2 and note the deviation when reporting (2026-08-19, planner)`.
· Source: `wonderbread/helpers.py:487-494` vs `512-527`

**D4 · Three GPT-4 snapshots in one benchmark.** `gpt-4-0125-preview` (QA), `gpt-4-1106-preview` (SOP
generation), `gpt-4-turbo` (rubric). All three are retired aliases, so an exact reproduction of the
authors' numbers is no longer possible at any price and a substitute judge must be declared under
[Substituting a different judge model](#substituting-a-different-judge-model) above.
· Source: `helpers.py:490`, `eval_completion.py:15`, `eval_rubric.py:169`

**D5 · Judge-call cost is dominated by SOP Generation, not QA.** QA is a flat 480 calls per model.
SOP Generation is `p + g` calls **per demonstration**, so its cost scales with how verbose the tested
model is — the metric gets more expensive exactly when the model does worst on compactness. **Any
budget written from "one call per item" will be wrong for this benchmark.**

---

## 2 · MultiChallenge

Scope: **all 273 items are judged**, one judge call per item per attempt. There is no unjudged
portion and no gold answer anywhere in the benchmark — the rubric *is* the label.
Sources read: `ekwinox117/multi-challenge@5ccefcca6a39020d66c1383c4e6a809cb07afa33` (2025-02-05,
"fixed axis names, minor bugs."), vendored at `Action_processes_benchmarks/Multi-challenge_bench`
with `.git` stripped, plus the repo README. **Code read, not executed** — see D1 for why it cannot be.

Dataset counted from `data/benchmark_questions.jsonl` on 2026-08-19:

| Axis | Items |
|---|---:|
| `INFERENCE_MEMORY` | 113 |
| `INSTRUCTION_RETENTION` | 69 |
| `SELF_COHERENCE` | 50 |
| `RELIABLE_VERSION_EDITING` | 41 |
| **total** | **273** |

Conversations run 3–19 messages. `PASS_CRITERIA` is the string `"YES"` on all 273 rows.

**A1 Grading model.** `gpt-4o-2024-08-06`, OpenAI `client.beta.chat.completions.parse` (the
structured-output endpoint), `temperature=0`. Hard-coded in the evaluator — not a CLI flag, so
changing the judge means editing source.
· Source: `src/evaluator.py:30-35`; endpoint at `src/models/openai.py:31-37`

**A2 Material shown to the model.**
- **Shown:** the tested model's final response, and the item's `TARGET_QUESTION`.
- **Withheld:** the conversation history, the final user turn, any reference answer, the axis label,
  and `PASS_CRITERIA`. There is no system message — the whole prompt is one `user` turn.

The naming is misleading and worth stating plainly: the field called `TARGET_QUESTION` **is the
rubric**, and it is what lands in the prompt's `<CRITERIA>` slot. Example row: *"Are the restaurants
chosen within a 5-minute walk from the UN headquarters?"* The field called `PASS_CRITERIA` is not a
criterion at all — it is the expected verdict string.

**This contradicts our own guide**, which says to give the judge "the dialogue, final user request,
model answer, and instance rubric". The code gives **only** the answer and the rubric. Judging a
19-turn conversation's final answer without the conversation is a materially narrower task, and any
rubric that needs the history to be checkable is effectively being guessed at. See D5 for which one
our run follows.
· Source: `src/evaluator.py:40-43` (`JUDGE_PROMPT.format(response, target_question)`),
`src/conversation.py:9-10`, `src/models/openai.py:23-24`

**A3 Human comparison.** `not reported`. Neither the README nor any file in the vendored tree reports
agreement between the GPT-4o judge and human graders — no sample size, no κ, no correlation. The
rubrics are human-written, but that makes the *labels* human, not the *grader* validated. The prompt
carries no few-shot examples, so the question of whether few-shots came from human labels does not
arise.
· Source: `not stated upstream` (grep for `human|agreement|kappa|annotat` over the tree returns only
the README abstract)

**A4 Grading instructions.** One variant, no system message, no few-shot examples. Verbatim in
[Appendix C](#appendix-c--multichallenge-judge-prompt).
· Source: `src/evaluator.py:11-24`

**A5 Scoring criteria.** **None are named.** The judge is given the item's own rubric and told
`Be VERY STRICT!` (capitalised and exclaimed in the source). There is no rubric of named qualities —
correctness, relevance and the rest are whatever the instance-specific question happens to ask about.
This is deliberate: the README states that a generic "is this a good answer?" rubric would be invalid.
· Source: `src/evaluator.py:19`

**A6 Score meanings.** Binary. The judge returns `verdict ∈ {"YES","NO"}`, enforced by a
`Literal["YES","NO"]` field on a pydantic model, so an off-vocabulary verdict is impossible at the
API layer. **Direction is per-item, not global:** an item passes when `verdict == PASS_CRITERIA`, so
a row whose `PASS_CRITERIA` were `"NO"` would pass on a `"NO"` verdict. On the data as shipped all
273 rows carry `"YES"`, so today the mapping is simply YES = pass — but a future data update adding a
`"NO"` row would silently invert one item's direction, which is why this is recorded rather than
simplified. Neither point carries a definition beyond the rubric text.
· Source: `src/evaluator.py:7-9` (schema), `:78` (`passed = verdict == pass_criteria`);
`PASS_CRITERIA` distribution counted 2026-08-19

**B1 Type of judgment.** Binary pass/fail on a single answer against an instance-specific rubric. No
pairwise comparison, no ranking. The judge emits free-text `reasoning` alongside the verdict; the
reasoning is stored in the raw CSV and **never used in scoring**.
· Source: `src/evaluator.py:7-9, 44`; `src/result_parser.py:44`

**B2 Repeated grading.** **One judge call per (item, attempt).** Repetition happens on the
*generation* side, not the grading side: `--attempts k` samples k responses from the tested model
(default 1) and each is judged once. Combination across attempts is **any-pass** — an item counts as
passed if at least one attempt passed, i.e. **pass@k, not mean-of-k**. There is no
disagreement-resolution rule because the same response is never graded twice; judge variance is
untested and, at `temperature=0` with one call, unmeasured.
· Source: `main.py:31-32`; `src/data_loader.py:39-51`; `src/evaluator.py:99-107`;
`src/result_parser.py:27-30`

**B3 Use of a correct answer.** **No reference answer exists.** MultiChallenge ships no gold
responses — the human-written binary rubric replaces one. The judge is therefore never shown a
reference, and there is no asymmetry across criteria because there is only one criterion. Rubric
provenance: human-authored per instance by the benchmark authors; the repo documents no validation
procedure for the rubrics themselves.
· Source: `data/benchmark_questions.jsonl` schema (`QUESTION_ID, AXIS, CONVERSATION, TARGET_QUESTION,
PASS_CRITERIA` — no answer field); `src/data_loader.py:20-26`

**B4 Final score calculation.**

```
per item  : passed      = (verdict == PASS_CRITERIA)
per item  : item_passed = any(passed over that item's attempts)      # any-pass
per axis  : axis_score  = 100 * (# items passed in axis) / (# items in axis)
headline  : overall     = mean(axis_score over axes)                 # MACRO, unweighted
```

The headline is a **macro average over the four axes, not a micro average over the 273 items.** The
axes are unequal — 113 / 69 / 50 / 41 — so `RELIABLE_VERSION_EDITING`'s 41 items carry the same
weight as `INFERENCE_MEMORY`'s 113. Item-level micro accuracy and this number are different
quantities and differ whenever performance varies by axis.

Ties: not applicable. **Failed judgments: counted as failures, never dropped.** Two distinct paths
both write `verdict='NO', passed=False` — an item whose `question_id` has no response
(`reasoning='NA - Question ID not found'`), and any exception during the judge call
(`reasoning='Error during evaluation: ...'`). The second also writes `axis='NA'`, with a consequence
the code does not intend — see D2.
· Source: `src/result_parser.py:9-38`; `src/evaluator.py:52-61, 80-90`

**C1 Generation settings.**
- *Judge:* `gpt-4o-2024-08-06`, `temperature=0`, structured output via pydantic `JudgeResponse`, one
  call per (item, attempt), concurrency `--max-workers_eval` (default 1). **No seed, no retry, no
  timeout** — an exception is caught once and converted into a `NO` verdict rather than retried.
  `max_tokens=4096` is passed at the call site but never reaches the API (D1).
- *Tested model:* whatever `--model-provider` / `--provider-args` supply; `--attempts` responses per
  item, concurrency `--max-workers_response_gen` (default 1). A generation exception is swallowed and
  the literal string `"Error generating response for question_id N: <err>.\n FAIL THIS QUESTION"` is
  stored **as the response** and sent to the judge — so **a provider outage is scored as a model
  failure, not a harness failure.** Pre-count that string before trusting a score.
· Source: `src/evaluator.py:30-35`; `main.py:31-36`; `src/models/openai.py:30-37`;
`src/data_loader.py:44-51`

**C2 Output handling.** Structured output, so extraction is a schema and not a regex — the API
returns a parsed `JudgeResponse` and `verdict` is already constrained to `YES`/`NO`. There is no
malformed-response path at the parsing layer. A refusal, a length stop, or any API error surfaces as
an exception and is converted to `NO` / `passed=False`, i.e. **silently scored as a failed item** —
violating this project's standing rule that call and parse failures are their own category. Our run
adds that counter; the `reasoning` field already distinguishes the cases, so the count is recoverable
from the raw CSV without changing the score path.
· Source: `src/evaluator.py:7-9, 43, 80-90`; project rule in `JUDGE_DOCUMENTATION_RULE.md` §C2

**C3 Software and access.** `ekwinox117/multi-challenge@5ccefcc`, vendored 2026-08-19. Pinned deps:
`openai==1.53.0`, `pydantic==2.10.6`, `python-dotenv==1.0.1`, `tqdm==4.66.2`, `transformers==4.44.1`,
`torch==2.5.1`. Access via the OpenAI API with `OPENAI_API_KEY` from a `.env` at repo root. **Read on
2026-08-19; never executed here.** No MultiChallenge run has been submitted from this repo and no
number from it is in `Results.xlsx`.
· Source: `requirements.txt`, `README.md` setup section, `VENDORED_SOURCES.md`

### MultiChallenge — discrepancies and decisions

**D1 · The vendored judge harness cannot run.** `Evaluator.__init__` constructs the judge with
`max_tokens=4096`, but `OpenAIModel.__init__` accepts only `(self, model, temp, response_format)` and
takes no `**kwargs`. Python raises `TypeError: __init__() got an unexpected keyword argument
'max_tokens'` before any API call. Verified by AST comparison of the signature against the call site
on 2026-08-19. This is not a subtle path — it fires on the first line of evaluation for every
invocation. Anyone reporting a MultiChallenge number from this commit either patched it or used a
different checkout.
`our choice: remove the kwarg rather than add the parameter (2026-08-19, planner)` — so the request
stays byte-identical to what the authors' working version must have sent; the judge writes a short
reasoning string plus a token, and 4096 was never binding.
· Source: `src/evaluator.py:33` vs `src/models/openai.py:9`

**D2 · An exception during judging silently corrupts the headline number.** The error handler writes
`axis='NA'`. `ResultParser` builds `axis_counts` from whatever axis strings it sees, so `'NA'` becomes
a **fifth axis with its own 0% score**, and `overall_score = sum(axis_scores)/len(axis_scores)`
divides by 5 instead of 4. **A single transient API error drops the reported score by roughly a fifth
of itself** — for a model scoring 60%, to about 48% — with nothing in the summary file indicating why.
`our choice: exclude axis == 'NA' from axis_counts and report a separate judge_error_count
(2026-08-19, planner)`.
· Source: `src/evaluator.py:84` vs `src/result_parser.py:18-33`

**D3 · The recorded attempt index is wrong whenever `--attempts > 1`.** `evaluator.py:74` writes
`'attempt': j`, but `j` is the inner loop variable of the *submission* loop and has already reached
its final value by the time the *collection* loop runs, so every row is stamped with the last attempt
index. The same leak affects the exception handler, which reads `self.conversations[i]` with a stale
`i` and attributes an error to the wrong `question_id`. Neither changes `overall_score` — scoring
reads `passed`, not `attempt` — but both make the raw CSV untrustworthy for per-attempt analysis.
`our choice: leave upstream behaviour, do not use the attempt column (2026-08-19, planner)`.
· Source: `src/evaluator.py:63-66, 68-79, 82-90`

**D4 · The axis names in the code's own docstring do not match the data.** `conversation.py:7` lists
`REFINEMENT, EXPLICIT IF, COHERENCE, RECOLLECTION`; the JSONL ships `INFERENCE_MEMORY,
INSTRUCTION_RETENTION, SELF_COHERENCE, RELIABLE_VERSION_EDITING`. The upstream commit message ("fixed
axis names") suggests the docstring is the stale half. Harmless — nothing reads it — but it is why a
reader may map our axis labels onto the paper's four categories incorrectly.
· Source: `src/conversation.py:7` vs `data/benchmark_questions.jsonl`

**D5 · Code and our guide disagree on what the judge sees (A2).** **Our run follows the code**,
because the published MultiChallenge numbers were produced by the code, and adding the conversation
to the prompt would produce numbers not comparable to anyone else's. The guide is being corrected,
not the code. `our choice: follow upstream code (2026-08-19, planner)`.

**D6 · `PASS_CRITERIA` is vestigial on the data as shipped** — 273/273 rows are `"YES"`, so the
comparison at `evaluator.py:78` always reduces to `verdict == "YES"`. Kept in our runner anyway; see
A6 for why removing it would be a silent trap if the dataset is ever updated.

---

## 3 · AwareBench

Scope: **60 of 4,075 rows are judged** — the `mission_open-ended` dimension, and nothing else. The
other 4,015 rows carry an exact-match key on disk. The section below argues that excluding them is a
decision with a cost attached, not an omission.
Sources read: `HowieHwong/Awareness-in-LLM@07598ff`, vendored at
`Transition_processes_benchmarks/Awareness_in_LLM`; the paper *I Think, Therefore I am: Benchmarking
Awareness of Large Language Models Using AwareBench* (arXiv 2401.17882v2); and this project's
`AWARENESS_NOTES.md` and `Output_template/README.md`.

**The judge code is not in this repository, and neither is the judge prompt.** Upstream's README
routes evaluation to the external `trustllm` package (`from trustllm import ethics;
ethics.EthicsEval().awareness_eval(...)`), which is not vendored here and was not installed. Every
field below is therefore sourced to the paper, to our own derivation, or marked `not stated upstream`
— and three are marked **blockers**, meaning the judge cannot be run faithfully until they are
cleared. That is the honest state of this benchmark and the reason the record is written before the
run rather than after.
· Source: `README.md:20-86` (upstream evaluation section); `AWARENESS_NOTES.md` §2.4

### Why 4,015 of the 4,075 rows need no judge

The rule from the top of this file, applied: the judge goes exactly where there is no key — and on
AwareEval that is a countable set. Counted from `dataset/AwareEval.json` on 2026-08-19:

| Dimension | Rows | Answer key on disk | Scored by |
|---|---:|---|---|
| `perspective` | 1,400 | `label` (`A`/`B`/`C`) on 900 MCQ rows; a normalised noun on 500 story rows | accuracy / exact match |
| `mission_explicit` | 966 | `label`, `A`/`B`/`C`, 322 each | accuracy |
| `capability` | 600 | `label`, `A`/`B`, 300 each | accuracy |
| `culture` | 522 | `correct` / `wrong`, 264/258 | binary accuracy |
| `mission_implicit` | 327 | `Label` — **capitalised** | accuracy |
| `emotion` | 200 | `"1"`–`"4"` as strings | accuracy |
| **`mission_open-ended`** | **60** | **none — no `label`, no `choices`** | **GPT-4 judge** |
| total | 4,075 | 4,015 keyed | |

The 60 open-ended rows are the only ones carrying just `content` and `section` and no answer field.
**The split is a property of the data, not a choice we made about it.**

**Four reasons the 4,015 stay off the judge:**

1. **They already have a right answer.** Routing a keyed row through a judge replaces a deterministic
   score with an estimate of the same quantity — strictly worse, never better.
2. **It would break comparability.** The paper's 13-model table is accuracy on these rows. A
   judge-scored version is a different measurement and could not be placed beside it, costing us the
   only external sanity check this benchmark has.
3. **The cost is 67×.** 60 rows × 2 evaluator prompts = 120 judge calls per model. Judging all 4,075
   under the same two prompts would be 8,150 — spent to make four columns noisier.
4. **The failure it would mask is a parser bug, not a scoring gap.** The one honest argument for a
   judge on keyed rows is "our parser cannot extract the letter". The answer is to fix the parser and
   count extraction failures as their own category — already this project's standing rule — not to
   hire a model to guess what the response meant.

**What the exclusion costs.** `mission_implicit`'s key is under `Label` and `Choices` with capital
letters while every other dimension uses lowercase. A loop written against the majority spelling
returns `None` for all 327 rows, scores every one wrong, and drops the headline by about 8 points —
and the first version of our own notes fell for exactly this, recording that those rows had no key at
all. **Deciding "these 4,015 are keyed, so they need no documentation" is only safe because the keys
were opened and counted**; that check is what licenses the exclusion, and it is why it is written down
rather than assumed.
· Source: `dataset/AwareEval.json`, counted 2026-08-19; `AWARENESS_NOTES.md` §2.4 trap 1, §2.5

**One consequence that makes the 60 rows matter far more than 1.5% of the data.** The paper's
aggregation weights dimensions equally, not by item count:

```
mission_avg       = mean(mission_explicit, mission_implicit, mission_open_ended)
awareness_overall = mean(capability, mission_avg, emotion, perspective, culture)
```

so `mission_open-ended` carries **1/15 of the headline** — the same weight as `mission_explicit`'s
966 rows. A judge wobble on 60 rows moves the reported score as much as a real 16-point capability
shift on 966. **The judged fraction is 1.5% of the items and 6.7% of the number.** That asymmetry is
why this record exists at all.
· Source: aggregation derived and verified in `AWARENESS_NOTES.md` §2.6 — it reproduces GPT-4's
Figure 4 score (89.02) and the 13-model average (65.69) exactly

**How the excluded rows are scored instead:** exact match against the key on disk — `label`, `Label`
for `mission_implicit`, `correct`/`wrong` for `culture`, `"1"`–`"4"` for `emotion`, and a normalised
noun for the 500 `perspective` story rows. The scorer is ours, written locally against paper §5.1
because `awareness_eval()` is not vendored.

### The judged 60 rows

**A1 Grading model.** GPT-4. The paper names no snapshot and no endpoint; the resolving code is in
`trustllm`, which is not vendored. TrustLLM's machine-ethics evaluation generally uses a GPT-4-turbo
judge alongside a `LibrAI/longformer-harmful-ro` classifier, but **which of the two touches the
awareness rows is not determinable from anything published here.** Every GPT-4 snapshot of that era
is retired, so exact reproduction is impossible regardless; our run declares a substitute under
[Substituting a different judge model](#substituting-a-different-judge-model) above.
· Source: `not stated upstream` — paper §5.1 says "GPT-4"; `README.md:72-86` delegates to `trustllm`

**A2 Material shown to the model.** The item's `content` — a persuasive prompt arguing for an
AI-interest position — and the tested model's free-text response. **Withheld:** any reference answer
(none exists), the `section` label naming the persuasion strategy, and the rest of the dataset. The
exact prompt assembly is in `trustllm` and is not verifiable here; this is read off the paper's
Figures 8–10.
· Source: paper Figures 8, 9, 10; `Output_template/README.md` § "The judge column"

**A3 Human comparison.** `not reported`. The paper validates *construction* with human-AI quality
checks — multiple-choice labels were fixed by permuting option order and keeping only answers that
survived permutation, with the research team adjudicating the rest — but that validates the **labels
on the 4,015 keyed rows**, not the judge on the 60. No agreement statistic between GPT-4 and human
graders on `mission_open-ended` is reported anywhere. Notably, the paper's own stated reason for
running two evaluator prompts is that *"prompt-induced randomness can affect GPT-4's evaluation
results"* — an admission of judge instability with no measurement of it against humans.
· Source: paper §4 (construction), §5.1 and Table 1 note; `AWARENESS_NOTES.md` §2.2, §2.3

**A4 Grading instructions. Not recoverable from this repository — blocker 1.** The prompts exist as
three figures in the paper (Figure 8: multi-dimension generation quality; Figure 9: standard
human-alignment evaluator; Figure 10: role-playing human-alignment evaluator) and as source in
`trustllm`. Per rule 2 — prompts are copied verbatim, never paraphrased — no prompt text is
reproduced here until it is transcribed from one of those two sources. A paraphrase would be a
different prompt producing different scores, and writing one into this record would make the record
actively misleading.

**To clear this blocker:** transcribe Figures 8–10 from arXiv 2401.17882v2 into the blocks below,
then cross-check against `trustllm`'s source for the placeholders and any system message the figures
omit. Where figure and package disagree, both go in and our run states which it follows.

```text
[Figure 9 — standard evaluator prompt: NOT YET TRANSCRIBED]
```
```text
[Figure 10 — role-playing evaluator prompt: NOT YET TRANSCRIBED]
```
```text
[Figure 8 — multi-dimension generation-quality prompt: NOT YET TRANSCRIBED]
```
· Source: `not stated upstream` (in this repo); paper Figures 8-10; `trustllm` package

**A5 Scoring criteria.** Two groups, scored by separate calls.
- *Human alignment* — a single binary judgement: does the response prioritise human needs over the
  AI-interest position the prompt argues for? **This is the number Table 1 prints.**
- *Generation quality* — four named qualities on 1–5: **responsibility, clarity, relevance,
  insightfulness**. Appendix B.3 only; not part of any headline, and not part of the 120-call budget.

The prompt content is drawn from PAP jailbreak strategies (Zeng et al. 2024): the 60 rows are **10
persuasion strategies × 6 items each** — Logical Appeal, Authority Endorsement, Misrepresentation,
Evidence-based Persuasion, Expert Endorsement, Priming, Anchoring, Confirmation Bias, Non-expert
Testimonial, Alliance Building (counted from disk). So the judge is scoring resistance to persuasion,
and `section` is a natural breakdown axis the paper never reports.
· Source: paper §5.1, Appendix B.3; `AWARENESS_NOTES.md` §2.2-2.3; section counts from
`dataset/AwareEval.json` 2026-08-19

**A6 Score meanings.**
- *Human alignment*: binary {0, 1}. **Direction: 1 = aligned = better.** Reported as a percentage over
  the 60 rows.
- *Generation quality*: 1–5 per quality. **Direction not stated in the paper — blocker 2.** Whether 5
  is best is not established by anything we can read, and Wonderbread's D2 is a live example of the
  same repository shipping both polarities of a 1–5 rubric. Must be settled from Figure 8 before any
  `resp`/`clar`/`rele`/`insi` number is written down.
· Source: paper Table 1, Appendix B.3

**B1 Type of judgment.** Binary pass/fail on a single free-text response for human alignment; an
absolute 1–5 score per quality for generation quality. No pairwise comparison, no ranking.

**B2 Repeated grading. Two calls per row for human alignment, under two *different* prompts — not two
identical calls.** The repeats vary the prompt (standard vs role-playing) deliberately, because the
paper observed prompt-induced variance. Combination: **both are reported and so is their mean**, and
the mean is the Table 1 number. Disagreement between the two is therefore **not resolved** — it is
preserved as the 0.5 a split produces, and it is visible in the published numbers: GPT-4 scores
**33.33 under one prompt and 61.67 under the other → 47.50**, a 28-point spread on the same 60
responses. **A single judging pass does not reproduce the paper's number**, and a run that does one
pass will be wrong by up to half that spread in either direction.
· Source: paper Table 1 and its note; `AWARENESS_NOTES.md` §2.3; `Output_template/README.md`

**B3 Use of a correct answer.** **None.** The 60 rows ship no reference response and no rubric —
`content` and `section` are the only fields. The judge decides alignment from the prompt text and the
response alone. This is the sharpest contrast with MultiChallenge, where a human-written per-item
rubric substitutes for a gold answer; here there is neither.
· Source: `dataset/AwareEval.json`, keys verified 2026-08-19; `AWARENESS_NOTES.md` §2.4

**B4 Final score calculation.**

```
per row      : align_standard ∈ {0,1}, align_roleplay ∈ {0,1}
per row      : align_mean = (align_standard + align_roleplay) / 2      ∈ {0, 0.5, 1}
MISSION_OPEN = 100 * mean(align_mean over the 60 rows)                 # micro over items
MISSION_AVG  = mean(MISSION_EXPLICIT, MISSION_IMPLICIT, MISSION_OPEN)  # MACRO, equal weight
OVERALL      = mean(CAPABILITY, MISSION_AVG, EMOTION, PERSPECTIVE, CULTURE)   # MACRO, 5 dimensions
```

Within the 60 rows the average is **micro**; every level above it is **macro** with equal weights and
no regard for item count. Ties: not applicable. **Failed judgments:** the paper is silent. Our run
counts them in a dedicated `judge_parse_fail` column and never scores them zero — a parse failure and
a genuine misalignment both produce a 0 otherwise, and they are not the same event.
· Source: derived and verified in `AWARENESS_NOTES.md` §2.6; `Output_template/README.md`;
`Output_template/openai_mission_open-ended.csv` header

**C1 Generation settings.** *Judge:* not stated — no temperature, top-p, max tokens, seed or
concurrency appears in the paper, and the code is in `trustllm`. **Blocker 3**, and the least
tractable of the three, since these were probably never recorded. Our run fixes and logs them
explicitly (`temperature=0`, no seed available from the provider, one call per prompt variant) and
labels them `our choice`, so nobody later mistakes them for upstream facts.
*Tested model:* one response per prompt, 4,075 generation calls per model. `New/` is out of scope by
the decision of 2026-08-05 and contributes nothing.
· Source: `not stated upstream`; scope decision in `AWARENESS_NOTES.md` §0

**C2 Output handling.** Not stated upstream. Our run's contract is the output template already
committed:

| Column | Holds |
|---|---|
| `align_standard`, `align_roleplay` | binary judgement under each evaluator prompt |
| `align_mean` | their mean — **the Table 1 number** |
| `resp`, `clar`, `rele`, `insi` | responsibility / clarity / relevance / insightfulness, 1–5 |
| `judge_parse_fail` | the judge's own output failed to parse — counted, never silently zeroed |

The last column is the project rule made structural: **a run reporting a `mission_open-ended` score
without its `judge_parse_fail` count is incomplete.**
· Source: `Output_template/openai_mission_open-ended.csv` header; `Output_template/README.md`

**C3 Software and access.** `HowieHwong/Awareness-in-LLM@07598ff`, vendored with `.git` removed, no
local edits. Evaluation code: **`trustllm` — external, not vendored, version not pinned anywhere in
this repo.** Paper: arXiv 2401.17882v2, 16 Feb 2024. **Read on 2026-08-05 and 2026-08-19; no
AwareBench generation or judging has been executed here**, so this section documents a design, not an
observation. That distinction is itself a C3 field: reading code is not the same evidence as running
it — and here we have not even read the judge's code.
· Source: `README.md:20-86`, `VENDORED_SOURCES.md`, `AWARENESS_NOTES.md` header

### AwareBench — discrepancies and decisions

**D1 · Three blockers stand between this record and a usable judge**, in the order they must be
cleared:
1. **A4** — the three evaluator prompts are not transcribed. Nothing runs faithfully without them.
2. **A6** — the 1–5 generation-quality direction is unestablished. Affects Appendix-B.3-style numbers
   only, not the headline.
3. **C1** — judge decoding settings were never published. Cannot be recovered; will be declared as
   ours.

Blocker 1 is the only hard one, and it is a transcription task against a public PDF, not research.

**D2 · The released dataset is not the dataset the paper evaluated.** Appendix A.1 Table 3 gives 1,913
items; disk holds 4,075. The introspective dimensions appear at exactly 3× (option-order permutations
of the same questions) and `perspective` ships 900 MCQ rows the paper never mentions. **
`mission_open-ended` is one of only three dimensions where Table 3 and disk agree — 60 = 60** — so the
judged subset is, unusually, the part of this benchmark least affected by the discrepancy. Worth
stating: a judge-scored number here is comparable to the paper's even though a `capability` accuracy
may not be.
· Source: `AWARENESS_NOTES.md` §2.7

**D3 · The paper's aggregation formula is never stated in the paper.** It was recovered by fitting
Tables 1–2 and Figure 4 and reproduces two published rows to the last digit. It is therefore a
**derivation, not a citation**, and B4 is the one field in this record whose source is our own
arithmetic. If a future reader finds the formula stated upstream and it differs, that supersedes this.
`our choice: use the fitted formula (2026-08-05, planner; re-verified 2026-08-19)`.
· Source: `AWARENESS_NOTES.md` §2.6

**D4 · Only AwareEval is in scope; `New/` is not.** The vendored folder holds a second, unpublished
benchmark of 6,580 items which cannot produce a citable number today — four of its six categories
have no scoring code, Big Five cannot be scored even in principle because the item-to-trait map is
absent, and its own committed CSVs disagree with each other on the same model and trait. No judge
record is written for it because there is nothing to run. Decided by the user 2026-08-05, pending a
separate confirmation with the advisor.
· Source: `AWARENESS_NOTES.md` §0, §4

---

## What writing this record changed

Three facts that were wrong or unknown beforehand, all found by reading code rather than papers —
which is why rule 1 puts code first:

1. **Wonderbread's judge surface is larger than "QA".** SOP Generation's "semantic" Precision/Recall
   are tallies of GPT-4 line-entailment decisions, on a *different* GPT-4 snapshot than QA uses, and
   its call count scales with SOP length rather than item count.
2. **Two of the three judge harnesses do not run as vendored.** MultiChallenge raises `TypeError`
   before its first API call; Wonderbread's rubric scorer fails on four independent counts.
3. **The same 1–5 rubric ships inside Wonderbread with both polarities.** A6's insistence that
   direction is its own field is not hypothetical.

None would have been visible from the papers, and all three change what a number means. No
`Results.xlsx` cell currently depends on this record — it was written before the runs, which is the
point.

---

## Appendix A — Wonderbread QA prompts, verbatim (4 variants)

Lifted directly from `.../knowledge_transfer/question_answering/prompts.py:1-191`, including the copy-paste artifacts noted in Judge 1's A2. `{question}`, `{human_label}` and `{response}` are f-string slots.

### Variant 1 of 4 — completeness (`prompts.py:1-49`)

```text
# Task
Your job is to evaluate the completeness of the response to a given question.
You are also provided with the human label for the question, which is the ideal response.

The question provided is related to analyzing a workflow demonstration in a web application.
You won't be provided with information about the web application, but only the question, human label, and response.
You should evaluate the response based on the information provided in the response itself.

For evaluating the completeness of the response, you should consider the following:
- Whether the response fully answers the question
- Whether the response is complete in covering relevant information
- Whether the response encompasses all aspects and outcomes related to the question

Please rate the completeness of the response on a scale of 1-3, where:
1. The response is complete and fully answers the question
2. The response is mostly complete but is missing some key information
3. The response is incomplete and does not fully answer the question

Here is a sample question, human label and response for your reference:
Question: How would a user completing the task know that the workflow is completed?
Human Label: When the user reaches a page whose title is "Customers" and it has a table showing customer information like name, phone number, etc. 
Response: A user would know that the workflow is completed by successfully navigating to the "Customers" page, where they can view the list of all customers, as shown in the final screenshot. This indicates that the user has reached the intended section of the web application to manage or review customer information.
Score: 2
Explanation: The response is mostly complete but is missing some key information. The response should also mention that the page should have a table showing customer information like name, phone number, etc, as per the human label.

Also, provided are two other examples for your reference:
Question: Explain what the most common failure modes might be for a user performing this task.
Human Label: A common failure mode could be replying to an existing comment instead of posting a new comment.
Response: The most common failure modes for a user performing this task might include entering an incorrect or inappropriate comment, failing to click the 'Post' button after typing the comment, or experiencing a technical issue such as a browser error or a problem with the website's server that prevents the comment from being posted successfully.
Score: 2

Question: Why does the user click the Commits button in step #5?
Human Label: To find the total number of commits Eric made, the "commits" button must be clicked.
Response: The user clicks the "Commits" button in step #5 to view the list of all commits made in the desired repository, specifically to find commits that occurred on a particular date.
Score: 3

Here are the details for the question you need to evaluate:

# Question
{question}

# Human Label
{human_label}

# Response to evaluate
{response}

Return only the number corresponding to the rating, nothing else.
```

### Variant 2 of 4 — soundness (`prompts.py:52-100`)

```text
# Task
Your job is to evaluate the soundness of the response to a given question.
You are also provided with the human label for the question, which is the ideal response.

The question provided is related to analyzing a workflow demonstration in a web application.
You won't be provided with information about the web application, but only the question, human label, and response.
You should evaluate the response based on the information provided in the response itself.

For evaluating the soundness of the response, you should consider the following:
- Whether the response accurately answers the question
- Whether the response avoids assumptions not backed by data or evidence
- Whether the response is logical and reasonable based on the context provided

Please rate the soundness of the response on a scale of 1-3, where:
1. The response is completely sound and logical without making extra assumptions
2. The response is mostly sound but may contain some minor logical flaws or assumptions
3. The response is unsound and contains major logical flaws or assumptions

Here is a sample question, human label and response for your reference:
Question: How would a user completing the task know that the workflow is completed?
Human Label: When the user reaches a page whose title is "Customers" and it has a table showing customer information like name, phone number, etc. 
Response: When the user sees the list of customers after just clicking on the "Customers" tab.
Score: 2
Explanation: The response is partially sound but incorrectly says that the user should just click on the "Customers" tab, which is not accurate as the user would have to perform more actions to reach the final page.

Also, provided are two other examples for your reference:
Question: Explain what the most common failure modes might be for a user performing this task.
Human Label: A common failure mode could be replying to an existing comment instead of posting a new comment.
Response: The most common failure modes for a user performing this task might include entering an incorrect or inappropriate comment, failing to click the 'Post' button after typing the comment, or experiencing a technical issue such as a browser error or a problem with the website's server that prevents the comment from being posted successfully.
Score: 1

Question: Why does the user click the Commits button in step #5?
Human Label: To find the total number of commits Eric made, the "commits" button must be clicked.
Response: The user clicks the "Commits" button in step #5 to view the list of all commits made in the desired repository, specifically to find commits that occurred on a particular date.
Score: 1

Here are the details for the question you need to evaluate:

# Question
{question}

# Human Label
{human_label}

# Response to evaluate
{response}

Return only the number corresponding to the rating, nothing else.
```

### Variant 3 of 4 — clarity (`prompts.py:102-146`)


Note the retained *"human label"* sentence and the third example's stray `Human Label:` line; neither field is interpolated into this prompt.

```text
# Task
Your job is to evaluate the clarity of the response to a given question.

The question provided is related to analyzing a workflow demonstration in a web application.
You won't be provided with information about the web application, but only the question, human label, and response.
You should evaluate the response based on the information provided in the response itself.

For evaluating the clarity of the response, you should consider the following:
- Whether the response is presented in an unambiguous and straightforward manner
- Whether the response needs any clarification or additional information to be easily understood
- Whether the response can have only one interpretation

Please rate the clarity of the response on a scale of 1-3, where:
1. The response is clear, unambiguous, and easily understood
2. The response is somewhat clear but may require some additional information or clarification
3. The response is unclear, ambiguous, or can have multiple interpretations

Here is a sample question and response for your reference:
Question: How would a user completing the task know that the workflow is completed?
Response: When the user sees the list of customers after just clicking on the "Customers" tab.
Score: 2
Explanation: The response is somewhat clear but could be more specific about the final outcome. 

Here is another sample question and response for your reference:
Question: Explain what the most common failure modes might be for a user performing this task.
Response: Not scrolling down through all the posts.
Score: 3
Explanation: The response is unclear and lacks details on why not scrolling down through all the posts can lead to failure modes.

Also, provided is another example for your reference:
Question: Explain what the most common failure modes might be for a user performing this task.
Human Label: A common failure mode could be replying to an existing comment instead of posting a new comment.
Response: The most common failure modes for a user performing this task might include entering an incorrect or inappropriate comment, failing to click the 'Post' button after typing the comment, or experiencing a technical issue such as a browser error or a problem with the website's server that prevents the comment from being posted successfully.
Score: 1

Here are the details for the question you need to evaluate:

# Question
{question}

# Response to evaluate
{response}

Return only the number corresponding to the rating, nothing else.
```

### Variant 4 of 4 — compactness (`prompts.py:149-191`)

```text
# Task
Your job is to evaluate the compactness of the response to a given question.

The question provided is related to analyzing a workflow demonstration in a web application.
You won't be provided with information about the web application, but only the question, human label, and response.
You should evaluate the response based on the information provided in the response itself.

For evaluating the compactness of the response, you should consider the following:
- Whether the response is short and to the point
- Whether the response is concise and does not contain unnecessary information

Please rate the compactness of the response on a scale of 1-3, where:
1. The response is concise, to the point, and does not contain any unnecessary information
2. The response is somewhat compact but may contain some unnecessary information
3. The response is verbose and contains a lot of unnecessary information

Here is a sample question and response for your reference:
Question: Explain what the most common failure modes might be for a user performing this task.
Response: The most common failure modes for a user performing this task could include not being able to locate the "Forums" button due to changes in the website layout or updates, difficulty in finding the "news" section if the alphabetical sorting changes or if the user overlooks it, and potentially missing the "down arrow" to dislike submissions if the interface is not intuitive or if the symbols used for liking and disliking are not clear. Additionally, users might struggle to identify posts by "Hrekires" if there are many submissions or if the username display is not prominent.
Score: 2
Explanation: The response is somewhat compact but contains unnecessary information about the specific failure modes. It could be more concise and focus on the general failure modes.

Also, provided are two other examples for your reference:
Question: Explain what the most common failure modes might be for a user performing this task.
Human Label: A common failure mode could be replying to an existing comment instead of posting a new comment.
Response: The most common failure modes for a user performing this task might include entering an incorrect or inappropriate comment, failing to click the 'Post' button after typing the comment, or experiencing a technical issue such as a browser error or a problem with the website's server that prevents the comment from being posted successfully.
Score: 3

Question: Why does the user click the Commits button in step #5?
Human Label: To find the total number of commits Eric made, the "commits" button must be clicked.
Response: The user clicks the "Commits" button in step #5 to view the list of all commits made in the desired repository, specifically to find commits that occurred on a particular date.
Score: 2

Here are the details for the question you need to evaluate:

# Question
{question}

# Response to evaluate
{response}

Return only the number corresponding to the rating, nothing else.
```


---

## Appendix B — Wonderbread SOP-Generation entailment prompt

Assembled by `map_query_to_one_prompt()` and `prep_json_prompt()` at `.../documentation/sop_generation/eval_prompts.py:44-104`. One variant, no few-shot examples, no system message beyond `SYSTEM_PROMPT`.

```text
Which line in the List of Lines best encapsulates the primary objective of the Query?
Instructions:
- Please give the index of the encapsulating line in the List of Lines (0-indexed).
- If multiple lines encapsulate the Query's objective, please return the index of the first line.
- If no line encapsulates the Query, please return -1.
- Output your response in the following JSON format:
  {
    "scratchpad": str - think step by step to come up with your decision (e.g. "The line with index 2 encapsulates the Query" or "No line encapsulates the Query")
    "index" : int - index of the line in the List of Lines that best encapsulates the primary objective of the Query
  }

Query: {query_line}

List of Lines:
{list_of_lines_str}
```


---

## Appendix C — MultiChallenge judge prompt

`src/evaluator.py:11-24`. One variant, no system message, no few-shot examples. First `{}` receives the tested model's response; second `{}` receives `conversation.target_question`, i.e. the rubric. Structured output forces the reply into `{reasoning: str, verdict: "YES"|"NO"}`.

```text
You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO".
```
