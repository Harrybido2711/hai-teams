# LLM-as-a-judge in our benchmark suite — summary

Two pages for discussion. The evidence is in [JUDGE_RECORD.md](JUDGE_RECORD.md); section numbers
below point into it. Everything here was established by reading the benchmarks' own source code,
2026-08-19, not from their papers — which matters, because on all three the code and the paper
disagree somewhere.

## 1. Only 3 of our 10 benchmarks need a judge

| | Benchmark | Judged portion | Scored instead by |
|---|---|---|---|
| **judge** | Wonderbread | Question Answering + SOP Generation | — |
| **judge** | MultiChallenge | all 273 items | — |
| **judge** | AwareBench | 60 of 4,075 rows | accuracy on the other 4,015 |
| no | Multi-party Goal Tracking | — | gold `[MASK]` strings; residue to a **human** |
| no | PlanBench | — | VAL/PDDL symbolic validator |
| no | NegotiationToM | — | exact match + micro/macro F1 |
| no | EmoBench | — | MCQ accuracy, majority vote over permutations |
| no | DocVQA | — | ANLS |
| no | BIG-Bench Hard | — | exact match |
| no | MMLU | — | accuracy |

The seven need no judge documentation at all: they have a gold label, so any number they produce can
be recomputed by anyone holding the data. A judge score cannot be recomputed from anything except the
exact model, prompt and aggregation — which is the whole reason the record exists.

## 2. Three findings that change what we planned

**(a) Wonderbread needs a judge for more than we thought.** We had it recorded as "Question Answering
only, verify two more subtasks". The code shows **SOP Generation is judged too**: its "semantic
Precision/Recall" are not string metrics but tallies of GPT-4 line-by-line entailment decisions —
and on a *different* GPT-4 snapshot than QA uses. Budget consequence: QA is a flat 480 calls per
model, but SOP Generation is one call per SOP line in **both** directions, so **its cost scales with
how verbose the model under test is**, not with the number of items. Any budget written as "one call
per item" is wrong. *(record §1, D5)*

**(b) Two of the three judge harnesses do not run as published.** MultiChallenge passes an argument
its own constructor does not accept and raises `TypeError` before the first API call — verified by
comparing the signature to the call site. Wonderbread's SOP-Improvement rubric scorer fails on four
independent counts (two broken imports, a shadowed function, a wrong argument count). Both are
repairable; the point is that **published numbers from these repos were not produced by the code as
released**, so we should not describe our runs as "using the official evaluator" without saying what
we patched. *(record §2 D1, §1 D1)*

**(c) The same 1–5 rubric ships inside Wonderbread with opposite polarities** — "1 (best) to 5
(worst)" in one file, "1 (worse) to 5 (best)" in another. A score read under the wrong one inverts
every conclusion. This is why our record treats scale *direction* as a field of its own. *(§1 D2)*

## 3. What each judge actually sees — the methodological point

Worth raising explicitly, because it narrows what these benchmarks measure:

- **MultiChallenge** is a *multi-turn* benchmark, but its judge is shown **only the final answer and
  the rubric — never the conversation** (up to 19 messages). Any rubric needing the history to be
  checkable is effectively being guessed at. We follow upstream anyway, because deviating would make
  our numbers incomparable to everyone else's — but the limitation belongs in any writeup. *(§2 A2, D5)*
- **Wonderbread QA** withholds the screenshots and action trace from all four criteria, and withholds
  the gold answer from two of them (clarity, compactness). Those two are judgements about prose with
  no ground truth at all. *(§1 A2)*
- **AwareBench** gives its judge no reference answer and no rubric — only the prompt and the
  response. *(§3 B3)*

**Only one of the three judges has ever been validated against humans:** Wonderbread QA, n = 30,
Spearman 0.80–0.89, 87–97% exact agreement (recomputed by us from their committed data). Who the
human graders were is not stated, so we cannot tell whether that is near the human–human ceiling.
MultiChallenge and AwareBench report **no** human comparison. *(§1 A3, §2 A3, §3 A3)*

## 4. AwareBench: judging 60 rows out of 4,075, and why that is defensible

The judged 60 (`mission_open-ended`) are the only rows in the file carrying no answer key — the split
is a property of the data, not a choice. We deliberately do **not** route the other 4,015 through a
judge, for four reasons: they already have a right answer; judging them would break comparability
with the paper's 13-model table, our only external check; it would cost 67× more (120 → 8,150 calls
per model); and the one problem it might paper over is a parser bug, which should be fixed rather
than outsourced to a model.

**The asymmetry worth knowing:** the paper weights dimensions equally, not by item count, so those 60
rows carry **1/15 of the headline score** — the same weight as a 966-row task. The judged fraction is
1.5% of the items and 6.7% of the number, so judge noise there is disproportionately visible. *(§3)*

## 5. Cost per model evaluated

| Benchmark | Generation calls | Judge calls |
|---|---:|---|
| AwareBench | 4,075 | **120** (60 rows × 2 evaluator prompts) |
| MultiChallenge | 273 × attempts | **273** × attempts |
| Wonderbread QA | 120 | **480** (120 × 4 criteria) |
| Wonderbread SOP Generation | per demonstration | **(pred lines + gold lines) per demo** — scales with verbosity |

## 6. Open questions for you

1. **Judge-model substitution is now forced, not optional.** All three GPT-4 snapshots Wonderbread
   used (`gpt-4-0125-preview`, `gpt-4-1106-preview`, `gpt-4-turbo`) are retired, as is the GPT-4 used
   by AwareBench. Exact reproduction is impossible at any price. We would substitute a current model,
   log it, and report our numbers as *not* directly comparable to the published ones. **Is that
   acceptable, or would you rather we report only the objectively scored benchmarks?**
2. **AwareBench scope.** We plan to run the published `AwareEval` (4,075 items) and **not** the
   unpublished `New/` folder in the same repo — `New/` has no scoring code for four of its six
   categories and cannot produce a citable number today. Confirming this was flagged as needing your
   input.
3. **Wonderbread SOP Improvement.** Its scorer does not execute and the rubric direction is
   ambiguous. We propose excluding that subtask rather than writing a scorer from the paper. Agree?
4. **AwareBench's judge prompts are not published.** They exist only as figures in the paper and in
   an external package. We will transcribe them verbatim rather than paraphrase — a paraphrase is a
   different prompt and produces different scores. This is a transcription task, not a blocker on
   your side, but it is why no AwareBench judge run has started.

## 7. Status

No judge-scored number exists in `Results.xlsx` yet. The record was written **before** the runs, so
that decisions like the substitution above are made on the record rather than discovered afterwards.
