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

The **Material shown** row of the table in §4 is the one worth arguing about, because it narrows what
each benchmark actually measures relative to how it is described:

- **MultiChallenge is a *multi-turn* benchmark whose judge never sees the conversation.** It gets the
  final answer and the rubric, nothing else — so any rubric that needs the history to be checkable is
  effectively being guessed at. We follow upstream anyway, because deviating would make our numbers
  incomparable to everyone else's, but the limitation belongs in any writeup. *(record §2 A2, D5)*
- **Two of Wonderbread QA's four criteria have no ground truth at all.** Clarity and compactness are
  judgements about prose, made without the gold answer and without the demonstration. *(§1 A2)*
- **AwareBench's judge gets neither a reference answer nor a rubric** — only the prompt and the
  response. *(§3 B3)*

And the **Human comparison** row is nearly empty: only Wonderbread QA was ever checked against people
(n = 30, Spearman 0.80–0.89, recomputed by us from their committed data), and even there the graders
are unidentified, so we cannot tell whether 0.86 is near the human–human ceiling or well below it.
MultiChallenge and AwareBench report none.

## 4. Field-by-field comparison

The thirteen required fields, side by side. Wonderbread is split into its two working judges: they
disagree with each other on almost every row, so collapsing them into one column would hide the
point. AwareBench is not in the table — its grading model, prompts and decoding settings were never
published (§7 below), so most of its cells would read "not stated upstream", which is itself the
finding rather than a comparison.

### Grading setup

| Field | Wonderbread · QA | Wonderbread · SOP Generation | MultiChallenge |
|---|---|---|---|
| **Grading model** | `gpt-4-0125-preview` — the code passes a label `'GPT4'` that resolves elsewhere | `gpt-4-1106-preview` — **a different snapshot**, hard-coded as a module constant | `gpt-4o-2024-08-06`, hard-coded; structured-output endpoint |
| **Material shown** | question + tested answer always; **gold answer only for 2 of 4 criteria**; never the screenshots, action trace or SOP | one SOP line as the query + the *whole* other SOP as an indexed list; nothing else | tested answer + the item's rubric — **never the conversation** (up to 19 turns), never a gold answer |
| **Human comparison** | **yes** — n = 30, Spearman 0.80–0.89, exact agreement 87–97%. **Who the graders were is not stated** | not reported | not reported |
| **Grading instructions** | 4 variants, each with 3 few-shot examples carrying scores. Verbatim in record App. A | 1 variant, **no few-shot examples**. Verbatim in App. B | 1 variant, no few-shot, **no system message**. Verbatim in App. C |
| **Scoring criteria** | soundness, completeness, clarity, compactness — **4 separate calls**, no stated priority | one: semantic entailment of a step's primary objective | **none named** — the item's own rubric, plus the instruction `Be VERY STRICT!` |
| **Score meanings** | 1–3, **1 = best, 3 = worst**; every point defined | an index, or −1 for "no match" — not a scale | binary YES/NO; passes when `verdict == PASS_CRITERIA` (all 273 rows ship `"YES"`) |

### Comparison and scoring process

| Field | Wonderbread · QA | Wonderbread · SOP Generation | MultiChallenge |
|---|---|---|---|
| **Type of judgment** | absolute score of one answer | mapping / entailment decision | binary pass/fail |
| **Repeated grading** | 1 call per (item, criterion) = 4 per item. No repeats, so judge variance is unmeasured | 1 call per line **in both directions**; no repeats | 1 call per (item, attempt). Repeats happen on the *generation* side; combined **any-pass**, i.e. pass@k not mean-of-k |
| **Use of a correct answer** | the `Human Label` column — shown to completeness and soundness, **withheld from clarity and compactness**. Provenance undocumented | the human-written Gold SOP | **none exists.** The per-item rubric replaces a gold answer |
| **Final score calculation** | mean per criterion over 120 items (**micro**); four numbers, **no composite**. Short/empty answers become `"NA"` and are silently uncounted | fraction of lines with index ≠ −1 → precision, recall, ordering | per-axis accuracy, then **macro mean over the 4 axes** — and the axes are unequal (113/69/50/41), so 41 items weigh as much as 113. Failed judgments count as failures |

### Replication details

| Field | Wonderbread · QA | Wonderbread · SOP Generation | MultiChallenge |
|---|---|---|---|
| **Generation settings** | temperature **0.0**, max_tokens 4096, no seed, sequential. Retry on 429 is **unbounded recursion** | **no temperature passed → API default 1.0, so this judge is not deterministic.** Completions cached to `sop_cache/` and reused unless forced | temperature 0, structured output, no seed, **no retry, no timeout**. `max_tokens` is passed but never reaches the API |
| **Output handling** | "Return only the number" — but **no extraction, no cast, no validation**; the raw string is stored as the score | JSON parse, **one retry**, then raise. `int()` cast on the index. Not silent | schema-enforced, so the verdict cannot be off-vocabulary. But **any exception becomes a `NO`**, i.e. a silent failed item |
| **Software and access** | `wonderbread@ed052c6`, `openai` client. **Read 2026-08-19, not executed** | same repo and date | `multi-challenge@5ccefcc`; `openai==1.53.0`, `pydantic==2.10.6`. **Read 2026-08-19, not executed** |

### What the table makes visible

- **Nothing is shared between the three.** Different models, different material, different scales,
  different aggregation, different failure handling. "We used GPT-4 as a judge" describes none of them
  accurately.
- **Two of the three scales run in a direction a reader will guess wrong**: Wonderbread QA is
  1 = best, and its `-1` entailment sentinel is a non-answer rather than a low score.
- **Only one row in the whole table reports a human check**, and even there the graders are anonymous.
- **The failure rows differ in a way that moves numbers in opposite directions**: Wonderbread QA drops
  unscorable items from the denominator, MultiChallenge counts them as failures. One inflates a score,
  the other deflates it.

## 5. AwareBench: judging 60 rows out of 4,075, and why that is defensible

The judged 60 (`mission_open-ended`) are the only rows in the file carrying no answer key — the split
is a property of the data, not a choice. We deliberately do **not** route the other 4,015 through a
judge, for four reasons: they already have a right answer; judging them would break comparability
with the paper's 13-model table, our only external check; it would cost 67× more (120 → 8,150 calls
per model); and the one problem it might paper over is a parser bug, which should be fixed rather
than outsourced to a model.

**The asymmetry worth knowing:** the paper weights dimensions equally, not by item count, so those 60
rows carry **1/15 of the headline score** — the same weight as a 966-row task. The judged fraction is
1.5% of the items and 6.7% of the number, so judge noise there is disproportionately visible. *(§3)*

## 6. Cost per model evaluated

| Benchmark | Generation calls | Judge calls |
|---|---:|---|
| AwareBench | 4,075 | **120** (60 rows × 2 evaluator prompts) |
| MultiChallenge | 273 × attempts | **273** × attempts |
| Wonderbread QA | 120 | **480** (120 × 4 criteria) |
| Wonderbread SOP Generation | per demonstration | **(pred lines + gold lines) per demo** — scales with verbosity |

## 7. Open questions for you

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

## 8. Status

No judge-scored number exists in `Results.xlsx` yet. The record was written **before** the runs, so
that decisions like the substitution above are made on the record rather than discovered afterwards.
