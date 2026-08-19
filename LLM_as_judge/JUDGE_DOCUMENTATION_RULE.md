# Rule: what must be recorded about a judge, before its numbers are used

A judge-scored number is not reproducible unless the judge itself is documented. Accuracy against a
gold label can be re-derived from the data; a GPT score cannot be re-derived from anything except the
exact model, the exact prompt, and the exact aggregation. This file defines the record that every
judge-scored benchmark in this repo must carry before its numbers enter `Results.xlsx`.

It is a **rule and a template**, not a filled-in survey. The thirteen required fields are quoted
verbatim in the next section and expanded after it. The filled record is
[JUDGE_RECORD.md](JUDGE_RECORD.md) — **one file covering every judge in the suite**, written
2026-08-19. Benchmarks with no LLM judge get no record; that file's first section names them and
says why.

Companions: [JUDGE_RECORD.md](JUDGE_RECORD.md) decides *which* benchmarks need a judge and holds the
filled record for the three that do; [GPT_LLM_AS_JUDGE_GUIDE.md](GPT_LLM_AS_JUDGE_GUIDE.md) is how to
*build* one for our own use. This file is how to *document* one that already exists.

## The requirement, verbatim

This is the specification as given, reproduced word for word. It is the authoritative list; the
sections after it expand each line but never replace or reword it. If the two ever disagree, this
block wins.

```text
For each benchmark, record the following:

Grading Setup
Grading model: What model graded the answers? Record the exact model name and version, when available.
Material shown to the model: Did it receive the question, tested answer, correct answer, background documents, or previous conversation?
Human comparison: Were model grades compared with human judgments? Record the sample size, potential few shot examples, agreement measures, and who the human graders were (students, researchers, etc.).
Grading instructions: Copy the complete instructions given to the model, including any examples.
Scoring criteria: What qualities was the model asked to consider, such as correctness, relevance, clarity, safety, or completeness?
Score meanings: What scale was used, and was each score clearly defined?

Comparison and Scoring Process
Type of judgment: Did the model score one answer, compare two answers, rank several answers, or give a pass/fail decision?
Repeated grading: Was each answer graded once or multiple times? Record how disagreements across runs were resolved.
Use of a correct answer: Was the model given a reference answer? If so, how was that answer created and checked?
Final score calculation: How were ratings combined across questions, criteria, models, or repeated runs? Note how ties and failed grading attempts were handled.

Replication Details
Generation settings: Record all reported settings that may affect grading, including randomness settings, number of grading runs, and any fixed random number.
Output handling: How was the model’s response converted into a score? Record required formatting, extraction rules, and treatment of malformed responses.
Software and access: Record the code, software packages, service used to access the model, and date of evaluation.
```

Thirteen fields in three groups. The numbering used below — A1–A6, B1–B4, C1–C3 — follows this list
in order, so a record can be checked against it field by field.

## Scope

| Needs a record | Needs none |
|---|---|
| AWAREBENCH (the 60 `mission_open-ended` rows only), Wonderbread (its judge-scored subtasks), MultiChallenge (all items) | Multi-party Goal Tracking, PlanBench, NegotiationToM, EmoBench, DocVQA, BIG-Bench Hard, MMLU |

**The right-hand column gets no record at all** — there is no judge, so there is nothing about one to
document. They are listed once, by name and with their scoring code, in the first section of
[JUDGE_RECORD.md](JUDGE_RECORD.md), so that "no record" and "not yet checked" are never confused.

## Three rules that make a record worth having

1. **Every field cites its source.** A file path with a line number for code, a section or appendix
   number for a paper, or the literal words `not stated upstream`. A field is never left blank and a
   value is never inferred from what a benchmark "probably" does.
2. **Prompts are copied verbatim.** Full text, every few-shot example, the system message, the
   output-format sentence, and the placeholders exactly as they appear. A paraphrase is a different
   prompt and produces different scores.
3. **Where code and paper disagree, record both** and state which one our run follows. Silently
   preferring one is how a reproduction ends up measuring something nobody described.

Anything we decide ourselves — because upstream never specified it — is recorded as
`our choice: <value> (<date>, <who decided>)`, so it is never mistaken for an upstream fact.

## Section A — Grading setup

**A1 · Grading model.** Exact model name and version string, the provider and endpoint, and the
snapshot date if the name is a moving alias. When the code passes a label through a wrapper (a
`"GPT4"` argument that resolves to a dated model id somewhere else), record the string the API
actually receives, and the resolving file.

**A2 · Material shown to the judge.** Enumerate which of these the judge receives: the question or
instruction, the tested model's answer, a reference/gold answer, source documents (images, videos,
traces, retrieved passages), the prior conversation, and the per-item rubric. **State explicitly what
is withheld** — a judge that never sees the conversation history, or never sees the gold answer, is
scoring something narrower than the benchmark's description suggests, and that is exactly the fact a
reader needs.

**A3 · Human comparison.** Was the judge validated against human judgments? Record the sample size,
who the humans were (paper authors, students, crowdworkers, domain experts), the agreement measure
used (Cohen's κ, percent agreement, correlation) and its value, and whether the few-shot examples
inside the prompt were drawn from those same human labels. `not reported` is a valid and important
answer — write it rather than leaving the field out.

**A4 · Grading instructions.** The complete prompt in a fenced block. One block per variant, and the
number of variants stated, because some benchmarks deliberately judge the same item under more than
one prompt. Include the system message, all in-prompt examples with their example scores and
explanations, and the final formatting instruction.

**A5 · Scoring criteria.** The named qualities the judge is told to weigh — correctness, relevance,
completeness, clarity, compactness, soundness, safety, faithfulness — and their stated priority if
the prompt gives one. If several criteria are scored by separate calls rather than one call, say so
here and count the calls in B2.

**A6 · Score meanings.** The scale, **its direction**, and the definition attached to each point.
Direction is a separate field on purpose: a 1–3 scale where 1 is best and a 1–3 scale where 3 is best
look identical in a results table and invert every conclusion. Record whether each point carries a
written definition or only the endpoints do.

## Section B — Comparison and scoring process

**B1 · Type of judgment.** One of: absolute score for a single answer, pairwise comparison, ranking
of several answers, binary pass/fail, or a mapping/entailment decision (does this line correspond to
that one). Include what the judge outputs alongside the score — reasoning, explanation, an index.

**B2 · Repeated grading.** How many judge calls per item, and per criterion. If more than one, the
rule for combining them: mean, majority vote, any-pass, or both reported separately. State how ties
and disagreements between runs are resolved, and whether repeats vary anything (prompt variant,
option order, temperature) or are identical calls.

**B3 · Use of a correct answer.** Whether the judge is given a reference answer, which fields carry it, who wrote
it, and how it was validated. If some criteria see the reference and others do not, list them
separately — that asymmetry changes what each score means.

**B4 · Final score calculation.** The full path from per-item judgments to the reported number:
aggregation within criteria, across items, across repeats, and across subsets. Say whether the
headline number is a **micro** average over items or a **macro** average over categories; they differ
whenever categories are unequal in size. Then: how ties are handled, and what happens to a failed or
unparseable judgment — dropped from the denominator, counted as a failure, or written as `NA`. Each
of those three choices moves the score in a different direction.

## Section C — Replication details

**C1 · Generation settings.** For the judge: temperature, top-p, max tokens, seed, structured-output
or response-format setting, concurrency, retry and timeout policy, and the number of grading runs.
For the tested model, record the same settings where they change what the judge sees — including how
many answers were sampled per question.

**C2 · Output handling.** The required output format, the exact extraction rule (regex, JSON parse,
structured-output schema, "first number in the string"), and the malformed-output policy. Per this
project's standing rule, parse failures are **counted and reported as their own category, never
silently scored zero** — record whether upstream does that, and what our run does.

**C3 · Software and access.** Repository URL and the commit hash actually read, relevant package
versions, the API or service used to reach the model, who ran the evaluation, and the date. If a
record was written from a repo that we did not run, say that too: reading code is not the same
evidence as executing it.

## Blank record — the shape each benchmark's section follows in `JUDGE_RECORD.md`

````markdown
# <Benchmark> — judge record

Scope: <which subtask(s)/how many items of how many are judged; the rest are scored how>
Written: <date> · Sources read: <repo@commit>, <paper §>

## A · Grading setup
A1 Grading model:          <value>   · Source: <file:line | paper § | not stated upstream>
A2 Material shown to the model: <shown: ... | withheld: ...>   · Source:
A3 Human comparison:       <n, who, measure, value | not reported>   · Source:
A4 Grading instructions:   <verbatim, below>   · Source:
A5 Scoring criteria:       <value>   · Source:
A6 Score meanings:         <scale, direction, per-point definitions>   · Source:

### A4 prompts (verbatim, variant 1 of N)
```text
<full prompt, placeholders as-is>
```

## B · Comparison and scoring
B1 Type of judgment:       <value>   · Source:
B2 Repeated grading:       <calls/item; combination rule; tie rule>   · Source:
B3 Use of a correct answer: <given? provenance? which criteria see it>   · Source:
B4 Final score calculation: <aggregation path; micro/macro; ties; failed judgments>   · Source:

## C · Replication
C1 Generation settings:    <judge: ...; tested model: ...>   · Source:
C2 Output handling:        <format, extraction, malformed policy>   · Source:
C3 Software and access:    <repo@commit, packages, API, who, date>   · Source:

## Discrepancies and decisions
- <code vs paper conflicts; upstream bugs; our choice: ... (date, who)>
````

Formatting of a filled line, so provenance stays inline rather than in a footnote:

```
A1 Grading model:  gpt-4o-2024-08-06, OpenAI chat completions  · Source: src/evaluator.py:31
A3 Human comparison:  not reported in repo; paper § to check   · Source: not stated upstream
```

## Order of work when a record is filled

1. **Read the code first, the paper second.** The code is what produced the published numbers; the
   paper is what the authors meant to do. Both go in, code first.
2. **Fill every field, including the ones that come back empty.** An unfillable field is itself a
   replication finding and belongs in the record, not in someone's memory.
3. **Record upstream bugs where they are found.** A judge harness that crashes, mislabels an attempt
   index, or passes the wrong variable into a prompt slot changes what the number means, and the next
   person will otherwise rediscover it at their own cost.
4. **Add the benchmark to the verdict table** at the top of [JUDGE_RECORD.md](JUDGE_RECORD.md), in
   the judged list or the needs-no-record list, so the verdict and the record never drift apart.

## Status

| Benchmark | Section of `JUDGE_RECORD.md` | Status |
|---|---|---|
| Wonderbread | §1 | **written 2026-08-19** — scope corrected: QA *and* SOP Generation are judged; the SOP-Improvement rubric scorer does not execute |
| MultiChallenge | §2 | **written 2026-08-19** — harness raises `TypeError` as vendored; fix recorded in its D1 |
| AWAREBENCH | §3 (60 of 4,075 rows) | **written 2026-08-19** — 3 blockers open: judge prompts not transcribed (A4), 1-5 direction unset (A6), judge decoding settings unpublished (C1) |
| the other seven | named in the opening section | no record needed — no LLM judge |

**What writing it changed.** Three facts that were wrong or unknown before, all found by reading code
rather than papers — which is why rule 1 puts code first:

1. **Wonderbread's judge surface is larger than "QA".** SOP Generation's "semantic" Precision/Recall
   are tallies of GPT-4 line-entailment decisions, on a *different* GPT-4 snapshot than QA uses.
2. **Two of the three judge harnesses do not run as vendored** — MultiChallenge raises `TypeError`
   before its first API call, and Wonderbread's rubric scorer fails on four independent counts.
3. **The same 1-5 rubric ships in Wonderbread with both polarities** — "1 (best) to 5 (worst)" in one
   file, "1 (worse) to 5 (best)" in another. A6's insistence that direction is its own field is not
   hypothetical.

None of the three would have been visible from the papers, and all three change what a number means.
