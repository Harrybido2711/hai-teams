# Wonderbread — benchmark card

Monitoring progress toward goals. Upstream `HazyResearch/wonderbread`. **LLM-judged.** No runner of
ours, no results.

## Paths

| | Path |
|---|---|
| Local | `Action_processes_benchmarks/Wonderbread_bench` |
| Quest | not verified — nothing points to a remote copy, but Quest was not checked (out of scope 2026-08-22). Confirm before assuming a remote path |

## Layout

```
Wonderbread_bench/
├── wonderbread/            the upstream package (helpers.py carries the judge plumbing)
├── data/                   including data/experimental_results — upstream's, not ours
├── setup.py · requirements.txt · Dockerfile
└── README.md
```

## What the judge actually covers — larger than "QA"

Recorded in `LLM_as_judge/JUDGE_RECORD.md` §1, and found by reading code rather than the paper:

- **SOP Generation is judged too.** Its "semantic" Precision/Recall are tallies of GPT-4
  line-entailment decisions, on a **different GPT-4 snapshot** than QA uses. The call budget
  therefore scales with SOP length, not with item count.
- **The SOP-Improvement rubric scorer does not execute** as vendored — it fails on four independent
  counts.
- **The same 1–5 rubric ships with both polarities**: "1 (best) to 5 (worst)" in one file, "1 (worse)
  to 5 (best)" in another. Which direction a number uses cannot be inferred from the number.

## Open scope question

The dataset requires roughly **33 GB**. Whether that is in scope was put to the advisor and has not
come back. Do not begin transferring data on the assumption that it is settled.

## Before any run

Fill nothing in from memory: §1 of the judge record is authoritative on the judge, and
`JUDGE_DOCUMENTATION_RULE.md` is authoritative on what must be recorded before its numbers are used.
