# MultiChallenge — benchmark card

Coordination, over multi-turn conversations. Upstream `ekwinox117/multi-challenge`. **LLM-judged, on
every item.** No runner of ours, no results.

## Paths

| | Path |
|---|---|
| Local | `Action_processes_benchmarks/Multi-challenge_bench` |
| Quest | not verified — nothing points to a remote copy, but Quest was not checked (out of scope 2026-08-22). Confirm before assuming a remote path |

## Layout

```
Multi-challenge_bench/
├── main.py                 entry point
├── src/evaluator.py        the judge call
├── src/conversation.py · src/data_loader.py · src/result_parser.py
├── data/
└── requirements.txt
```

## It does not run as vendored

`LLM_as_judge/JUDGE_RECORD.md` §2 records the failure and the fix: the harness raises a `TypeError`
**before its first API call**, so the failure costs nothing to reproduce and is not a quota problem.
The one-line fix is in that record's D1 — apply it from there rather than re-deriving it.

## Before any run

Every item goes through the judge, so the call count equals the item count and the judge's decoding
settings are part of the result. `JUDGE_DOCUMENTATION_RULE.md` C1 is the list of settings that have
to be recorded; §2 of the record is what has already been established.
