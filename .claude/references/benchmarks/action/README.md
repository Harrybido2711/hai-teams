# Action processes — Wonderbread · MultiChallenge

Monitoring progress toward goals, and coordination. Both were vendored on 2026-08-19 and they share
three properties that decide how any work here starts:

- **Both are judged by an LLM**, so neither number can be re-derived from the data. Their records are
  §1 and §2 of `LLM_as_judge/JUDGE_RECORD.md`, and nothing from either may enter `Results.xlsx`
  until that record is filled — that is the rule in `JUDGE_DOCUMENTATION_RULE.md`.
- **Neither has a runner.** No per-provider harness, no SLURM script, no results. Writing one is the
  first task in both cases, not a step after a pilot.
- **Both harnesses fail as vendored.** This is the finding that writing the judge records produced,
  and it is why "just run the upstream code" is not available here.

| Benchmark | Page | Judge covers | State as vendored |
|---|---|---|---|
| Wonderbread | [wonderbread.md](wonderbread.md) | QA **and** SOP Generation | rubric scorer does not execute — four independent counts |
| MultiChallenge | [multichallenge.md](multichallenge.md) | every item | raises `TypeError` before its first API call |
