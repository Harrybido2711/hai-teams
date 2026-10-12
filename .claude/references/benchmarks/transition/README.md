# Transition processes — AwareBench · PlanBench · mpgt-eval

Mission analysis, strategy formulation, goal specification. Unlike the other three folders, these
three share almost nothing operationally — different scoring, different state, different blockers.
Only AwareBench has produced results so far, and those are not yet confirmed.

| Benchmark | Page | Scoring | What stands between here and a number |
|---|---|---|---|
| AwareBench | [awarebench.md](awarebench.md) | accuracy; its 60 LLM-judged rows were dropped 2026-10-05 | full run for the five selected models 2026-10-12, 4,015 rows each; a scorer gap to decide, then the user's confirmation |
| PlanBench | [planbench.md](planbench.md) | VAL/PDDL validator — no LLM | no runner of ours |
| mpgt-eval | [mpgt.md](mpgt.md) | human review | no runner of ours |

Two of the three are scored **without a model in the loop** — a PDDL validator and a human. That
makes them cheap to trust and expensive to run, which is the opposite of the profile everywhere else
in this repo, and it should change how a run is planned.
