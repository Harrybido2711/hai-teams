# Transition processes — AwareBench · PlanBench · mpgt-eval

Mission analysis, strategy formulation, goal specification. Unlike the other three folders, these
three share almost nothing operationally — different scoring, different state, different blockers.
The only safe generalisation is that **none of them has produced a result yet.**

| Benchmark | Page | Scoring | What stands between here and a number |
|---|---|---|---|
| AwareBench | [awarebench.md](awarebench.md) | mostly accuracy; 60 rows LLM-judged | scope is settled and templates exist, but no generation has run |
| PlanBench | [planbench.md](planbench.md) | VAL/PDDL validator — no LLM | no runner of ours |
| mpgt-eval | [mpgt.md](mpgt.md) | human review | no runner of ours |

Two of the three are scored **without a model in the loop** — a PDDL validator and a human. That
makes them cheap to trust and expensive to run, which is the opposite of the profile everywhere else
in this repo, and it should change how a run is planned.
