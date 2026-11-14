# Transition processes — AwareBench · PlanBench · mpgt-eval

Mission analysis, strategy formulation, goal specification. Unlike the other three folders, these
three share almost nothing operationally — different scoring, different state, different blockers.
AwareBench (recorded 2026-10-12) and mpgt-eval (recorded 2026-11-13) have produced results;
PlanBench's full run is on Quest.

| Benchmark | Page | Scoring | What stands between here and a number |
|---|---|---|---|
| AwareBench | [awarebench.md](awarebench.md) | accuracy; its 60 LLM-judged rows were dropped 2026-10-05 | full run for the five selected models 2026-10-12, recorded in both workbooks; repeat runs for criteria 4–5 pending |
| PlanBench | [planbench.md](planbench.md) | VAL validator + upstream's text parsers, run offline — no LLM | runners for the five selected models, all 10 tasks, written 2026-11-09; 10% pilot done 2026-11-13; full run on Quest since 2026-11-13, then audit and the user's confirmation |
| mpgt-eval | [mpgt.md](mpgt.md) | lenient matcher on the `[MASK]` slots — ours; upstream's was a person (2026-11-09) | full run of the five selected models, zero-shot, recorded in both workbooks 2026-11-13; open: whether an off-count answer list is aligned by position |

None of the three has a model in the loop when scoring — accuracy, a PDDL validator, and for mpgt
an automatic matcher where upstream had a person. A number here can always be recomputed from the
stored responses.
