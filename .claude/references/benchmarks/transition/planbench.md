# PlanBench — benchmark card

Strategy formulation. Upstream `karthikv792/LLMs-Planning`. **No LLM judge** — plans are checked by
the VAL/PDDL validator, so a result here is verifiable rather than judged. No runner of ours, no
results.

## Paths

| | Path |
|---|---|
| Local | `Transition_processes_benchmarks/LLMs-Planning_bench` |
| Quest | not verified — nothing points to a remote copy, but Quest was not checked (out of scope 2026-08-22). Confirm before assuming a remote path |

## Layout — two parallel trees, both upstream's

```
LLMs-Planning_bench/
├── plan-bench/              response_generation.py · response_evaluation.py · llm_plan_pipeline.py
│                            prompt_generation.py · problem_generators.py · obfuscator.py · results/
├── llm_planning_analysis/   the same script names again, plus stats_generation.py,
│                            results/ and results_backprompting/
├── planner_tools/           the validator side
└── README.md
```

**The duplication is real and is upstream's**, not a copy left behind by us. Decide which tree is
being used and say so in writing; the two carry the same filenames and different code, so a path
alone does not identify which one ran.

**Neither `results/` directory has anything to do with this project.** They are the upstream authors'
runs, shipped with the vendored copy. The test is the folder name: a result of ours is named after
the model that produced it, and these are not — so nothing here may be read as, compared against, or
tabulated as one of our numbers.

## Before any run

The validator is an external dependency, not a Python scorer: budget for getting VAL working before
budgeting for model calls. That ordering is the opposite of every judged benchmark here, where the
model calls are the expensive part.
