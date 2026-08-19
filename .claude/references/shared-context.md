# Shared context

Read what bears on your task before acting. These are committed, so they arrive with a clone and
stay in sync as the project moves — prefer them over anything remembered from a previous session.
**If one of them contradicts what you were told, say so rather than silently picking one.**

## Which document is authoritative on what

| Document | Authoritative on |
|---|---|
| `Interpersonal_processes_benchmarks/NegotiationToM/negotiation.md` | The key findings: current results, the dataset traps that silently change scores, reasoning-token cost, the silent-failure catalogue, and the conventions that must not drift. Read this first for anything NegotiationToM |
| `Interpersonal_processes_benchmarks/NegotiationToM/ISSUES.md` | Problems already hit, what was rejected, what shipped, and the false alarms recorded so they are not investigated twice |
| `Interpersonal_processes_benchmarks/NegotiationToM/DATA_NOTES.md` | Dataset traps: cutoff tiling, the `"None"` sentinel, which gold fields are correct, expected row counts |
| `benchmark_evaluation_guide.md` | What each benchmark requires — metrics, judges, assets |
| `Interpersonal_processes_benchmarks/EmoBench/EMO_SCRIPT.md`, `Interpersonal_processes_benchmarks/NegotiationToM/Negotiation_script.md` | Per-benchmark task semantics. Authoritative on what a task *means*; their file listings go stale, so verify listings against the tree |

## Repo layout

Benchmarks are grouped by the team process they measure — the grouping is the tracker's, so a
folder states what a benchmark is evidence for (reorganised 2026-08-19; paths before that date in
older notes are stale):

```
Transition_processes_benchmarks/   Awareness_in_LLM, LLMs-Planning_bench, Multi-party_Goal_Tracking_bench
Action_processes_benchmarks/       Wonderbread_bench, Multi-challenge_bench
Interpersonal_processes_benchmarks/ NegotiationToM, EmoBench
Tasks_benchmarks/                  DocVQA, bbh, mmlu
Random_stuff/                      SQA Release 1.0, TruthfulQA-main, sycophancy-eval-main
```

`PLAN.md` at the repo root is the map: what each folder is, which benchmark needs a judge, and what
state each run is in. `VENDORED_SOURCES.md` records which upstream commit each vendored copy came
from — those folders carry no `.git` of their own.

**On Quest nothing moved.** The remote layout is still flat under `/gpfs/projects/p32983/`, so
`/gpfs/projects/p32983/NegotiationToM` remains correct and must not be "fixed" to match local.

Every benchmark uses **one folder per model** — a Python eval script, a SLURM `.sh`, a `results/`
directory. Copy the closest existing folder and swap the client; do not invent a new structure,
because the cross-model summary CSVs depend on this shape.

```
<BENCH>_<Provider>/            e.g. EMO_Qwen/, NEG_GPT/
├── <provider>_<bench>_eval.py
├── run_<bench>.sh
└── results/<TASK>/
```

The active work is NegotiationToM, which additionally factors the shared logic into
`neg_eval_core.py`, with six thin runners (`NEG_{GPT,Gemini,XAI,Qwen,Gemma,Deepseek}/<provider>_neg_eval.py`)
each supplying only their own `call_api`.

## Numbers worth knowing before counting anything (NegotiationToM)

A full run is **14,138 rows**: desire 4,760 + belief 4,760 + intention **4,618**.

- **intention at 4,760 means a known bug has returned** — odd-length dialogues annotate one target
  utterance, not two.
- `scored_rows` is **4,604** for desire and belief: 156 rows per task are excluded because their
  gold is the sentinel `"None"`, which marks an unannotated sample rather than "wants nothing".
- `All_EM` in the low single-digit percents is **expected**, not a defect: it ANDs all 5–6 rows of a
  dialogue, and intention gets no partial credit there even though its F1 does.

## Two ways a results directory lies

- **A finished job proves nothing about whether its data is usable.** Grok's five shards all exited
  `COMPLETED 0:0` with a perfect 14,138 rows while belief and intention were 100% empty — its
  credits had run out. Report the **non-empty `raw_response` rate and the null-`pred` count**, not
  just row counts.
- **Verify from the `.jsonl`, not the `.csv`.** Reasoning models emit newlines inside
  `raw_response`, so `cut -d, -f1` on the CSV mis-parses and can report *more* unique uids than
  rows. That false alarm looked model-specific and nearly triggered a needless re-run.
