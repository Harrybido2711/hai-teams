# Shared context

Read what bears on your task before acting. These are committed, so they arrive with a clone and
stay in sync as the project moves — prefer them over anything remembered from a previous session.
**If one of them contradicts what you were told, say so rather than silently picking one.**

## Which document is authoritative on what

| Document | Authoritative on |
|---|---|
| `NegotiationToM/negotiation.md` | The key findings: current results, the dataset traps that silently change scores, reasoning-token cost, the silent-failure catalogue, and the conventions that must not drift. Read this first for anything NegotiationToM |
| `NegotiationToM/ISSUES.md` | Problems already hit, what was rejected, what shipped, and the false alarms recorded so they are not investigated twice |
| `NegotiationToM/DATA_NOTES.md` | Dataset traps: cutoff tiling, the `"None"` sentinel, which gold fields are correct, expected row counts |
| `benchmark_evaluation_guide.md` | What each benchmark requires — metrics, judges, assets |
| `EmoBench-master/EMO_SCRIPT.md`, `NegotiationToM/Negotiation_script.md` | Per-benchmark task semantics. Authoritative on what a task *means*; their file listings go stale, so verify listings against the tree |

## Repo layout

Benchmarks live one per directory: `NegotiationToM/`, `EmoBench-master/`, `bbh/`, `DocVQA/`,
`mmlu/`, `TruthfulQA-main/`, `LLMs-Planning-main/`, `sycophancy-eval-main/`.

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
