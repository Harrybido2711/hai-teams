# Interpersonal processes — NegotiationToM · EmoBench

Conflict management and affect management. Both are **this project's own runners around a vendored
dataset**, both run on Quest, and neither uses an LLM judge — they are scored against gold labels, so
a number here can be re-derived from the data.

What the two share, and what a third benchmark in this folder would be expected to follow:

- **One folder per model**, holding `<provider>_<bench>_eval.py`, an sbatch `.sh`, and `results/`.
  The cross-model summary CSVs depend on this shape; copy the closest folder and swap the client.
- **The same sbatch shape**: `--account=p32983`, `--partition=long`, 8 GB, and the runner invoked with
  `--task all --save-every 20`.
- **Output split by task** under `results/<task>/`, with a `_overall.csv` beside the per-item rows.
- **Both directories are ours on Quest** (`NegotiationToM/`, `EmoBench-master/`) — note the remote
  EmoBench name differs from the local one.

| Benchmark | Page | Tasks | Items | Scoring |
|---|---|---|---|---|
| NegotiationToM | [negotiationtom.md](negotiationtom.md) | desire · belief · intention | 14,138 rows per full run | EM + micro/macro F1 |
| EmoBench | [emobench.md](emobench.md) | EA · EU | 400 each, 800 total | MCQ accuracy |

NegotiationToM is the one every generic reference was originally written around, so it is also the
one whose facts leaked furthest. If a generic file states a row count, suspect it.
