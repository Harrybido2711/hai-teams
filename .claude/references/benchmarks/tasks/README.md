# General task ability — DocVQA · BIG-Bench Hard · MMLU

Not team-process benchmarks: these measure raw task ability, and exist so a team-process number can
be read against what the model can do at all. None uses an LLM judge.

What this folder's benchmarks share, and how they differ from the process benchmarks:

- **Flat layout, no per-model folders.** `<provider>_eval.py` + `<provider>_eval_script.sh` sit
  beside the data, and results land as `<provider>_<task>.csv` in the same directory. Do not
  "correct" this to the one-folder-per-model shape — the summary CSVs read the flat names.
- **One JSON per task or subject**, so the task list is `ls *.json`, not a constant in the code.
- **Provider coverage is uneven and is a scope decision, not an oversight** — see `PLAN.md`.

| Benchmark | Page | Unit | Scoring |
|---|---|---|---|
| DocVQA | [docvqa.md](docvqa.md) | 5,349 validation questions + page images | ANLS |
| BIG-Bench Hard | [bbh.md](bbh.md) · [bbh-parameters.md](bbh-parameters.md) | 20 task files | exact match — but see the page, there are two scorers |
| MMLU | [mmlu.md](mmlu.md) | 13 subject files | accuracy |

DocVQA is the outlier: it is sharded, it carries 3.5 GB of page images, and it is the only one here
with its own incident log.
