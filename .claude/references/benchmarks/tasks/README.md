# General task ability — DocVQA · BIG-Bench Hard · MMLU

Not team-process benchmarks: these measure raw task ability, and exist so a team-process number can
be read against what the model can do at all. None uses an LLM judge.

What this folder's benchmarks share, and how they differ from the process benchmarks:

- **Layout is per-model in bbh, mixed in the other two.** **bbh was converted on 2026-08-29**: one
  `<model>/` folder per provider holds its runner, its job script, its result CSVs and its own SLURM
  logs, with the task JSONs shared at the benchmark root. This page previously said not to convert
  it, because "the summary CSVs read the flat names" — that reason did not hold: nothing outside the
  folder opened a bbh CSV by path, and each runner writes only its own `<provider>_overall_results.csv`.
  **mmlu is half-converted** (`gemma/ xai/ openai/ llama/` exist; the other providers are still flat
  beside the data) and **DocVQA has per-model folders** (`gemma_DocVQA/`, `qwen_DocVQA/`). Check the
  benchmark's own page before assuming either shape.
- **A runner that moves into a subfolder must stop resolving paths against the cwd**, or it finds no
  data and writes an empty result while the job still exits `COMPLETED 0:0`. `bbh/kimi/kimi_outlog`
  is that failure, recorded: 20 splits, every one `No such file or directory`. bbh's runners now
  anchor on `__file__`; mmlu's subfolder runners have not been checked.
- **One JSON per task or subject**, so the task list is `ls *.json`, not a constant in the code.
- **Provider coverage is uneven and is a scope decision, not an oversight** — see `PLAN.md`.

| Benchmark | Page | Unit | Scoring |
|---|---|---|---|
| DocVQA | [docvqa.md](docvqa.md) | 5,349 validation questions + page images | ANLS |
| BIG-Bench Hard | [bbh.md](bbh.md) · [bbh-parameters.md](bbh-parameters.md) | 20 task files | exact match — but see the page, there are two scorers |
| MMLU | [mmlu.md](mmlu.md) | 13 subject files | accuracy |

DocVQA is the outlier: it is sharded, it carries 3.5 GB of page images, and it is the only one here
with its own incident log.
