# General task ability — DocVQA · BIG-Bench Hard · MMLU

Not team-process benchmarks: these measure raw task ability, and exist so a team-process number can
be read against what the model can do at all. None uses an LLM judge.

What this folder's benchmarks share, and how they differ from the process benchmarks:

- **bbh now has EmoBench's shape; the other two do not.** **bbh was rebuilt on 2026-08-29** onto
  `BBH_<Slot>/` folders — `<vendor>_bbh_eval.py`, `run_bbh.sh`, `log.txt`/`log.err`, and
  `results/<task>/<model-slug>.{jsonl,csv}` + `_overall.csv` — with the task JSONs in `data/` and a
  shared `bbh_eval_core.py` holding the scorer. This page previously said not to convert it, because
  "the summary CSVs read the flat names"; that reason did not hold — nothing outside the folder ever
  opened a bbh CSV by path. **mmlu is half-converted** (`gemma/ xai/ openai/ llama/` exist; the other
  providers are still flat beside the data, and their runners have not been checked for the cwd bug
  below) and **DocVQA has per-model folders** (`gemma_DocVQA/`, `qwen_DocVQA/`). Check the
  benchmark's own page before assuming either shape.
- **One scorer per benchmark, imported from its core** (`CLAUDE.md`). bbh is where this was broken
  and is now the worked example: five of its eight runners scored strictly, and rescoring the same
  stored responses moved three models by 0.19–0.64. Neither mmlu nor DocVQA has a shared core yet —
  check before comparing their models.
- **A runner that moves into a subfolder must stop resolving paths against the cwd**, or it finds no
  data and writes an empty result while the job still exits `COMPLETED 0:0`. `bbh/kimi/kimi_outlog`
  is that failure, recorded: 20 splits, every one `No such file or directory`. bbh's runners now
  anchor on `__file__`; mmlu's subfolder runners have not been checked.
- **One JSON per task or subject**, so the task list is `ls *.json`, not a constant in the code.
- **Provider coverage is uneven and is a scope decision, not an oversight** — see `PLAN.md`.

| Benchmark | Page | Unit | Scoring |
|---|---|---|---|
| DocVQA | [docvqa.md](docvqa.md) | 5,349 validation questions + page images | ANLS |
| BIG-Bench Hard | [bbh.md](bbh.md) · [bbh-parameters.md](bbh-parameters.md) · [bbh-scoring.md](bbh-scoring.md) | 20 task files, 4,833 items | one lenient matcher, `lenient_v5` — read bbh-scoring.md before comparing any two models, or before trusting a suspiciously low task score |
| MMLU | [mmlu.md](mmlu.md) | 13 subject files | accuracy |

DocVQA is the outlier: it is sharded, it carries 3.5 GB of page images, and it is the only one here
with its own incident log.
