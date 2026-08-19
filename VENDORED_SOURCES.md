# Where the vendored benchmarks came from

Each benchmark folder in this repo is a **copy** of an upstream project, not a clone: its `.git` is
removed so the files belong to `hai-teams` and to no other history. That makes the folders tracked
normally by this repo — but it also destroys the only record of *which* upstream version was taken,
so that record lives here instead.

Add a row before removing a `.git`. A folder with no row is a folder nobody can re-derive.

| Folder | Upstream | Version taken | How |
|---|---|---|---|
| `Transition_processes_benchmarks/Awareness_in_LLM` | https://github.com/HowieHwong/Awareness-in-LLM | `07598ff` | clone, `.git` removed (see `AWARENESS_NOTES.md`) |
| `Transition_processes_benchmarks/Multi-party_Goal_Tracking_bench` | https://github.com/AddleseeHQ/mpgt-eval | `f7cd1d08f0e149f88230f7690f57d731b4f8f8a4` · 2023-11-03 · "updated readme" | clone 2026-08-19, `.git` removed |
| `Transition_processes_benchmarks/LLMs-Planning_bench` | https://github.com/karthikv792/LLMs-Planning | not recorded — folder arrived as `LLMs-Planning-main`, i.e. a ZIP of the default branch | ZIP download |
| `Action_processes_benchmarks/Multi-challenge_bench` | https://github.com/ekwinox117/multi-challenge | `5ccefcca6a39020d66c1383c4e6a809cb07afa33` · 2025-02-05 · "fixed axis names, minor bugs." | clone 2026-08-19, `.git` removed |
| `Action_processes_benchmarks/Wonderbread_bench` | https://github.com/HazyResearch/wonderbread | `ed052c67aeada04167cdfe92ff8de454aa94627a` · 2025-07-30 · "Update README.md" | clone 2026-08-19, `.git` removed |
| `Interpersonal_processes_benchmarks/NegotiationToM` | https://github.com/HKUST-KnowComp/NegotiationToM | not recorded | — |
| `Interpersonal_processes_benchmarks/EmoBench` | https://github.com/Sahandfer/EmoBench | not recorded — folder arrived as `EmoBench-master`, i.e. a ZIP of the default branch | ZIP download |
| `Tasks_benchmarks/DocVQA` | https://www.docvqa.org/datasets | not recorded | — |
| `Tasks_benchmarks/bbh` | BIG-Bench Hard | not recorded | — |
| `Tasks_benchmarks/mmlu` | https://github.com/hendrycks/test | not recorded | — |

"not recorded" is honest and useful: it says the version is unknown, rather than implying the folder
matches upstream HEAD today. Fill a row in when a folder is next refreshed.
