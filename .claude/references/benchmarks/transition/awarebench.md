# AwareBench — benchmark card

Mission analysis, formulation and planning. Upstream `HowieHwong/Awareness-in-LLM`. No LLM judge
since 2026-10-05: its only judged rows were dropped. No generation has run.

## Paths

| | Path |
|---|---|
| Local | `Transition_processes_benchmarks/Awareness_in_LLM` |
| Quest | not verified — nothing points to a remote copy, but Quest was not checked (out of scope 2026-08-22). Confirm before assuming a remote path |

## The one thing to know first: this folder holds two benchmarks

| | `dataset/AwareEval.json` | `New/` |
|---|---|---|
| Items | **4,075** | 6,580 across 37 files |
| Documented | yes, README + paper | no — upstream never describes it |
| Runs as-is | yes; scoring is ours to write | no |
| In scope | **yes** | **no** — decided by the user 2026-08-05 |

They share no taxonomy, file format or metric. "The awareness benchmark" is ambiguous; say which.
`New/` remains documented in `AWARENESS_NOTES.md` §3–§4 as out-of-scope material, pending a separate
confirmation with the advisor about whether it is wanted at all.

## The judge

**None is run.** The 60 `mission_open-ended` rows were the only LLM-judged rows, and the user dropped
them on 2026-10-05.
- **Consequences** (budget, the two-term `mission`, and why the headline is no longer the paper's):
  `AWARENESS_NOTES.md` §5.0.
- **The judge we chose not to run:** `LLM_as_judge/JUDGE_RECORD.md` §3.

## Budget

Per model: **4,015 generation calls, no judge calls**.

## What exists, and what does not

`Output_template/` holds the result CSVs the run must produce, except `*_mission_open-ended.csv`,
which is no longer produced;
`awareness_paper_baseline.csv` holds the published numbers to check against. There is **no runner and
no scoring code** — upstream points at the external `trustllm` package, but every metric is stated in
the paper §5.1, so a local scorer is straightforward.

## Read this before touching it

`AWARENESS_NOTES.md` in the folder is deep and authoritative. Four sections change what a run means:
§2.5 (the schema changes per dimension — one parser cannot read the file), §2.7 (the released file is
not the file the paper evaluated), §2.8 (`perspective` is scored second-order only), §4 (what is
broken as delivered). §5 is the run plan.
