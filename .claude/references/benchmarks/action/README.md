# Action processes — Wonderbread · MultiChallenge

Monitoring progress toward goals, and coordination. Both were vendored on 2026-08-19 and they share
three properties that decide how any work here starts:

- **Both are judged by an LLM**, so neither number can be re-derived from the data. Their records are
  §7 of `LLM_as_judge/JUDGE_SELECTION.md`, and nothing from either may enter `Tempo_results.xlsx`
  until that record is filled — the rule in its §7.1. It is filled (§7.2–7.4, §7.6), and the user had
  both benchmarks recorded on 2026-11-11 with Haiku 5.5's agreement with humans still unmeasured (§8).
  The original judge models are retired; the
  pilot that compared four replacements is its §6.4, and **the user chose Claude Haiku 5.5 on
  2026-10-31** (its §8).
- **Their runners share one core.** `Action_processes_benchmarks/judge_core.py` holds the judge
  configurations, `Generator` (one tested model's client and parameters), the call infrastructure
  and the checkpoint; each benchmark's `*_eval_core.py::run_cli` is the generation loop every model
  runs, so a model's folder holds only its client. The pilot (10% sample, `gpt-5.6-luna`, four
  judges): `run_judges.sh`, `compare_judges.py`. The full run (every item, the five selected
  models, Haiku 5.5): each folder's `run_mc.sh` / `run_wb.sh`, then `run_judge_full.sh`.
  `test_judge_pilot.py` pins parsers, sets, prompts, scorer and the generator plumbing.
- **Both harnesses fail as vendored.** This is the finding that writing the judge records produced,
  and it is why "just run the upstream code" is not available here. The pilot's runners import
  upstream's prompts, builders and scorers but not its entry points.

| Benchmark | Page | Judge covers | State |
|---|---|---|---|
| Wonderbread | [wonderbread.md](wonderbread.md) | QA **and** SOP Generation | full run done and judged; four models (Deepseek N/A, text-only); **recorded 2026-11-11**, not audited |
| MultiChallenge | [multichallenge.md](multichallenge.md) | every item | full run done and judged; all five, on the same 272 items; **recorded 2026-11-11**, not audited |
