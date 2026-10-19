# Action processes — Wonderbread · MultiChallenge

Monitoring progress toward goals, and coordination. Both were vendored on 2026-08-19 and they share
three properties that decide how any work here starts:

- **Both are judged by an LLM**, so neither number can be re-derived from the data. Their records are
  §7 of `LLM_as_judge/JUDGE_SELECTION.md`, and nothing from either may enter `Tempo_results.xlsx`
  until that record is filled — the rule in its §7.1. The original judge models are retired; which
  replacement judge to use, and the pilot that compares the four candidates, are its §6.
- **Their runners are the judge pilot's, and they share one core.** Since 2026-10-19 both have a
  `gpt-5.6-luna` generation runner and a judge runner for the four candidate judges, on a 10%
  sample. The judge configurations, call infrastructure and checkpoint are
  `Action_processes_benchmarks/judge_core.py`; the four judges run as one array,
  `run_judges.sh`; `compare_judges.py` compares them; `test_judge_pilot.py` pins the parsers,
  samples, prompts and scorer. No other model has a runner here.
- **Both harnesses fail as vendored.** This is the finding that writing the judge records produced,
  and it is why "just run the upstream code" is not available here. The pilot's runners import
  upstream's prompts, builders and scorers but not its entry points.

| Benchmark | Page | Judge covers | State |
|---|---|---|---|
| Wonderbread | [wonderbread.md](wonderbread.md) | QA **and** SOP Generation | judge pilot: GPT responses generated 2026-10-19 (job 4480407); judges not yet run |
| MultiChallenge | [multichallenge.md](multichallenge.md) | every item | judge pilot: GPT responses generated 2026-10-19 (job 4480406); judges not yet run |
