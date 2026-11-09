# mpgt-eval — benchmark card

Multi-party goal tracking, for goal specification. Upstream `AddleseeHQ/mpgt-eval`, vendored
2026-08-19. **Scored by human review** — no LLM judge. The paper's metric is exact match on the gold
`[MASK]` strings with the residue reviewed by a person (`LLM_as_judge/JUDGE_SELECTION.md` §1); **no
scoring code ships in the repo** — the `gpt/` scripts only write predictions to a CSV. No runner of
ours, no results.

## Paths

| | Path |
|---|---|
| Local | `Transition_processes_benchmarks/Multi-party_Goal_Tracking_bench` |
| Quest | not verified — nothing points to a remote copy, but Quest was not checked (out of scope 2026-08-22). Confirm before assuming a remote path |

## Layout

```
Multi-party_Goal_Tracking_bench/
├── gpt/            zero_shot_dst_only_gpt.py · few_shot_dst_only_gpt.py
│                   zero_shot_goal_only_gpt.py · few_shot_goal_only_gpt.py
├── Preprocessing/  train-test-split.py · dst_and_goal_masking.py · inject_noise_to_dialogue.py
├── mapping/        elan-to-dlm.py · count-turns.py
└── README.md
```

Upstream ships **four prompt conditions** — zero-shot and few-shot, each for DST-only and goal-only.
They are four separate scripts, not four flags, so "running mpgt" means choosing which of the four
is the comparison, and a table that mixes them is not a comparison.

## Expected counts — verified 2026-11-09

Measured from `Preprocessing/data/split-diogs/` and the four `gpt/` scripts. One row is one dialogue
— `src` holds `[MASK]`s inside a `[start]`…`[end]` window, `tgt` is the gold — and one row is one
API call. Every script loops over **six prompt styles** (basic, specific, story, roleplay,
annotation, reasoning), so a condition is six runs unless a style is chosen.

| Condition | Test file | Rows | `[MASK]` slots | Calls per model, upstream loop |
|---|---|---|---|---|
| zero-shot DST-only | `zero-shot/dst-only/dst-only.jsonl` | 29 | 389 (3–27 per row) | 29 × 6 = 174 |
| zero-shot goal-only | `zero-shot/goal-only/goals-only.jsonl` | 29 | 225 (2–16 per row) | 174 |
| few-shot DST-only | `pcent-20/dst-only/run{1,2,3}-test.jsonl` | 23 each | 319 · 302 · 293 | 23 × 6 × 3 = 414 |
| few-shot goal-only | `pcent-20/goal-only/run{1,2,3}-test.jsonl` | 23 each | 178 · 176 · 172 | 414 |

- **The API side is small and the review side is not.** All four conditions are 1,176 calls per
  model; zero-shot with one style is 58. Human review scales with `[MASK]` slots that miss exact
  match, not with rows: zero-shot, one style, is up to 614 slots per model.
- Few-shot prepends the first **two** train dialogues (`range(0, 2)`) and uses runs 1–3 of the five
  splits on disk; `pcent-50` and `pcent-80` are not read by any script.
- `goal-given-dst` and `multi-task` splits exist on disk (29 rows; 210 and 542 slots zero-shot) but
  no script in `gpt/` reads them.
- Upstream calls `gpt-3.5-turbo` at temperature 0.4 through the pre-1.0 `openai.ChatCompletion` API,
  sleeping 45–90 s after every call (it differs per script) — a reference for the prompts, not a
  runner to reuse.
- A zero-shot output with fewer than 29 rows dropped dialogues.

## Before any run

Human review is the scoring step. That makes the cost per item a person's time, not a provider call,
and it means the number cannot be regenerated later from stored responses. Decide who reviews and
against what written criteria **before** generating, or the generation has to be repeated.

`Preprocessing/inject_noise_to_dialogue.py` implies a noise condition in the upstream design —
establish whether it is part of the intended comparison before treating the clean condition as the
whole benchmark.
