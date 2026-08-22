# mpgt-eval — benchmark card

Multi-party goal tracking, for goal specification. Upstream `AddleseeHQ/mpgt-eval`, vendored
2026-08-19. **Scored by human review** — no LLM judge and no automatic metric. No runner of ours, no
results.

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

## Before any run

Human review is the scoring step. That makes the cost per item a person's time, not a provider call,
and it means the number cannot be regenerated later from stored responses. Decide who reviews and
against what written criteria **before** generating, or the generation has to be repeated.

`Preprocessing/inject_noise_to_dialogue.py` implies a noise condition in the upstream design —
establish whether it is part of the intended comparison before treating the clean condition as the
whole benchmark.
