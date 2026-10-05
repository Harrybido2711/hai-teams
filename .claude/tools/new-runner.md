# new-runner

Phases 1–2: write a runner for one model on one benchmark, smoke-test it locally, review it, commit
it. **Stops before Quest** — the user verifies the scripts, then `launch-run` takes over.

```
Workflow({scriptPath: ".claude/workflows/new-runner.js", args: {
  benchmark: "negotiationtom", model: "NEG_GPT_5.6_Luna", modelId: "gpt-5.6-luna",
  reason: "the OpenAI slot moved to gpt-5.6-luna"}})
```

## Phases

| Phase | Agent | Skill | Stops the workflow when |
|---|---|---|---|
| Brief | `summarizer` | [write-runner](../skills/write-runner/SKILL.md) step 1 | the benchmark has no page → `needs-analysis` ([analyse-benchmark](../skills/analyse-benchmark/SKILL.md) first) |
| Write | `executor` | [write-runner](../skills/write-runner/SKILL.md) steps 2–6 | it does not compile, or the smoke test writes no rows |
| Review | `reviewer` ↔ `executor` | the reviewer's own checklist | still not `safe-to-run` after two fix rounds |
| Commit | `executor` | [sync-pass](../skills/sync-pass/SKILL.md) layers 1 and 3 | — |

## Input

| Field | Required | Notes |
|---|---|---|
| `benchmark` | yes | the page stem — `.claude/references/benchmarks/*/<benchmark>.md` |
| `model` | yes | the folder to create or change |
| `modelId` | yes | the provider's id; `check_docs.py --model` runs on it |
| `reason` | yes | one sentence |
| `copyFrom` | no | the folder to copy; otherwise the brief chooses |

## Output

`{outcome: ready-for-user | needs-change | needs-analysis | blocked | aborted, files, smoke, review,
commit, next}`. `ready-for-user` means reviewed and committed — **not** verified; that is the user's.

## Adapting

Change freely: skip Brief when the brief is already in hand; add a second reviewer lens for a change
to a shared core (it moves every model of the benchmark); widen Write to several models of one
provider. Keep: no `ssh`, transfer or `sbatch` — phase 3 needs the user; the reviewer gate can refuse;
the smoke test opens real rows.

## When it fails

| Return | Means | Do |
|---|---|---|
| `needs-analysis` | no page for this benchmark | run [analyse-benchmark](../skills/analyse-benchmark/SKILL.md), then this again |
| `blocked` | no compile, or zero smoke rows | read `written.detail`; usually a key, an import or a client signature |
| `needs-change` | two review rounds did not converge | read `blockers`; the fix is a decision, not another round |
