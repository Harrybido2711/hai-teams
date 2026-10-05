# launch-run

Phase 3: prove Quest matches local, gate, submit a pilot or full run, and gate on its first rows.
**Only after the user has verified the scripts** — the script refuses without `userVerified: true`.

```
Workflow({scriptPath: ".claude/workflows/launch-run.js", args: {
  benchmark: "emobench", model: "EMO_GPT_5.6_Luna", stage: "pilot",
  userVerified: true, reason: "first run at effort=low"}})
```

## Phases

| Phase | Agent | Skill | Stops the workflow when |
|---|---|---|---|
| Sync | `executor` | [quest-sync](../skills/quest-sync/SKILL.md) | drift remains, or either file count is zero |
| Gate | `reviewer` | [quest-sync](../skills/quest-sync/SKILL.md) step 2, [submit-run](../skills/submit-run/SKILL.md) step 2 | anything fails — it decides the checkpoint disposition too |
| Submit | `executor` | [submit-run](../skills/submit-run/SKILL.md) steps 2–4 | `sbatch` returns no job id |
| First rows | `watcher` | [check-run](../skills/check-run/SKILL.md) | — reports `launched` or `launched-unhealthy` |

## Input

| Field | Required | Notes |
|---|---|---|
| `benchmark`, `model` | yes | page stem and model folder |
| `stage` | yes | `pilot` or `full` |
| `userVerified` | yes | `true` — the user's phase-3 gate, stated explicitly |
| `reason` | yes | one sentence |
| `script` | no | otherwise the page's run order decides |
| `gateMinutes` | no | `5` — spacing of the two first-rows samples |

## Output

`{outcome: launched | launched-unhealthy | blocked | launch-failed | aborted, jobs, disposition,
archive, sync, first_rows, next}`

## Adapting

Change freely: submit per-task arrays instead of one (same total shards, so written rows stay valid);
launch several models of one benchmark by running Sync once and Submit per model; lengthen the gate
for a slow model. Keep: the `userVerified` refusal, the md5 proof with non-zero counts, the reviewer
gate that can say no, and a first-rows check judged by rows written, not job state.

## When it fails

| Return | Means | Do |
|---|---|---|
| `aborted: … userVerified` | the user has not verified the scripts | hand them over; do not pass `true` on their behalf |
| `blocked` at Sync | drift or zero files | read `synced`; usually a wrong Quest path — take it from the page |
| `blocked` at Gate | the reviewer found a reason it would fail | fix the `blockers`, run again |
| `launched-unhealthy` | the first rows are bad | [fix-run](fix-run.md) once the fix is decided |
