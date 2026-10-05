# fix-run

Phase 4, the standing kill-and-resync authorisation end to end: kill a job producing unusable data,
put the **already decided** fix on Quest, gate, resubmit, confirm, record. No need to ask first.

```
Workflow({scriptPath: ".claude/workflows/fix-run.js", args: {
  benchmark: "negotiationtom", model: "NEG_Qwen",
  reason: "105 empty responses; reasoning never disabled on Quest",
  fix: "qwen_neg_eval.py passes reasoning={'enabled': False}"}})
```

**Not for a job that is merely slow** — that is [monitor-run](monitor-run.md). Not for a first
submit — that is [launch-run](launch-run.md).

## Phases

| Phase | Agent | Skill | Stops the workflow when |
|---|---|---|---|
| Observe | `watcher` | [check-run](../skills/check-run/SKILL.md) | the observer returns nothing — nothing has been changed |
| Stop | `executor` | [kill-and-resync](../skills/kill-and-resync/SKILL.md) step 2, [submit-run](../skills/submit-run/SKILL.md) step 2 | `scancel` unconfirmed, or another job was affected |
| Sync | `executor` | [quest-sync](../skills/quest-sync/SKILL.md) | — the gate checks it |
| Gate | `reviewer` | its own checklist, on the fix diff and the resubmit | anything fails |
| Resubmit | `executor` | [submit-run](../skills/submit-run/SKILL.md) step 4 | — |
| Confirm | `watcher` | [check-run](../skills/check-run/SKILL.md) | — compares against the stopped job |
| Record | `tracker` | its own entry template | — |

## Input

| Field | Required | Notes |
|---|---|---|
| `benchmark`, `model` | yes | page stem and model folder |
| `reason` | yes | why it is being killed, with the number |
| `fix` | yes | what changed locally, and where. No fix decided → do not kill yet |
| `jobId`, `script` | no | skip discovery; the resubmit script defaults to the one the job used |

## Output

`{outcome: resubmitted-healthy | resubmitted-check-again | blocked | aborted, cancelled_job,
disposition, archive, sync, new_job, health, recorded}` — `disposition` is archived, pruned or kept.

## Adapting

Change freely: several models broken by one shared-core bug → Observe and Stop per model, one Sync,
one Gate, Resubmit per model; a fix that changes the prompt forces `archived`. Keep: scancel before
any transfer; never delete rows; the reviewer gate that can refuse; confirm by rows written.

## When it fails

| Return | Means | Do |
|---|---|---|
| `aborted: … decide the fix` | `fix` missing | decide it first; killing without one only loses the slot |
| `aborted: … cancelled` | `scancel` unverified; nothing else touched | verify by hand |
| `aborted: … another job` | blast radius exceeded the target | stop; this one is for a human |
| `blocked` at Gate | the resubmit would fail too | fix the `blockers`; the job is already cancelled, so nothing is burning |
