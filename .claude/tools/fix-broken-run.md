# fix-broken-run

Kill a job that is producing unusable data, fix and resync the code, gate, resubmit, confirm,
record. Covers the standing kill-and-resync authorisation end to end.

```
Workflow({scriptPath: ".claude/workflows/fix-broken-run.js",
          args: {model: "NEG_Qwen", reason: "105 empty responses; reasoning never disabled on Quest",
                 pilot: true}})
```

**Not for a job that is merely slow.** Slow is `check-status`.

## Input

| Field | Required | Default | Notes |
|---|---|---|---|
| `model` | yes | — | the model directory on Quest |
| `reason` | yes | — | why it is being killed, one sentence |
| `pilot` | no | `false` | which sbatch script the resubmit uses |
| `jobId` | no | discovered | pass it to skip discovery |

## Output

`{model, reason, cancelled_job, disposition, archive, sync{in_sync,files},
gate{safe,blockers}, new_job, health, issues_updated}`

`disposition` is the checkpoint decision — resumed, pruned or archived — and `archive` is the
timestamped path when one was made. `new_job` is absent when the gate refused.

## Preflight

- Confirm the job really is producing bad data, not just slow: rows written, non-empty
  `raw_response` rate, null-`pred` count. A job that exits `COMPLETED 0:0` with a full, entirely
  empty row count looks fine in `squeue`.
- Know which fix is going up. This workflow cancels, syncs and resubmits; it does not invent the fix.

## When it fails

| Return | Means | Do |
|---|---|---|
| `aborted: the observer returned nothing` | could not read live state; **nothing was changed** | re-run, or check SSH |
| `aborted: could not confirm the job was cancelled` | `scancel` unverified; nothing else was touched | verify by hand before anything else |
| `aborted: another model appears to have been affected` | blast radius exceeded the target | stop; this one is for a human |
| `gate.safe: false` with `blockers` | the reviewer found the resubmit would fail too | the gate is allowed to say no — fix the blockers, then run it again |
