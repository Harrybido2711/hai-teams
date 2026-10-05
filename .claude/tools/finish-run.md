# finish-run

A run has finished: pull it down, audit it, and — **only after the user confirms** — record it in the
benchmark page and both workbooks and sync. The first call stops at the confirmation.

```
Workflow({scriptPath: ".claude/workflows/finish-run.js", args: {
  benchmark: "docvqa", model: "DocVQA_GPT_5.6_Luna"}})
# show the audit to the user; once they confirm:
Workflow({scriptPath: ".claude/workflows/finish-run.js", resumeFromRunId: "<runId>",
          args: {benchmark: "docvqa", model: "DocVQA_GPT_5.6_Luna", confirmed: true}})
```

The resume reuses the pull and audit from cache: their prompts do not depend on `confirmed`.

## Phases

| Phase | Agent | Skill | Stops the workflow when |
|---|---|---|---|
| Pull | `executor` | [pull-results](../skills/pull-results/SKILL.md) | local rows do not match Quest's |
| Audit | `evaluator` | [audit-results](../skills/audit-results/SKILL.md) | a pilot (→ `pilot-audited`); no `confirmed` (→ `awaiting-confirmation`); untrustworthy |
| Record | `executor` | [record-results](../skills/record-results/SKILL.md), [sync-pass](../skills/sync-pass/SKILL.md) | — |
| Log | `tracker` | its own entry template | only runs when the audit listed problems |

## Input

| Field | Required | Notes |
|---|---|---|
| `benchmark`, `model` | yes | page stem and model folder |
| `stage` | no | `full` (default) or `pilot` — a pilot is audited, never recorded |
| `confirmed` | no | `true` only once the user has confirmed this run |

## Output

`{outcome: awaiting-confirmation | pilot-audited | recorded | record-incomplete | blocked | aborted,
audit, record, logged, next}`

## Adapting

Change freely: several models of one benchmark → Pull and Audit per model, one Record; a rescore
instead of a run → replace Pull with [rescore-offline](../skills/rescore-offline/SKILL.md). Keep: the
stop before Record until the user confirms; cells recomputed from per-task files on disk; the
before/after cell diff of every workbook.

## When it fails

| Return | Means | Do |
|---|---|---|
| `blocked` at Pull | row counts differ | the run or the transfer is incomplete — `detail` says which |
| `awaiting-confirmation` | working as designed | show `audit` to the user |
| `blocked` at Audit | untrustworthy even with confirmation | the numbers need fixing, not recording |
| `record-incomplete` | the cell diff or the push failed | read `record.detail` before anything else |
