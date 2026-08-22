# harvest-patterns

Sweep GitHub for Claude Code workflow/agent repos, keep only mechanisms that transfer to a
Python/SLURM eval project, try to refute each, and record adopted **and** rejected in
`../references/external-patterns.md`.

```
Workflow({scriptPath: ".claude/workflows/harvest-patterns.js",
          args: {focus: "unattended supervision", maxRepos: 5}})
```

It proposes and records. It never installs a plugin, edits an agent or workflow, or touches Quest.

## Input

| Field | Required | Default | Notes |
|---|---|---|---|
| `focus` | no | whatever this project is weakest at | what to look for |
| `maxRepos` | no | `5` | how many candidates reach the expensive extract stage |
| `minStars` | no | `80` | floor for being worth reading |
| `activeWithinDays` | no | `365` | not pushed since then is archived in practice |

## Output

`{outcome, focus, searched, screened, not_examined, proposed[{rank,name,repo,verdict,
local_file_affected,smallest_version}], rejected[], gap, recorded_to, next}`

`not_examined` is printed rather than dropped, so a bounded sweep is never mistaken for a complete
one.

## Preflight

- Re-running is cheap: repos already judged in `external-patterns.md` are skipped unless they have
  moved. Check that file before asking for a wider sweep.
- Nothing is applied by this workflow. Each proposal carries `smallest_version` — adopt that, or
  hand it to `executor` as a decided change.

## When it fails

| Return | Means | Do |
|---|---|---|
| `outcome: swept, nothing survived refutation` | the skeptic killed every candidate | a real and useful result; `rejected` is recorded so the next sweep skips them |
| `recorded_to: NOT WRITTEN` | the record phase failed | the findings exist only in the return value — write them before the session ends |
| `not_examined` large | the sweep was bounded | raise `maxRepos` if the focus matters |
