# verify-change

Attack a change looking for the paths where it silently does nothing, judge what must be fixed
before it is trusted, record the findings. Run it **before** a real run depends on the change.

```
Workflow({scriptPath: ".claude/workflows/verify-change.js",
          args: {change: "retry now treats an empty body as a failure, not a success",
                 files: ["neg_eval_core.py"]}})
```

## Input

| Field | Required | Default | Notes |
|---|---|---|---|
| `change` | yes | — | what the change is meant to *guarantee*, a few sentences. Not the diff |
| `files` | no | discovered from git | the files to read |
| `probes` | no | — | specific attacks to try, appended to the standard ones |

## Output

`{change, sound, gaps[], unverified[], blocking[], deferrable[], recommendation,
jobs_on_old_code, issues_updated}`

`jobs_on_old_code` is the one people forget: a change verified locally means nothing for a job that
imported its modules an hour ago.

## Preflight

- The change is written and compiles (`python3 -m py_compile`, `bash -n`). This tool attacks a
  change; it does not write or repair one.
- State the guarantee, not the edit. "Added a check on line 90" gives the reviewer nothing to
  refute; "an empty body can no longer be counted as a successful call" does.

## When it fails

| Return | Means | Do |
|---|---|---|
| `aborted: args.change is required` | the guarantee was not stated | describe what it must prevent |
| `blocking` non-empty | the change does not hold on some path | fix those before any run relies on it |
| `unverified` non-empty | a claim nobody could confirm either way | treat as unproven, not as passed |
| `sound: true`, `jobs_on_old_code` non-empty | the code is right, the running job is not | that job's rows are still on the old behaviour |
