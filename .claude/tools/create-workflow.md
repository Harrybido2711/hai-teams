# create-workflow

Adapting a baseline workflow to a task, editing one, or — rarely — adding one. Not a script — the
constraints the `Workflow` tool enforces, each of which cost a launch to discover.

**The five workflows are baselines, not the only allowed shapes.** For a one-off variation, read the
script, change it for the task (its detail page's *Adapting* section says what may change and what
must not), and pass it inline as `script`. Edit the committed file only when the change should hold
next time too — when a run exposes something a workflow should have caught, add the check there.
**A new workflow is the last resort:** a new procedure is a new skill, and the baselines compose it.

**How a baseline is built.** Each phase names its agent with `agentType` and the skill file that
agent follows (`.claude/skills/<name>/SKILL.md`). The script holds no benchmark-specific path, count
or task name — agents read them from the benchmark's page — which is what lets one baseline serve all
ten benchmarks.

## Input

A new file in `.claude/workflows/<name>.js`, committed. Committed matters: a workflow passed inline
to the tool is lost the moment the session ends.

Invoke by path — `Workflow({scriptPath: ".claude/workflows/<name>.js", args: {...}})` — which always
works. `{name: "<name>"}` also resolves, but only from a session that started *after* the file
existed: the registry is built once at startup, so a workflow written mid-session is invisible to it
until the next one.

## Output

A workflow returns an object the planner branches on. Give it a status-like field whose values match
the `STATUS:` vocabularies in `../references/handoffs.md` — do not invent a third wording for a state
the agents already name.

## Preflight

- **`meta` must be a pure literal.** No concatenation, variables or template interpolation;
  `'a' + 'b'` in a field is rejected as a BinaryExpression.
- **Normalise `args` first, then validate and `return` early.** `args` can arrive as a JSON *string*
  rather than an object: the caller writes an object and the script receives the serialised form, so
  every field reads as `undefined` and the guard blames the caller for omitting an argument they did
  supply. Three launches were lost to this on 2026-08-04. Use
  `typeof args === 'string' ? JSON.parse(args) : args`, and treat unparseable input as its own error
  rather than letting it reach the missing-argument message.
- **Every prompt carries the hard rules.** State what its agents must not do — no provider API
  calls, no `sbatch`/`scancel` outside the phase that owns it, no edits outside the target. A
  reviewer once wrote four probe scripts and spent real quota because its prompt did not forbid it.
- **The gate must be able to say no.** `fix-run` returns without submitting when the reviewer
  refuses; `finish-run` returns without recording until the user confirms. A verification phase
  that cannot block is a formality.
- **Test the control flow before the first real launch.** Load the script with stub `agent()`s that
  return each schema's passing values, then flip one gate at a time to its failing value: every
  path should end where the detail page says, and no prompt should contain `undefined`.
- **Register it** in `README.md` in this directory, with a detail file here whose sections are
  Phases / Input / Output / Adapting / When it fails — the doc check enforces both. A tool nothing
  routes to is never used.

## When it fails

| Symptom | Cause |
|---|---|
| rejected before any agent runs | a computed value in `meta` |
| "X is required" for an argument that was passed | `args` arrived as a string and was not normalised |
| a fleet spawns, then each agent discovers the same missing input | validation was not at the top |
| the workflow submits despite a refusal | the gate's verdict is not branched on |
