# create-workflow

Writing a new saved workflow, or editing one. Not a script — the constraints the `Workflow` tool
enforces, each of which cost a launch to discover.

**Improving an existing workflow is editing its file, not writing a new one.** When a run exposes
something a workflow should have caught, add the check there rather than remembering to do it by
hand.

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
- **The gate must be able to say no.** `fix-broken-run` returns without submitting when the reviewer
  refuses. A verification phase that cannot block is a formality.
- **Register it** in `README.md` in this directory, and write its detail file here. A tool nothing
  routes to is never used.

## When it fails

| Symptom | Cause |
|---|---|
| rejected before any agent runs | a computed value in `meta` |
| "X is required" for an argument that was passed | `args` arrived as a string and was not normalised |
| a fleet spawns, then each agent discovers the same missing input | validation was not at the top |
| the workflow submits despite a refusal | the gate's verdict is not branched on |
