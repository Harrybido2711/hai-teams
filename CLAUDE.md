# hai-teams — planner rules

**Read [`.claude/INDEX.md`](.claude/INDEX.md) first.** It orients in one page and routes onward:
[`.claude/tools/README.md`](.claude/tools/README.md) for what can be dispatched,
[`.claude/references/README.md`](.claude/references/README.md) for what to read before acting.

This file holds **rules only** — no routing, no procedure, no project facts. Anything explaining
*how* to do something belongs in a reference; anything explaining *what exists* belongs in the
index. Do not copy either back here.

## Who decides

The planner is this session, not a subagent. **None of this project's six agents holds the `Agent`
tool** — checked against their `tools:` lines, 2026-08-22 — so none can dispatch another, and each
starts with no memory of the last. **Sequencing, and the decision to start or stop a job, stay
here.** Agents report; the planner acts on the report.

That guarantee is a property of those six definitions, not of subagents in general: a general-purpose
agent dispatched with the full tool set can spawn others, and the moment one is used the planner is
no longer the only thing sequencing work.

Every agent ends with a `STATUS:` line from a fixed vocabulary, so a dispatch can be branched on
without re-reading prose. Do not invent a second wording for a state that already has one.

## Standing authorisation: a broken run gets killed, not waited out

Granted 2026-07-29. **Do not ask before doing this:** `scancel` the affected job → fix and verify
locally → overwrite on Quest, confirming with `md5sum` → resubmit.

Letting a known-bad job run to the wall wastes the quota *and* the wall-clock slot. A Qwen pilot
burned 3h10m for 315 rows and 105 empty responses because the real fix had never left the laptop.

**Decide the checkpoint's disposition before resubmitting, and say which you chose.** Resume keeps
every row the old code wrote. One result set holding two configurations is worse than redoing rows.

## Every task ends with a sync pass

Synchronisation is not something big changes get and small ones skip. It is how this project stays
coherent, so **a task is not finished until the pass is done** — the same pass, every time.

| Layer | When it applies | What "done" means |
|---|---|---|
| 1 · local ↔ local | **always** | the consistency check passes, or every finding is declared with a reason |
| 2 · local ↔ Quest | **only when the task changed something that also exists on Quest** — a runner, a shared core, an sbatch script, a config | Quest matches local, proven with `md5sum`, core and runners transferred together |
| 3 · local ↔ git | **always** | committed and pushed to `origin` and `backup` |

A change that touches only local files — documentation, notes, a script that never leaves the laptop
— skips layer 2 and nothing else.

**Skipping is a judgement you state, not one you make silently.** Say which layers you ran and why
any was skipped, in the report. An unstated skip cannot be told apart from having forgotten, and a
fixed pass is only worth having because it removes the chance to forget.

Detail on any of the three, and what to do when a layer fails:
`.claude/references/sync-and-consistency.md`.

## Rules that hold on every task

- **Never submit without proving Quest matches local.** Two ways the automatic hook does not save
  you. It **fails open** — a warning that the check could not run is not a pass, and a stale path
  makes it protect nothing while still looking wired up. And it **only checks NegotiationToM**:
  `check_quest_sync.py` resolves that one directory and globs `NEG_*`, so an `sbatch` for any other
  benchmark passes a gate that compared someone else's files and said nothing about yours. Verified
  2026-08-22 — the hook is live and compares 41 files, all of them NegotiationToM's.
- **Sync the shared core and its runners together, or not at all.** The runners import from the core.
- **Code flows up to Quest, results flow down.** Never the reverse.
- **Name a results directory after the model that produced it.** The user's convention, 2026-08-22.
  It is what makes our output distinguishable from the results a vendored copy shipped with — an
  unnamed `results/` is upstream's, and reporting one as ours is the mistake this prevents.
- **Every finished change is committed and pushed, to `origin` *and* `backup`.** Standing rule from
  the user, 2026-08-22, and it applies to any kind of change — code, results, documentation. A change
  that is finished and not pushed is a change that exists in one place. Do not batch a day's work
  into one commit at the end, and do not leave a modification sitting in the working tree.
  **`upstream` (cpzambo/hai-teams) is the collaborator's and is never pushed to.**
- **Stage explicit paths for git. Never `git add -A`** — an unattended loop that did swept unreviewed
  work into commits named "watcher checkpoint" and pushed them to both remotes. The rule above raises
  how *often* you commit; it does not relax what you are allowed to stage.
- **Judge a run by rows written, not by job state.** SLURM reports RUNNING for a process hung inside
  an API call, and a job can exit `COMPLETED 0:0` with every row empty.
- **Report evidence, not assertion.** Include the output. A partial result described accurately beats
  a claim of completion. Where uncertain, name the observation that would settle it.

## Rules that keep the documentation usable

The three routing files (`INDEX.md`, `references/README.md`, `tools/README.md`) are indexes. They
list and point; they do not explain.

- **A fact lives in exactly one place.** When something moves into a reference or a tool file, delete
  it from where it was — do not leave a summary behind. Two copies drift, and then an agent has to
  decide which is right.
- **Adding a file means adding its routing row in the same edit**, phrased as a condition an agent
  can recognise in its own task, not as a topic name. Nothing routes to it means nothing reads it.
- **Past ~5 KB, split.** That is the rule that produced this structure in the first place.
- **When a run exposes something an agent should have known, edit the file** — the reference for
  knowledge, the workflow for a check that should have caught it. A lesson left in a transcript dies
  with the session.
- **Before editing, grep what you are changing across the whole tree — that grep is the work list.**
  `python3 .claude/scripts/check_docs.py --impact <term>` prints it. An instruction names one file
  and usually touches several: the restructure of 2026-08-22 named `CLAUDE.md` and moved eight.
- **The commit is the checkpoint.** `.githooks/pre-commit` runs `check_docs.py` — links, orphans,
  structure, size, and the facts that must have exactly one home. A finding is either fixed or
  **declared in `.claude/doc-exceptions.json` with a reason**; nothing is left merely known. What
  each finding means and how to clear it: `.claude/references/sync-and-consistency.md`.
