# Keeping things in sync

One instruction usually touches more files than it names, and drift is silent by default. Three
layers, each with its own enforcement point and its own failure mode.

| Layer | What must match | Enforced by | Fires |
|---|---|---|---|
| 1 · local ↔ local | documents with each other | `scripts/check_docs.py` via `.githooks/pre-commit` | every commit |
| 2 · local ↔ Quest | code on disk with code on the cluster | `scripts/check_quest_sync.py` via the `PreToolUse` hook | any command containing the submit keyword |
| 3 · local ↔ git | the working tree with both remotes | the rule in `CLAUDE.md`; `.githooks/post-commit` reports | every finished change |

Layer 1 rides on layer 3 on purpose: since every finished change has to be committed, the commit is
the one moment every change reliably passes through.

## Layer 1 — what each finding means

`python3 .claude/scripts/check_docs.py` prints `FAIL <check> <target>` and exits 1.

| Check | What happened | The fix |
|---|---|---|
| `link` | a relative link points at nothing — usually a file was moved or renamed | repoint the link, or move the file back. Never delete the link to silence it |
| `orphan` | a file in `references/` or `tools/` that no index links to | add its routing row, phrased as a condition an agent can recognise — or delete the file |
| `structure` | a workflow with no row or no detail file, or a detail file whose four sections drifted | add the row and the page in the same edit; keep the sections Input / Output / Preflight / When it fails |
| `benchmarks` | a page exists that the benchmark index does not list | add it to the index table |
| `size` | past the ~5 KB split rule | split it, or declare it with the split recorded as open work |
| `canary … is owned by X` | a fact that must have one home grew a second one | delete the copy, or declare the second home with a reason |
| `canary … left its owner` | the fact moved and the owner was not updated | change the owner in `CANARIES`. This is the normal aftermath of a deliberate move, not an error |
| `exceptions … carry no why` | a declaration with no reason | write the reason. A silent exception is the thing this mechanism exists to prevent |

## Before editing anything

```bash
python3 .claude/scripts/check_docs.py --impact "<the term you are changing>"
```

The output is the work list. The checker catches broken structure afterwards; only this catches the
file that should have changed and did not, because nothing there is *broken* — it is merely stale,
and staleness is invisible to every mechanical check.

## Declaring an exception

`.claude/doc-exceptions.json`, one entry per finding, `{check, target, why}`. The `why` is the whole
point, and it has to answer *why two copies is the right answer here*, not merely restate the
finding.

Reasons that have held up: a sweep log quotes an incident as evidence and editing it would falsify
the record; a group index quotes a member's headline number so a reader can route without opening
two files; an agent cannot rely on receiving `CLAUDE.md`, so a rule its judgement depends on must be
in its own prompt. A reason that has not: "this one is fine".

An exception whose finding disappears is reported as no longer needed — delete it then, so the file
stays a list of live decisions rather than an archive.

## Layer 2 — know what it does not cover

The submit gate **fails open** and currently **checks one benchmark only**; both limits, and the
contract of the hook, are in [quest-cluster.md](quest-cluster.md). Closing the second one is item 5
of `PLAN.md` § Open work and is required before the first submit for any other benchmark. A gate
that reports *in sync* after comparing someone else's files is worse than no gate, because nobody
doubts a green result.

Sync is also not something to do continuously: a live job has already imported its modules, so
replacing code under it changes nothing until it is cancelled and resubmitted.

## Layer 3

Committed and pushed to both `origin` and `backup`, per `CLAUDE.md`. `post-commit` only reports what
is unpushed; it never pushes, because publishing stays a decision someone makes.

## Disagreeing with the check

`git commit --no-verify` exists. Using it is a statement that the fix is the *next* commit, not that
the finding was wrong. If a check is wrong often enough to be routinely bypassed, the check is the
thing to change — edit `check_docs.py` and say so in the commit.
