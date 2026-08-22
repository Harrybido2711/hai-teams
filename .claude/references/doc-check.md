# The documentation check — reading its findings

`python3 .claude/scripts/check_docs.py` prints `FAIL <check> <target>` and exits 1;
`.githooks/pre-commit` runs it, so a commit cannot land while the tree contradicts itself. This page
is what each finding means. The three sync layers it belongs to are in
[sync-and-consistency.md](sync-and-consistency.md).

**Read the report before you commit, not in the same command as the commit.** The first time this
report ran it correctly listed the files that needed the same edit, and they were still shipped
stale, because the check and the `git commit` were chained together and the output arrived after the
commit had landed. Run the check, read it, then commit.

## The findings

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

## What layer 1 cannot catch

The checks are mechanical. **A document that contradicts itself, or another, in prose is invisible to
all of them** — no link breaks, no size is exceeded. Two ways it has happened here, each needing a
habit rather than a check:

- **Editing a file without re-reading what it claims about itself.** `CLAUDE.md` opened with "rules
  only — no routing" while a routing table was being added to it, in the same edit. A file's first
  paragraph is usually a constraint on the file; read it before adding to it.
- **Reading the file whose name looks right instead of the one that runs.** A BBH inventory was
  written from `<provider>_eval.py` when the `.sh` submits `<provider>_finish.py`, and four columns
  came out wrong. Follow the entry point — the sbatch script, the config, the caller — not the
  filename.

## Before editing anything

```bash
python3 .claude/scripts/check_docs.py --impact "<the term you are changing>"
```

The output is the work list. The checker catches broken structure afterwards; only this catches the
file that should have changed and did not — nothing there is *broken*, merely stale, and staleness
is invisible to every mechanical check.

## Declaring an exception

`.claude/doc-exceptions.json`, one entry per finding, `{check, target, why}`. The `why` has to answer
*why two copies is right here*, not restate the finding. Ones that have held up: editing a sweep log
to satisfy a uniqueness rule would falsify the record; an agent cannot rely on receiving `CLAUDE.md`,
so a rule its judgement depends on must be in its own prompt. One that has not: "this one is fine".

An exception whose finding disappears is reported as no longer needed — delete it then.
