---
name: sync-pass
description: Close a finished change — consistency check, explicit-path commit, push to origin and backup, and Quest md5 when code that lives there changed. Use at the end of every task, once per finished change, not once per batch.
---

# sync-pass

Phase 5 · run by whoever made the change · the last step of every workflow that changes files.

## Steps

1. **Before editing,** `python3 .claude/scripts/check_docs.py --impact <term>` — the work list.
2. **Layer 1:** `python3 .claude/scripts/check_docs.py`. Read it *before* committing, in a separate
   command. Fix each finding or declare it in `.claude/doc-exceptions.json` with a reason.
3. **Layer 2:** did the change touch a file that also exists on Quest? Yes → `quest-sync`, whole
   set. No → say "layer 2 skipped: local only".
4. **Layer 3:** `git status --short`; stage **explicit paths, only your own changes.** Another
   session may have uncommitted hunks in the same file — `git diff <file>` first; if it is mixed,
   stage only your hunks (`git apply --cached` a patch of just them). **"Your own" means edits this
   session made — ones you can point to in your own tool calls.** A modified file in the tree is not
   evidence it is yours, and neither is a user's question that assumes it is. For anything you did not
   write, `ListAgents` first: if another session in this repo is busy or waiting, ask it before
   staging. 2026-11-09: a session committed a peer's unreviewed runner and judge edits as its own
   (`fc27e5ff`); review then found two Qwen producers able to write one judge file, fixed in `3979c63d`.
   Commit with a heredoc message, then `git push origin main && git push backup main`.
5. **Layer 4:** only when a model finished a benchmark and the user confirmed it → `record-results`.
6. **Report** which layers ran and why any was skipped.

## Done when

The check passes, the commit is on both remotes, and the report names every skipped layer.

## Never

- `git add -A`, or stage a file you have not diffed.
- Push to `upstream`.
- Chain the check and the commit in one command — its findings arrive after the commit lands.

## Detail

[sync-and-consistency.md](../../references/sync-and-consistency.md) ·
[doc-check.md](../../references/doc-check.md)
