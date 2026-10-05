---
name: quest-sync
description: Push a code change set from local to Quest for any benchmark and prove it landed, md5 on both sides. Use before every submit, after any fix to code that also lives on Quest, and whenever the question is whether Quest matches local.
---

# quest-sync

Phase 3 · run by `executor` · re-checked by `reviewer` · used by the `launch-run` and `fix-run` workflows.
Code flows up; nothing but results ever flows down.

## Steps

1. **Take both paths from the benchmark's page** — never infer the Quest one.
2. **Hash every code file on both sides and join by filename.** The drift list *is* the work list:
   it catches the files you forgot you changed, not only the one you edited.

   ```bash
   L=<local path>; Q=<Quest path>; T=$(mktemp -d)
   cd "$L" && FILES=($(git ls-files '*.py' '*.sh' | grep -v '/results'))
   md5 -r "${FILES[@]}" | awk '{print $2, $1}' | sort > $T/local
   ssh quest "cd $Q && md5sum ${FILES[*]}" 2>$T/err | awk '{print $2, $1}' | sort > $T/quest
   wc -l < $T/local; wc -l < $T/quest; grep -v libcrypto $T/err
   join -a1 -a2 -e MISSING -o 0,1.2,2.2 $T/local $T/quest | awk '$2 != $3'
   ```

   Both counts must be non-zero. An empty join over two empty lists is not a pass.
3. **Is a job running on these files?** `ssh quest squeue -u uwr0681`. If yes and the change affects
   it, cancel first (`kill-and-resync`) — a live process has already imported its modules.
4. **Compile locally:** `python3 -m py_compile` on each drifted `.py`, `bash -n` on each `.sh`.
5. **Transfer the whole drift list together** — the shared core with its runners, always:
   `ssh quest "cat > $Q/$f" < "$L/$f"` per file.
6. **Re-run step 2.** It must print no drift line, with equal non-zero counts.

## Done when

The join in step 6 prints nothing and both counts are equal and non-zero — output pasted.

## Never

- Trust the `PreToolUse` hook for anything but NegotiationToM — it compares only that benchmark.
- Overwrite `.env` on Quest, or copy it off. Transfer only what is listed in step 2.
- Pull a file that exists only on Quest down to local. Report it instead.

## Detail

[quest-cluster.md](../../references/quest-cluster.md) § Transferring and § The pre-submit gate ·
[sync-and-consistency.md](../../references/sync-and-consistency.md) § layer 2
