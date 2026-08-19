#!/usr/bin/env python3
"""Compare NegotiationToM *code* between this machine and Quest.

Only code is checked. Results, logs and archives are deliberately excluded: those are produced on
Quest and pulled down, so they differ in the normal course of work and would bury a real drift in
false positives — a 206-file comparison on 2026-07-29 returned 53 such differences and exactly 6
that mattered.

Exit codes:  0 = in sync   1 = drift found   2 = the check itself could not run

Usage:
    python3 .claude/scripts/check_quest_sync.py [--quiet]
    python3 .claude/scripts/check_quest_sync.py --watch [SECONDS] [--log PATH]

--watch polls forever, appending one timestamped line per check. Only drift and check failures are
loud, so a log tailed with `grep -E "SYNC-DRIFT|check-failed"` stays silent while things are fine.
"""
import hashlib
import os
import subprocess
import sys
import time

QUEST = "quest"
QDIR = "/gpfs/projects/p32983/NegotiationToM"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_local():
    """Locate the local NegotiationToM folder.

    It has moved once already — on 2026-08-19 the repo was reorganised by team process and the
    folder became Interpersonal_processes_benchmarks/NegotiationToM. A hardcoded path that goes
    stale is worse than it sounds: this checker then finds no code, exits 2, and the pre-submit
    gate fails open with a warning. The check stops protecting anything while still appearing to
    be wired up. So search, and only fall back to a fixed guess if nothing is found.
    """
    for rel in ("Interpersonal_processes_benchmarks/NegotiationToM", "NegotiationToM"):
        cand = os.path.join(REPO, rel)
        if os.path.isdir(cand):
            return cand
    for dirpath, dirnames, _ in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        if os.path.basename(dirpath) == "NegotiationToM":
            return dirpath
    return os.path.join(REPO, "Interpersonal_processes_benchmarks/NegotiationToM")


LOCAL = os.environ.get("NEG_LOCAL_DIR") or _find_local()

# Code only. Anything under results*/ or ending in .txt/.csv/.jsonl is output, not input.
PATTERNS = ("*.py", "NEG_*/*.py", "NEG_*/*.sh", "*.sh")


def local_files():
    import glob

    found = set()
    for pat in PATTERNS:
        for path in glob.glob(os.path.join(LOCAL, pat)):
            rel = os.path.relpath(path, LOCAL)
            if rel.startswith("results") or "/results" in rel:
                continue
            found.add(rel)
    return sorted(found)


def local_hashes(files):
    out = {}
    for rel in files:
        with open(os.path.join(LOCAL, rel), "rb") as fh:
            out[rel] = hashlib.md5(fh.read()).hexdigest()
    return out


def quest_hashes(files):
    # One ssh round trip. Filenames are passed on stdin, never interpolated into the command, so
    # no quoting or word-splitting can mangle the list.
    script = f'cd {QDIR} && xargs -d "\\n" md5sum 2>/dev/null'
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", QUEST, script],
        input="\n".join(files).encode(),
        capture_output=True,
        timeout=180,
    )
    out = {}
    for line in proc.stdout.decode(errors="replace").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 32:
            out[parts[1].strip()] = parts[0]
    if not out:
        err = proc.stderr.decode(errors="replace")
        err = "\n".join(l for l in err.splitlines() if "libcrypto" not in l)
        print(f"could not read hashes from Quest:\n{err.strip()}", file=sys.stderr)
        sys.exit(2)
    return out


def watch(interval, log_path):
    """Poll forever. One line per check; drift and failures are the only loud ones."""
    while True:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--quiet"],
                capture_output=True, timeout=300,
            )
            body = (proc.stdout + proc.stderr).decode(errors="replace").strip()
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            body, rc = "check timed out after 300s", 2

        if rc == 0:
            line = f"[{stamp}] ok - in sync"
        elif rc == 1:
            line = f"[{stamp}] SYNC-DRIFT detected\n{body}"
        else:
            line = f"[{stamp}] check-failed rc={rc}\n{body}"

        with open(log_path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
        time.sleep(interval)


def main():
    if "--watch" in sys.argv:
        i = sys.argv.index("--watch")
        interval = 1800
        if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            interval = int(sys.argv[i + 1])
        log_path = os.path.join(os.path.dirname(LOCAL), "quest_sync.log")
        if "--log" in sys.argv:
            log_path = sys.argv[sys.argv.index("--log") + 1]
        print(f"polling every {interval}s -> {log_path}")
        watch(interval, log_path)

    quiet = "--quiet" in sys.argv
    files = local_files()
    if not files:
        print(f"no code files found under {LOCAL}", file=sys.stderr)
        sys.exit(2)

    local = local_hashes(files)
    remote = quest_hashes(files)

    # A comparison over an empty or truncated list passes trivially. Say the size out loud.
    if len(remote) < len(local):
        missing = sorted(set(local) - set(remote))
    else:
        missing = []
    differ = sorted(f for f in local if f in remote and local[f] != remote[f])

    if not quiet or differ or missing:
        print(f"NegotiationToM code: {len(local)} local / {len(remote)} on Quest")

    if not differ and not missing:
        if not quiet:
            print("in sync")
        return 0

    for f in missing:
        print(f"  MISSING ON QUEST  {f}")
    for f in differ:
        print(f"  DIFFER            {f}  local={local[f][:8]} quest={remote[f][:8]}")
    print(
        f"\n{len(differ) + len(missing)} file(s) out of sync. Transfer local -> Quest before "
        f"submitting any job; send neg_eval_core.py together with the runners, since the runners "
        f"import from it."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
