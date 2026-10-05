#!/usr/bin/env python3
"""Read-only three-way sync audit: local <-> GitHub, and local <-> Quest for every benchmark.

    python3 .claude/scripts/sync_audit.py                 # every benchmark whose page names a Quest path
    python3 .claude/scripts/sync_audit.py bbh docvqa      # just these page stems
    python3 .claude/scripts/sync_audit.py --fetch         # fetch origin and backup first
    python3 .claude/scripts/sync_audit.py --json out.json # also write the full lists

Paths come from each benchmark's page (`| Local |` and `| Quest |` rows), never from this file, so a
moved directory is fixed on its page and nowhere else. Code is compared by md5 (`*.py`, `*.sh`, code
flows up); results by size (`*.jsonl`, `*.csv` under a `results*` folder, results flow down).

It never writes to Quest or to the repo. Exit 1 when code differs or a result exists only on Quest —
not every finding is a fault (a renamed folder, merged shards), so read the lists before acting.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAGES = os.path.join(REPO, ".claude", "references", "benchmarks")
SKIP = ("/results", "results/", "archive", "__pycache__", "/.git/")
NOISE = ("libcrypto", "post-quantum", "store now", "upgraded", "openssh.com/pq")


def is_code(p):
    return p.endswith((".py", ".sh")) and not any(s in "/" + p for s in SKIP)


def is_result(p):
    return "results" in p and p.endswith((".jsonl", ".csv"))


def pages():
    """{stem: (local, quest)} for every page whose Quest row holds an absolute path."""
    out = {}
    for page in sorted(glob.glob(os.path.join(PAGES, "*", "*.md"))):
        text = open(page).read()
        local = re.search(r"^\| Local \| `([^`]+)`", text, re.M)
        quest = re.search(r"^\| Quest \|[^\n]*?`(/[^`]+)`", text, re.M)
        if local and quest:
            out[os.path.basename(page)[:-3]] = (local.group(1), quest.group(1))
    return out


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    err = "\n".join(l for l in r.stderr.splitlines() if not any(n in l for n in NOISE))
    return r.returncode, r.stdout, err


def git_layer(fetch):
    if fetch:
        for remote in ("origin", "backup"):
            run(["git", "fetch", "-q", remote], cwd=REPO)
    head = run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO)[1].strip()
    lines = [f"local HEAD {head}"]
    for remote in ("origin", "backup"):
        rc, out, _ = run(["git", "rev-list", "--left-right", "--count", f"HEAD...{remote}/main"], cwd=REPO)
        ahead, behind = (out.split() + ["?", "?"])[:2] if rc == 0 else ("?", "?")
        lines.append(f"{remote}/main: local ahead {ahead}, behind {behind}")
    dirty = [l for l in run(["git", "status", "--porcelain"], cwd=REPO)[1].splitlines() if l]
    lines.append(f"working tree: {len(dirty)} uncommitted path(s)")
    return lines, dirty


def audit(stem, local_rel, quest):
    ldir = os.path.join(REPO, local_rel)
    tracked = run(["git", "ls-files"], cwd=ldir)[1].split("\n")
    untracked = run(["git", "ls-files", "--others", "--exclude-standard"], cwd=ldir)[1].split("\n")
    lc = {p: hashlib.md5(open(os.path.join(ldir, p), "rb").read()).hexdigest()
          for p in tracked + untracked if p and is_code(p) and os.path.isfile(os.path.join(ldir, p))}
    rc, out, err = run(["ssh", "-o", "BatchMode=yes", "quest",
                        f"cd {quest} && find . -type f \\( -name '*.py' -o -name '*.sh' \\) -print0 | xargs -0 md5sum; "
                        f"echo '@@RESULTS@@'; find . -type f \\( -name '*.jsonl' -o -name '*.csv' \\) -path '*results*' "
                        f"-printf '%P\\t%s\\n'"])
    code_part, _, res_part = out.partition("@@RESULTS@@")
    qc = {}
    for line in code_part.strip().splitlines():
        h, _, p = line.partition("  ")
        p = p[2:] if p.startswith("./") else p
        if is_code(p):
            qc[p] = h
    qr = {}
    for line in res_part.strip().splitlines():
        p, _, s = line.partition("\t")
        if p:
            qr[p] = int(s)
    lr = {}
    for root, _, files in os.walk(ldir):
        for f in files:
            p = os.path.relpath(os.path.join(root, f), ldir)
            if is_result(p):
                lr[p] = os.path.getsize(os.path.join(root, f))
    common = set(lc) & set(qc)
    return {
        "local": local_rel, "quest": quest, "ssh_rc": rc, "ssh_err": err[:300],
        "code_counts": [len(lc), len(qc)],
        "code_differ": sorted(p for p in common if lc[p] != qc[p]),
        "code_local_only": sorted(set(lc) - set(qc)),
        "code_quest_only": sorted(set(qc) - set(lc)),
        "untracked_code": sorted(p for p in untracked if p and is_code(p)),
        "result_counts": [len(lr), len(qr)],
        "results_quest_only": sorted(set(qr) - set(lr)),
        "results_size_differ": sorted(p for p in set(qr) & set(lr) if qr[p] != lr[p]),
        "results_local_only": len(set(lr) - set(qr)),
    }


def main():
    argv = sys.argv[1:]
    fetch = "--fetch" in argv
    out_json = argv[argv.index("--json") + 1] if "--json" in argv else None
    wanted = [a for a in argv if not a.startswith("--") and a != out_json]
    known = pages()
    for w in wanted:
        if w not in known:
            sys.exit(f"no page with Local and Quest paths for '{w}'; known: {', '.join(known)}")
    lines, dirty = git_layer(fetch)
    print("== local <-> GitHub" + ("" if fetch else "  (refs as last fetched; --fetch to refresh)"))
    for l in lines:
        print("   " + l)
    report, bad = {"git": {"summary": lines, "dirty": dirty}}, False
    for stem in wanted or known:
        r = audit(stem, *known[stem])
        report[stem] = r
        print(f"== {stem}: local {r['local']}  <->  Quest {r['quest']}" + (f"  SSH rc={r['ssh_rc']} {r['ssh_err']}" if r["ssh_rc"] else ""))
        print(f"   code    local={r['code_counts'][0]} quest={r['code_counts'][1]}  differ={len(r['code_differ'])} "
              f"local-only={len(r['code_local_only'])} (untracked {len(r['untracked_code'])}) quest-only={len(r['code_quest_only'])}")
        print(f"   results local={r['result_counts'][0]} quest={r['result_counts'][1]}  quest-only={len(r['results_quest_only'])} "
              f"size-differ={len(r['results_size_differ'])} local-only={r['results_local_only']}")
        for key in ("code_differ", "code_quest_only", "results_quest_only", "results_size_differ"):
            for p in r[key][:8]:
                print(f"     {key}: {p}")
            if len(r[key]) > 8:
                print(f"     {key}: … {len(r[key]) - 8} more (--json for all)")
        bad |= bool(r["ssh_rc"] or r["code_differ"] or r["results_quest_only"])
    if out_json:
        json.dump(report, open(out_json, "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
