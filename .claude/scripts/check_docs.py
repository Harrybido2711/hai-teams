#!/usr/bin/env python3
"""Local documentation consistency check.

One instruction usually touches more files than the instruction names. This script is the
mechanical half of catching that: it will not tell you whether a sentence is still true, but it
will not let a link rot, an orphan appear, a structure break, or the same number quietly grow a
second home.

    python3 .claude/scripts/check_docs.py              # check; exit 1 on an undeclared failure
    python3 .claude/scripts/check_docs.py --impact md5 # what would a change to "md5" touch?

Every finding is either fixed or declared in .claude/doc-exceptions.json with a reason. A finding
that is neither is an error, so exceptions accumulate visibly instead of silently.
"""
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXC_PATH = os.path.join(REPO, ".claude", "doc-exceptions.json")
SIZE_LIMIT = 5 * 1024  # the project's own rule: past ~5 KB, split

# Facts that must have exactly one home. Numbers and incident phrases only — never a concept,
# which legitimately appears in a rule, a contract and a glossary at once.
NEG = ".claude/references/benchmarks/interpersonal/negotiationtom.md"
QC = ".claude/references/quest-cluster.md"
CANARIES = {
    # phrase: (the one file that owns it, what it is)
    "56,774": (QC, "the unattended-watcher data-loss incident"),
    "6 of 32": (NEG, "the 2026-07-29 stale-file count"),
    "14,138": (NEG, "NegotiationToM's full-run row count"),
    "4,618": (NEG, "NegotiationToM's intention row count"),
    "3h10m": ("CLAUDE.md", "the Qwen reasoning-token incident"),
    "watcher checkpoint": ("CLAUDE.md", "the git add -A incident"),
    "BinaryExpression": (".claude/tools/create-workflow.md", "the Workflow meta constraint"),
    "$16.99": (".claude/references/reasoning-cost.md", "the reasoning-vs-completion cost split"),
}


def rel(p):
    return os.path.relpath(p, REPO)


def doc_files():
    out = [os.path.join(REPO, f) for f in ("CLAUDE.md", "PLAN.md", "README.md")]
    out += glob.glob(os.path.join(REPO, ".claude", "**", "*.md"), recursive=True)
    out += glob.glob(os.path.join(REPO, "LLM_as_judge", "*.md"))
    skip = (os.sep + "memory" + os.sep, os.sep + "patches" + os.sep)
    return sorted(p for p in out if os.path.exists(p) and not any(s in p for s in skip))


def load_exceptions():
    if not os.path.exists(EXC_PATH):
        return {}, []
    raw = json.load(open(EXC_PATH))
    bad = [e for e in raw if not e.get("why")]
    return {(e["check"], e["target"]): e for e in raw}, bad


def check_links(docs):
    out = []
    for p in docs:
        base = os.path.dirname(p)
        for text, tgt in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", open(p).read()):
            if tgt.startswith(("http", "#", "mailto:")):
                continue
            if not os.path.exists(os.path.normpath(os.path.join(base, tgt.split("#")[0]))):
                out.append(("link", rel(p), "[%s](%s) points at nothing" % (text, tgt)))
    return out


def check_orphans(docs):
    """A reference or tool file nothing routes to is never read."""
    indexes = [p for p in docs if os.path.basename(p) in ("README.md", "INDEX.md")]
    blob = "".join(open(p).read() for p in indexes)
    out = []
    for p in docs:
        name = os.path.basename(p)
        routed_dir = os.sep + ".claude" + os.sep in p and (
            os.sep + "references" + os.sep in p or os.sep + "tools" + os.sep in p
        )
        if routed_dir and name != "README.md" and name not in blob:
            out.append(("orphan", rel(p), "exists but no index links to it"))
    return out


def check_structure():
    out = []
    tools = os.path.join(REPO, ".claude", "tools")
    readme = os.path.join(tools, "README.md")
    body = open(readme).read() if os.path.exists(readme) else ""
    for js in sorted(glob.glob(os.path.join(REPO, ".claude", "workflows", "*.js"))):
        name = os.path.basename(js)[:-3]
        if "`%s`" % name not in body:
            out.append(("structure", rel(js), "no row in tools/README.md"))
        if not os.path.exists(os.path.join(tools, name + ".md")):
            out.append(("structure", rel(js), "no detail file tools/%s.md" % name))
    want = ["Input", "Output", "Preflight", "When it fails"]
    for md in sorted(glob.glob(os.path.join(tools, "*.md"))):
        if os.path.basename(md) == "README.md":
            continue
        got = re.findall(r"^## (.+)$", open(md).read(), re.M)
        if got != want:
            out.append(("structure", rel(md), "sections are %s, want %s" % (got, want)))
    return out


def check_benchmarks():
    """Nothing under benchmarks/ is unreachable.

    A benchmark page is linked from the master index; a supporting detail file beside one need
    only be linked from its group index. Either router counts — what is forbidden is neither.
    """
    root = os.path.join(REPO, ".claude", "references", "benchmarks")
    index = os.path.join(root, "README.md")
    if not os.path.exists(index):
        return [("benchmarks", rel(root), "no README.md index")]
    master = open(index).read()
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*.md"))):
        if os.path.basename(p) == "README.md":
            continue
        group_dir = os.path.dirname(p)
        name = os.path.basename(p)
        group_index = os.path.join(group_dir, "README.md")
        group = open(group_index).read() if os.path.exists(group_index) else ""
        in_master = "%s/%s" % (os.path.basename(group_dir), name) in master
        if not in_master and name not in group:
            out.append(("benchmarks", rel(p),
                        "neither the master index nor %s/README.md links it"
                        % os.path.basename(group_dir)))
    return out


def check_size(docs):
    out = []
    for p in docs:
        if os.sep + "references" + os.sep in p or os.sep + "tools" + os.sep in p:
            n = os.path.getsize(p)
            if n > SIZE_LIMIT:
                out.append(("size", rel(p), "%d bytes; the rule is to split past ~%d" % (n, SIZE_LIMIT)))
    return out


def check_canaries(docs):
    out = []
    # Normalise whitespace first: a phrase broken across a line wrap is still the same fact,
    # and the first version of this check missed one for exactly that reason.
    flat = {rel(p): re.sub(r"\s+", " ", open(p).read()) for p in docs}
    for phrase, (owner, what) in CANARIES.items():
        hits = [f for f, body in flat.items() if phrase in body]
        if owner not in hits:
            out.append(("canary", "%s @ %s" % (phrase, owner),
                        "%s left its owner; move it back or change the owner" % what))
        for h in hits:
            if h != owner:
                out.append(("canary", "%s @ %s" % (phrase, h),
                            "%s is owned by %s" % (what, owner)))
    return out


def touched(paths, docs):
    """Given the files being committed, name the documents that talk about them and are not.

    This is the drift detector. A contradiction between two documents almost always starts as one
    of them being edited while the other, which describes it, was not — and no link is broken and
    no size is exceeded, so nothing else here notices.
    """
    staged = {os.path.normpath(p) for p in paths}
    flat = {rel(p): open(p).read() for p in docs}
    hits = []
    for p in sorted(staged):
        name = os.path.basename(p)
        if not name.endswith(".md") and not name.endswith(".py"):
            continue
        stem = name[:-3] if name.endswith(".md") else name
        others = [f for f, body in flat.items()
                  if os.path.normpath(os.path.join(REPO, f)) not in
                  {os.path.normpath(os.path.join(REPO, q)) for q in staged}
                  and (name in body or stem in body)]
        if others:
            hits.append((rel(os.path.join(REPO, p)) if not p.startswith(REPO) else rel(p), others))
    if not hits:
        return 0
    print("\nAlso mentions what you are committing — check these did not need the same edit:")
    for target, others in hits:
        print("  %s" % target)
        for o in others:
            print("      %s" % o)
    print("  (a report, not a failure: a mention often needs nothing)")
    return 0


def impact(term, docs):
    print("Files mentioning %r — this is the work list, not a suggestion:\n" % term)
    n = 0
    for p in docs:
        lines = [i + 1 for i, l in enumerate(open(p).read().splitlines()) if term in l]
        if lines:
            n += 1
            print("  %-58s lines %s" % (rel(p), ", ".join(map(str, lines[:8]))))
    print("\n%d file(s). Change them together or say in the commit why one was left." % n)
    return 0


def main():
    docs = doc_files()
    if len(sys.argv) > 2 and sys.argv[1] == "--impact":
        return impact(sys.argv[2], docs)
    report_touched = None
    if len(sys.argv) > 2 and sys.argv[1] == "--touched":
        report_touched = sys.argv[2:]

    findings = (check_links(docs) + check_orphans(docs) + check_structure()
                + check_benchmarks() + check_size(docs) + check_canaries(docs))
    exc, missing_why = load_exceptions()

    live = [f for f in findings if (f[0], f[1]) not in exc]
    excused = [f for f in findings if (f[0], f[1]) in exc]
    stale = [k for k in exc if k not in {(f[0], f[1]) for f in findings}]

    for check, target, msg in live:
        print("FAIL  %-11s %s\n            %s" % (check, target, msg))
    if missing_why:
        print("FAIL  exceptions  %d entr(ies) in doc-exceptions.json carry no 'why'" % len(missing_why))
    for k in stale:
        print("note  exception no longer needed: %s %s" % k)
    print("\n%d checked · %d failing · %d declared" % (len(docs), len(live), len(excused)))
    if report_touched and not live:
        touched(report_touched, docs)
    return 1 if (live or missing_why) else 0


if __name__ == "__main__":
    sys.exit(main())
