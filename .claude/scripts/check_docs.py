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
SIZE_SOFT = 5 * 1024   # a nudge to consider splitting, never a wall — see check_size

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


MODEL_LIMITS = ".claude/references/model-parameters.md"
MODEL_CALLS = (".claude/references/model-calls.md", ".claude/references/provider-gotchas.md")


def called_models():
    """Model ids a runner actually calls, from the sbatch scripts and argparse defaults.

    The runners are the ground truth: a model documented but never called is harmless, a model
    called but never documented is how someone writes the next runner from a reference that cannot
    tell them which client to use.
    """
    ids, pat = set(), re.compile(r"--model[= ]+\"?([A-Za-z0-9][\w./-]+)")
    dflt = re.compile(r"\"--model\"[^)]*?default=\"([^\"]+)\"", re.S)

    # Only OUR runners, identified by living beside a script we submit. Vendored upstream code has
    # its own --model flags -- Wonderbread's run_experiments.py offers GPT4 and Claude3 -- and
    # holding this project's references responsible for documenting those is a false alarm.
    ours = set()
    for f in glob.glob(os.path.join(REPO, "**", "*.sh"), recursive=True):
        if os.sep + ".git" + os.sep in f:
            continue
        try:
            if "#SBATCH" in open(f, encoding="utf-8", errors="replace").read():
                ours.add(os.path.dirname(f))
        except OSError:
            pass

    for d in ours:
        for ext in ("sh", "py"):
            for f in glob.glob(os.path.join(d, "*." + ext)):
                try:
                    text = open(f, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                ids |= set(pat.findall(text)) | set(dflt.findall(text))
    return {i for i in ids if not i.startswith("$")}


def check_models():
    """Every model a runner calls is reachable in both kinds of reference.

    Two different questions get asked about a model and they live in different files: what limits
    must a runner set (model-parameters.md), and how is it reached at all (model-calls.md, or the
    client table in provider-gotchas.md). gpt-5.6-luna was adopted with the first and not the
    second, and nothing noticed until it was asked about directly.
    """
    def has(rel_path, mid):
        full = os.path.join(REPO, rel_path)
        return os.path.exists(full) and mid in open(full, encoding="utf-8", errors="replace").read()

    out = []
    for mid in sorted(called_models()):
        limits = has(MODEL_LIMITS, mid)
        calls = [c for c in MODEL_CALLS if has(c, mid)]
        if not limits and not calls:
            out.append(("models", mid, "called by a runner and documented nowhere"))
        elif not limits:
            out.append(("models", mid, "no row in model-parameters.md — what limits must a runner set?"))
        elif not calls:
            out.append(("models", mid, "no recipe in model-calls.md or provider-gotchas.md — "
                                       "which client, key and id reach it?"))
    return out


def content_size(path):
    """Bytes a reader actually takes in.

    Markdown formatters pad table cells to align the pipes, which can add kilobytes without adding
    a word. The split rule is about how much there is to absorb, so alignment padding inside table
    rows is collapsed before measuring — otherwise the check fights the editor and loses.
    """
    out = 0
    for line in open(path):
        if line.lstrip().startswith("|"):
            line = re.sub(r" {2,}", " ", line)
            line = re.sub(r"-{3,}", "---", line)
        out += len(line.encode("utf-8"))
    return out


def file_budget(path):
    """This file's own size budget, if it declares one.

    A file may set its own with `<!-- size-budget: 8000 -->` on any line. The default is a guideline,
    not a property of the file: a dense lookup table and a page of reasoning do not deserve the same
    number, and the right size is a judgement the author makes, not one a constant makes for them.
    """
    for line in open(path):
        m = re.search(r"<!--\s*size-budget:\s*(\d+)\s*-->", line)
        if m:
            return int(m.group(1))
    return SIZE_SOFT


def check_size(docs):
    """Advisory only. Never blocks a commit.

    This used to fail the build at a fixed 5 KB. On 2026-08-23 that limit was hit four times in two
    days on one file, and each time it was paid for by deleting the sentence that explained *why* a
    rule existed — the check was making the documentation worse in the name of keeping it short. Size
    is now reported so the pressure to split is visible, and left to judgement.
    """
    out = []
    for p in docs:
        if os.sep + "references" + os.sep in p or os.sep + "tools" + os.sep in p:
            n = content_size(p)
            budget = file_budget(p)
            if n > budget:
                raw = os.path.getsize(p)
                note = "" if raw == n else " (%d on disk, the rest is table padding)" % raw
                out.append(("size", rel(p),
                            "%d bytes of content%s, over its %d budget — consider splitting, or "
                            "declare a budget with <!-- size-budget: N -->" % (n, note, budget)))
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


def model_card(mid, docs):
    """Everything the references say about one model, in one place.

    Facts about a model are split across four files by design — limits, invocation, client failures,
    cluster ceilings. That split is right and it makes retrieval cost four greps. This prints them
    together so a runner can be written from one command.
    """
    print("What the references hold on %r\n" % mid)
    hits = 0
    for p in docs:
        lines = [(i + 1, l.strip()) for i, l in enumerate(
            open(p, encoding="utf-8", errors="replace").read().splitlines()) if mid in l]
        if not lines:
            continue
        hits += 1
        print("  %s" % rel(p))
        for n, l in lines[:6]:
            print("      %4d  %s" % (n, l[:150]))
        if len(lines) > 6:
            print("      ... %d more" % (len(lines) - 6))
        print()
    if not hits:
        print("  nothing. If a runner calls it, that is a check_models finding.")
        return 1
    gaps = [f for f in check_models() if f[1] == mid]
    for _, _, why in gaps:
        print("  GAP: %s" % why)
    return 1 if gaps else 0


def main():
    docs = doc_files()
    if len(sys.argv) > 2 and sys.argv[1] == "--impact":
        return impact(sys.argv[2], docs)
    if len(sys.argv) > 2 and sys.argv[1] == "--model":
        return model_card(sys.argv[2], docs)
    report_touched = None
    if len(sys.argv) > 2 and sys.argv[1] == "--touched":
        report_touched = sys.argv[2:]

    findings = (check_links(docs) + check_orphans(docs) + check_structure()
                + check_benchmarks() + check_models() + check_canaries(docs))
    advisories = check_size(docs)
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
    for check, target, msg in advisories:
        print("note  %-11s %s\n            %s" % (check, target, msg))
    print("\n%d checked · %d failing · %d declared · %d size note(s)"
          % (len(docs), len(live), len(excused), len(advisories)))
    if report_touched and not live:
        touched(report_touched, docs)
    return 1 if (live or missing_why) else 0


if __name__ == "__main__":
    sys.exit(main())
