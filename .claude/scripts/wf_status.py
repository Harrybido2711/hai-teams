#!/usr/bin/env python3
"""Show live progress of a Claude Code workflow run.

There is no `/workflows` panel in this build, so read the run's own transcript files instead. With
no argument it picks the most recently started run.

    python3 .claude/scripts/wf_status.py            # latest run
    python3 .claude/scripts/wf_status.py wf_50d3e8  # a specific run (prefix is enough)
    python3 .claude/scripts/wf_status.py --list     # every run this session
"""
import glob
import json
import os
import sys
import time

BASE = os.path.expanduser("~/.claude/projects/-Users-harrychen-SONIC-hai-teams")


def runs():
    found = glob.glob(os.path.join(BASE, "*", "subagents", "workflows", "wf_*"))
    return sorted(found, key=os.path.getmtime, reverse=True)


def agent_rows(run_dir):
    out = []
    metas = sorted(glob.glob(os.path.join(run_dir, "*.meta.json")), key=os.path.getmtime)
    for meta_path in metas:
        aid = os.path.basename(meta_path)[len("agent-"):-len(".meta.json")]
        try:
            meta = json.load(open(meta_path))
        except Exception:
            meta = {}
        path = os.path.join(run_dir, f"agent-{aid}.jsonl")
        calls, last, result = 0, "", None
        try:
            for line in open(path, errors="ignore"):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                content = (rec.get("message") or {}).get("content")
                if not isinstance(content, list):
                    continue
                for chunk in content:
                    if isinstance(chunk, dict) and chunk.get("type") == "tool_use":
                        calls += 1
                        inp = chunk.get("input") or {}
                        last = inp.get("description") or str(inp.get("command", ""))[:70]
                        if chunk.get("name") == "StructuredOutput":
                            result = inp
        except OSError:
            pass
        mtime = os.path.getmtime(path) if os.path.exists(path) else 0
        out.append({"id": aid, "type": meta.get("agentType", "?"), "calls": calls,
                    "last": last, "result": result, "mtime": mtime})
    return out


def finished_ids(run_dir):
    done = set()
    jpath = os.path.join(run_dir, "journal.jsonl")
    if not os.path.exists(jpath):
        return done
    for line in open(jpath, errors="ignore"):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("type") in ("result", "completed", "finished"):
            done.add(rec.get("agentId"))
    return done


def main():
    all_runs = runs()
    if not all_runs:
        print("no workflow runs found")
        return 1

    if "--list" in sys.argv:
        for r in all_runs:
            age = (time.time() - os.path.getmtime(r)) / 60
            print(f"  {os.path.basename(r):24s} last active {age:6.1f} min ago")
        return 0

    picked = all_runs[0]
    for arg in sys.argv[1:]:
        if arg.startswith("wf_"):
            match = [r for r in all_runs if os.path.basename(r).startswith(arg)]
            if not match:
                print(f"no run matching {arg}")
                return 1
            picked = match[0]

    rows = agent_rows(picked)
    done = finished_ids(picked)
    live = sum(1 for r in rows if r["id"] not in done and r["result"] is None)
    print(f"{os.path.basename(picked)}   {len(rows)} agent(s), "
          f"{len(rows) - live} finished, {live} running\n")

    for i, r in enumerate(rows, 1):
        settled = r["id"] in done or r["result"] is not None
        age = time.time() - r["mtime"]
        state = "DONE" if settled else f"running ({age:.0f}s since last write)"
        print(f"  [{i}] {r['type']:10s} {state}")
        print(f"      {r['calls']} tool calls | last: {r['last'][:74]}")
        if r["result"]:
            body = json.dumps(r["result"], ensure_ascii=False)
            print(f"      -> {body[:400]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
