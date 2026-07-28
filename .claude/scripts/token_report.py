#!/usr/bin/env python3
"""Per-task token accounting for this project, read from Claude Code's session transcripts.

A "task" is one user turn plus everything the assistant did before the next user turn, which is the
unit a person actually thinks in ("how much did the Qwen investigation cost?").

    python3 .claude/scripts/token_report.py              # every session, newest first
    python3 .claude/scripts/token_report.py --top 15     # 15 most expensive tasks
    python3 .claude/scripts/token_report.py --session <id>
"""

import argparse
import glob
import json
import os

TRANSCRIPTS = os.path.expanduser(
    "~/.claude/projects/-Users-harrychen-SONIC-hai-teams/*.jsonl"
)

# Public per-1M-token prices. Update when they change; cache reads are the cheap tier and dominate
# long sessions, so ignoring them badly misstates the total.
PRICES = {
    "opus":   {"in": 15.00, "out": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    "sonnet": {"in": 3.00,  "out": 15.00, "cache_write": 3.75,  "cache_read": 0.30},
    "haiku":  {"in": 0.80,  "out": 4.00,  "cache_write": 1.00,  "cache_read": 0.08},
}


def price_for(model):
    name = (model or "").lower()
    for key in PRICES:
        if key in name:
            return PRICES[key]
    return PRICES["sonnet"]


def cost(usage, model):
    p = price_for(model)
    return (
        usage.get("input_tokens", 0) * p["in"]
        + usage.get("output_tokens", 0) * p["out"]
        + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]
        + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
    ) / 1e6


def text_of(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
    return ""


def parse(path):
    """Split a transcript into tasks: one user turn and the assistant work that follows it."""
    tasks = []
    current = None
    for line in open(path, encoding="utf-8"):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role") or entry.get("type")

        if role == "user":
            body = text_of(message)
            # tool results also arrive as user-role entries; only real prompts start a task
            if body and not body.startswith("<") and "tool_use_id" not in str(message)[:200]:
                if current:
                    tasks.append(current)
                current = {"prompt": " ".join(body.split())[:70], "in": 0, "out": 0,
                           "cache_w": 0, "cache_r": 0, "cost": 0.0, "turns": 0, "tools": 0}
            continue

        if role == "assistant" and current is not None:
            usage = message.get("usage") or {}
            if usage:
                current["in"] += usage.get("input_tokens", 0)
                current["out"] += usage.get("output_tokens", 0)
                current["cache_w"] += usage.get("cache_creation_input_tokens", 0)
                current["cache_r"] += usage.get("cache_read_input_tokens", 0)
                current["cost"] += cost(usage, message.get("model"))
                current["turns"] += 1
            content = message.get("content")
            if isinstance(content, list):
                current["tools"] += sum(
                    1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use")
    if current:
        tasks.append(current)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0, help="show only the N costliest tasks")
    ap.add_argument("--session", help="restrict to one session id")
    args = ap.parse_args()

    paths = sorted(glob.glob(TRANSCRIPTS), key=os.path.getmtime, reverse=True)
    if args.session:
        paths = [p for p in paths if args.session in p]
    if not paths:
        print("no transcripts found")
        return

    all_tasks = []
    for path in paths:
        all_tasks += parse(path)

    if args.top:
        shown = sorted(all_tasks, key=lambda t: -t["cost"])[:args.top]
        title = f"{args.top} costliest tasks"
    else:
        shown = all_tasks
        title = f"{len(all_tasks)} tasks, in order"

    print(f"{title}   ({len(paths)} session file(s))")
    print("=" * 104)
    print("%-52s %7s %7s %9s %9s %6s %6s %8s" % (
        "task", "in", "out", "cache_w", "cache_r", "turns", "tools", "USD"))
    print("-" * 104)
    for t in shown:
        print("%-52s %7d %7d %9d %9d %6d %6d %8.3f" % (
            t["prompt"], t["in"], t["out"], t["cache_w"], t["cache_r"],
            t["turns"], t["tools"], t["cost"]))

    print("-" * 104)
    tot = {k: sum(t[k] for t in all_tasks) for k in ("in", "out", "cache_w", "cache_r", "cost",
                                                     "turns", "tools")}
    print("%-52s %7d %7d %9d %9d %6d %6d %8.3f" % (
        f"TOTAL ({len(all_tasks)} tasks)", tot["in"], tot["out"], tot["cache_w"],
        tot["cache_r"], tot["turns"], tot["tools"], tot["cost"]))
    print()
    billed = tot["in"] + tot["out"] + tot["cache_w"] + tot["cache_r"]
    if billed:
        print(f"cache reads are {100 * tot['cache_r'] / billed:.1f}% of all billed tokens "
              f"(the cheap tier — a long session is mostly re-reading its own context)")


if __name__ == "__main__":
    main()
