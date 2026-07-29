#!/usr/bin/env python3
"""PreToolUse gate: never submit a SLURM job while local code differs from Quest.

Wired into .claude/settings.json as a PreToolUse hook on the Bash tool. Claude Code matches hooks
by tool *name* only, so the "does this command contain sbatch" test lives here rather than in the
matcher.

Contract with check_quest_sync.py:

    rc 0  in sync           -> stay out of the way: exit 0, no decision, normal permission flow
    rc 1  drift found       -> BLOCK, printing which files differ
    rc 2  check did not run -> ALLOW, with a loud warning

Anything unexpected -- the checker crashing, timing out, python3 missing -- is treated as rc 2.
A gate that cannot perform its check must never become a wall that stops all work; it says so
loudly and gets out of the way.

Blocking uses exit code 2 + stderr, which is the oldest and most broadly supported hook mechanism,
because a deny that silently fails to parse would let a bad submit through. The warning path uses
exit 0 + JSON on stdout, where an unparsed payload merely degrades to visible plain text and the
command still runs.
"""
import json
import os
import subprocess
import sys

CHECKER = "/Users/harrychen/SONIC/hai-teams/.claude/scripts/check_quest_sync.py"
TRIGGER = "sbatch"
TIMEOUT = 240  # the checker's own ssh round trip is capped at 180s


def allow():
    sys.exit(0)


def allow_with_warning(summary, detail):
    """Let the command through, but make sure the operator sees why the gate was blind."""
    body = summary if not detail else summary + "\n" + detail
    payload = {
        "systemMessage": body,
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": (
                "The pre-submit Quest sync check could not run, so this sbatch command was NOT "
                "verified against Quest. Confirm by hand that the code on Quest matches local "
                "before trusting the job's output.\n" + body
            ),
        },
    }
    print(json.dumps(payload))
    sys.exit(0)


def block(detail):
    print(detail, file=sys.stderr)
    sys.exit(2)  # PreToolUse: 2 blocks the call and feeds stderr back


def read_command():
    """Return the Bash command string, or None when this event is not ours to judge."""
    raw = sys.stdin.read()
    event = json.loads(raw)  # a schema change should be noisy, not silently disable the gate
    if event.get("tool_name") != "Bash":
        return None
    tool_input = event.get("tool_input") or {}
    return tool_input.get("command") or ""


def main():
    try:
        command = read_command()
    except Exception as exc:
        allow_with_warning(
            "sbatch sync gate: could not read the hook payload (%s: %s)." % (type(exc).__name__, exc),
            "The gate is not inspecting commands. Check .claude/scripts/sbatch_sync_gate.py.",
        )

    if command is None or TRIGGER not in command:
        allow()

    argv = ["python3", CHECKER, "--quiet"]
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=TIMEOUT)
    except FileNotFoundError:
        # python3 not on the hook's PATH; the interpreter running this gate will do.
        try:
            proc = subprocess.run(
                [sys.executable, CHECKER, "--quiet"], capture_output=True, timeout=TIMEOUT
            )
        except Exception as exc:
            allow_with_warning(
                "sbatch sync gate: could not run the sync check (%s: %s)." % (type(exc).__name__, exc),
                "",
            )
    except subprocess.TimeoutExpired:
        allow_with_warning(
            "sbatch sync gate: the Quest sync check timed out after %ds." % TIMEOUT,
            "Quest may be unreachable or slow. The submit was ALLOWED unverified.",
        )
    except Exception as exc:
        allow_with_warning(
            "sbatch sync gate: the sync check failed to start (%s: %s)." % (type(exc).__name__, exc),
            "",
        )

    out = proc.stdout.decode(errors="replace").strip()
    err = proc.stderr.decode(errors="replace").strip()
    err = "\n".join(l for l in err.splitlines() if "libcrypto" not in l).strip()
    detail = "\n".join(p for p in (out, err) if p)

    if proc.returncode == 0:
        allow()

    if proc.returncode == 1:
        block(
            "BLOCKED: local NegotiationToM code differs from Quest -- do not submit yet.\n\n"
            + detail
            + "\n\nTransfer the files above to %s, verify with md5sum, then submit. Send "
            "neg_eval_core.py together with the runners: the runners import from it, so a runner "
            "without the core dies at import.\n"
            "Re-run the check by hand with:\n  python3 %s" % (
                "/gpfs/projects/p32983/NegotiationToM", CHECKER
            )
        )

    allow_with_warning(
        "sbatch sync gate: the Quest sync check could not run (exit %s) -- this submit was NOT "
        "verified against Quest." % proc.returncode,
        detail,
    )


if __name__ == "__main__":
    main()
