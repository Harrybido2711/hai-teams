#!/usr/bin/env python3
"""Is Gemma's 32% watchdog-kill rate hung connections, or generation running past the ceiling?

Job 8526978 lost 81% of its wall clock to calls that never returned inside the 200s ceiling. The
two possible causes need opposite fixes — lower the ceiling for hangs, cap max_tokens for runaway
generation — and picking wrong costs another multi-day run. The question was argued twice from
aggregates and got a different answer each time, because the only numbers available were sums:

    guess 1  "runaway generation"  from max_tokens=8192 being large
    guess 2  "hung connections"    from a 4-minute row-count sample that was quantised by
                                   --save-every 20, giving 57 tok/s
    actual                         13.5-25 tok/s, at which the 8192 budget needs 328-607s and
                                   therefore CANNOT be excluded as the cause

This measures it instead. Two properties matter and both are deliberate:

  * **max_tokens is left at the production 8192.** A distribution you truncate is censored, and a
    censored tail cannot size the cap that produced it.
  * **The ceiling is 600s, not 200s.** A call killed at 200s reports nothing — no token count, no
    finish_reason, no latency. Raising the ceiling above the 350s a full 8192-token generation
    needs at the observed rate is what converts an invisible kill into an observation.

    python3 measure_gemma_budget.py            # 24 calls
    python3 measure_gemma_budget.py --n 40
"""

import argparse
import json
import os
import random
import signal
import statistics
import sys
import time

from dotenv import load_dotenv
from together import Together

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import Interpersonal_processes_benchmarks.NegotiationToM.neg_eval_core as core  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
client = Together(api_key=os.getenv("TOGETHER_API_KEY"), timeout=900)

MODEL = "google/gemma-4-31B-it"
PROBE_CEILING = 600          # far above the ~350s a full 8192 generation needs at the observed rate
PROD_CEILING = 200           # what the cancelled job used, for "would this have been killed?"
MAX_TOKENS = 8192            # production value, deliberately NOT lowered


class ProbeTimeout(BaseException):
    """BaseException so the SDK's own except-clauses cannot swallow it. See neg_eval_core."""


def _raise(signum, frame):
    raise ProbeTimeout()


def one_call(messages):
    """Return a dict describing exactly what happened, including the ways it can fail."""
    signal.signal(signal.SIGALRM, _raise)
    signal.alarm(PROBE_CEILING)
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0, max_tokens=MAX_TOKENS)
        seconds = time.time() - started
        _, completion, seen = core.usage_from(response)
        content = (response.choices[0].message.content or "").strip()
        return {"outcome": "returned", "seconds": seconds,
                "tokens": completion if seen else None,
                "finish": core.finish_reason_from(response),
                "empty": not content,
                "parsed": isinstance(core.parse_json(content), dict)}
    except ProbeTimeout:
        return {"outcome": "hung", "seconds": PROBE_CEILING}
    except Exception as error:
        return {"outcome": "error", "seconds": time.time() - started,
                "detail": f"{type(error).__name__}: {str(error)[:120]}"}
    finally:
        signal.alarm(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="dialogues (x3 tasks = calls)")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "NegotiationToM.json"), encoding="utf-8") as handle:
        data = json.load(handle)
    random.Random(11).shuffle(data)

    print(f"{args.n} dialogues x 3 tasks = {args.n * 3} calls, max_tokens={MAX_TOKENS}, "
          f"ceiling={PROBE_CEILING}s (production was {PROD_CEILING}s)\n", flush=True)

    results = []
    for i, sample in enumerate(data[:args.n]):
        for task in ("desire", "belief", "intention"):
            if task == "desire":
                messages = core.desire_messages(sample["dialogue"], "agent_1")
            elif task == "belief":
                messages = core.belief_messages(sample["dialogue"], "agent_1", "agent_2")
            else:
                messages = core.intention_messages(sample["dialogue"], "agent_1")
            r = one_call(messages)
            r["task"] = task
            results.append(r)
            note = (f"{r['seconds']:6.1f}s  {r['outcome']:<8}"
                    f"  tokens={r.get('tokens')}  finish={r.get('finish')}"
                    f"  empty={r.get('empty')}  parsed={r.get('parsed')}")
            if r["outcome"] == "error":
                note += f"  {r['detail']}"
            print(f"  [{len(results):>3}] {task:<9} {note}", flush=True)

    returned = [r for r in results if r["outcome"] == "returned"]
    hung = [r for r in results if r["outcome"] == "hung"]
    errored = [r for r in results if r["outcome"] == "error"]
    line = "=" * 78
    print(f"\n{line}\n VERDICT\n{line}")
    print(f"  returned {len(returned)}   hung>{PROBE_CEILING}s {len(hung)}   errored {len(errored)}"
          f"   of {len(results)}")
    if not returned:
        print("\n  *** nothing returned even at a 600s ceiling — this is not a token-budget "
              "problem at all.")
        return

    secs = sorted(r["seconds"] for r in returned)
    toks = sorted(r["tokens"] for r in returned if r["tokens"] is not None)
    trunc = [r for r in returned if r["finish"] == "length"]
    empty = [r for r in returned if r["empty"]]
    would_die = [r for r in returned if r["seconds"] > PROD_CEILING]

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    print(f"\n  latency  (s)     median {statistics.median(secs):.0f}  p90 {pct(secs, .9):.0f}  "
          f"max {secs[-1]:.0f}")
    if toks:
        print(f"  output tokens    median {statistics.median(toks):.0f}  p90 {pct(toks, .9)}  "
              f"p99 {pct(toks, .99)}  max {toks[-1]}")
        rate = sum(toks) / sum(r["seconds"] for r in returned if r["tokens"] is not None)
        print(f"  throughput       {rate:.1f} output tok/s")
    print(f"  truncated at {MAX_TOKENS}: {len(trunc)}   empty content: {len(empty)}")
    print(f"  would have been KILLED by the production {PROD_CEILING}s ceiling: "
          f"{len(would_die)} of {len(returned)}")

    print()
    if len(would_die) + len(trunc) > len(hung):
        print("  => RUNAWAY GENERATION dominates. The kills were long generations, not dead")
        print("     connections. Fix: cap max_tokens so budget/tok-per-sec sits inside the")
        print(f"     ceiling. Suggested max_tokens = {int(pct(toks, .99) * 1.5) if toks else 'n/a'}"
              " (p99 x 1.5); keep the ceiling generous.")
    elif hung:
        print("  => HUNG CONNECTIONS dominate. Calls that returned did so comfortably inside the")
        print(f"     production ceiling, yet {len(hung)} produced nothing in {PROBE_CEILING}s.")
        print("     Fix: lower set_call_timeout so a hang is abandoned sooner. max_tokens is not")
        print("     the binding constraint.")
    else:
        print("  => Neither reproduced in this sample. Do not change the budget on this evidence;")
        print("     re-run with a larger --n, or at the time of day the failing job ran.")
    print(f"{line}\n")


if __name__ == "__main__":
    main()
