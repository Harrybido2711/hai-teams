#!/usr/bin/env python3
"""Does `reasoning={"enabled": False}` fix Gemma, as it fixed Qwen?

Qwen and Gemma are the same provider, the same client and the same token budget. Qwen finished
14,138 rows at 1h57m per shard. Gemma managed 132 rows per shard in 4h20m before it was cancelled.
The only difference in the two call sites is one argument:

    NEG_Qwen/qwen_neg_eval.py:31    reasoning={"enabled": False}
    NEG_Gemma/gemma_neg_eval.py     (absent)

.claude/references/provider-gotchas.md records that Qwen reached this state for the same reason —
"thinking length varied from ~700 to past 32,768 output tokens for the same prompt at
temperature=0; one pilot produced 60 rows in 7 hours" — and that the shipped fix was that provider
control, explicitly "not prompt wording or a larger token budget".

Gemma's symptoms match exactly: 567 billed output tokens against a visible answer of ~15, i.e. 97%
invisible; 32% of calls exceeding a 200s ceiling; and five shards running only 1.05x faster than
one, which is what happens when the bottleneck is tokens generated rather than requests issued.

This runs the same prompts both ways and reports latency, billed tokens, and — the part that
decides whether the fix is acceptable — whether the parsed answers still agree.

    python3 measure_gemma_reasoning.py            # 6 dialogues x 2 tasks x 2 settings = 24 calls
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
CEILING = 240           # generous; reasoning-off should return in ~1-2s
MAX_TOKENS = 8192       # identical to Qwen's, held constant across both arms


class Timeout(BaseException):
    pass


def _raise(signum, frame):
    raise Timeout()


def one_call(messages, reasoning_off):
    kwargs = dict(model=MODEL, messages=messages, temperature=0, max_tokens=MAX_TOKENS)
    if reasoning_off:
        kwargs["reasoning"] = {"enabled": False}
    signal.signal(signal.SIGALRM, _raise)
    signal.alarm(CEILING)
    started = time.time()
    try:
        response = client.chat.completions.create(**kwargs)
        seconds = time.time() - started
        _, completion, seen = core.usage_from(response)
        content = (response.choices[0].message.content or "").strip()
        parsed = core.parse_json(content)
        return {"ok": True, "seconds": seconds, "tokens": completion if seen else None,
                "finish": core.finish_reason_from(response), "empty": not content,
                "parsed": parsed if isinstance(parsed, dict) else None}
    except Timeout:
        return {"ok": False, "seconds": CEILING, "why": f"no response in {CEILING}s"}
    except Exception as error:
        return {"ok": False, "seconds": time.time() - started,
                "why": f"{type(error).__name__}: {str(error)[:100]}"}
    finally:
        signal.alarm(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "NegotiationToM.json"), encoding="utf-8") as handle:
        data = json.load(handle)
    random.Random(11).shuffle(data)
    items = data[:args.n]

    print(f"{args.n} dialogues x 2 tasks x 2 settings = {args.n * 4} calls, "
          f"max_tokens={MAX_TOKENS} in both arms, ceiling {CEILING}s\n", flush=True)

    arms = {}
    for label, off in (("reasoning ON (current)", False), ("reasoning OFF (Qwen's fix)", True)):
        rows = []
        print(f"--- {label} ---", flush=True)
        for sample in items:
            for task in ("desire", "belief"):
                messages = (core.desire_messages(sample["dialogue"], "agent_1") if task == "desire"
                            else core.belief_messages(sample["dialogue"], "agent_1", "agent_2"))
                r = one_call(messages, off)
                r["task"] = task
                r["uid"] = f"{sample.get('dialogue_id', '?')}_{task}"
                rows.append(r)
                if r["ok"]:
                    print(f"  {r['seconds']:6.1f}s  tokens={r['tokens']}  finish={r['finish']}"
                          f"  empty={r['empty']}  parsed={'yes' if r['parsed'] else 'NO'}",
                          flush=True)
                else:
                    print(f"  {r['seconds']:6.1f}s  FAILED  {r['why']}", flush=True)
        arms[label] = rows
        print(flush=True)

    line = "=" * 76
    print(f"{line}\n VERDICT\n{line}")
    summary = {}
    for label, rows in arms.items():
        good = [r for r in rows if r["ok"]]
        secs = [r["seconds"] for r in good]
        toks = [r["tokens"] for r in good if r["tokens"] is not None]
        summary[label] = {"n_ok": len(good), "n": len(rows),
                          "median_s": statistics.median(secs) if secs else None,
                          "median_tok": statistics.median(toks) if toks else None,
                          "parsed": sum(1 for r in good if r["parsed"])}
        s = summary[label]
        med_s = "n/a" if s["median_s"] is None else "{:.1f}s".format(s["median_s"])
        med_t = "n/a" if s["median_tok"] is None else str(s["median_tok"])
        print(f"  {label:<28} ok {s['n_ok']}/{s['n']}   median {med_s:>7}   "
              f"median tokens {med_t:>6}   parsed {s['parsed']}/{s['n_ok']}")

    on, off = summary["reasoning ON (current)"], summary["reasoning OFF (Qwen's fix)"]
    if on["median_s"] and off["median_s"]:
        print(f"\n  speedup           : {on['median_s'] / off['median_s']:.1f}x")
    if on["median_tok"] and off["median_tok"]:
        print(f"  output tokens     : {on['median_tok']} -> {off['median_tok']} "
              f"({on['median_tok'] / max(off['median_tok'], 1):.0f}x fewer)")

    # Agreement is the acceptance test. A fix that makes it fast by making it wrong is not a fix.
    on_rows = {r["uid"]: r for r in arms["reasoning ON (current)"] if r["ok"] and r["parsed"]}
    off_rows = {r["uid"]: r for r in arms["reasoning OFF (Qwen's fix)"] if r["ok"] and r["parsed"]}
    shared = sorted(set(on_rows) & set(off_rows))
    agree = [u for u in shared if on_rows[u]["parsed"] == off_rows[u]["parsed"]]
    print(f"\n  answers comparable on {len(shared)} items; identical on {len(agree)}")
    if shared:
        print(f"  agreement         : {100 * len(agree) / len(shared):.0f}%")
        for u in shared:
            if on_rows[u]["parsed"] != off_rows[u]["parsed"]:
                print(f"    DIFFERS {u}: on={on_rows[u]['parsed']} off={off_rows[u]['parsed']}")
    print()
    if off["n_ok"] and on["median_s"] and off["median_s"] and on["median_s"] / off["median_s"] > 2:
        print("  => reasoning={'enabled': False} is the fix, exactly as it was for Qwen.")
        print("     Keep max_tokens=8192 and the shared 200s ceiling, matching NEG_Qwen.")
        if shared and len(agree) < len(shared):
            print("     NOTE: answers differ on some items, so the archived rows were written")
            print("     under a different decoding config and MUST NOT be resumed into this run.")
        elif shared:
            print("     Answers are identical on every comparable item, so this changes speed")
            print("     rather than output.")
    else:
        print("  => not reproduced. Do not ship the change on this evidence.")
    print(f"{line}\n")


if __name__ == "__main__":
    main()
