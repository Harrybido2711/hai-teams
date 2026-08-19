#!/usr/bin/env python3
"""What does capping Gemini's thinking actually save, and does it cost accuracy?

Thinking tokens bill at the output rate but are reported separately as `thoughts_token_count`, so
they are easy to miss — an early projection here came in an order of magnitude under the invoice
for exactly that reason. This runs the same items under each setting and reports **median as well
as mean**, because the distribution is heavy-tailed: a handful of calls spend 1,000+ tokens while
most sit in the low hundreds, and a small sample's mean is meaningless.

    python3 measure_thinking_cost.py                 # 18 dialogues x 2 tasks x 2 settings
    python3 measure_thinking_cost.py --n 40
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import Interpersonal_processes_benchmarks.NegotiationToM.neg_eval_core as core  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

MODEL = "gemini-2.5-flash"
FULL_RUN_CALLS = 14138          # desire 4,760 + belief 4,760 + intention 4,618


def one_call(messages, budget):
    """Returns (prompt, visible, thinking, parsed_ok) or None if the call failed."""
    config = dict(system_instruction=messages[0]["content"], temperature=0,
                  response_mime_type="application/json")
    if budget is not None:
        config["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=messages[1]["content"],
                config=types.GenerateContentConfig(**config))
            usage = response.usage_metadata
            parsed = core.parse_json((response.text or "").strip())
            return (usage.prompt_token_count or 0,
                    usage.candidates_token_count or 0,
                    usage.thoughts_token_count or 0,
                    isinstance(parsed, dict))
        except Exception as error:
            if attempt == 2:
                print(f"    skipped: {type(error).__name__}: {str(error)[:60]}", flush=True)
            time.sleep(5)
    return None


def run(items, budget, label):
    prompts, outs, thinks, ok = [], [], [], 0
    for sample in items:
        for task in ("desire", "belief"):
            if task == "desire":
                messages = core.desire_messages(sample["dialogue"], "agent_1")
            else:
                messages = core.belief_messages(sample["dialogue"], "agent_1", "agent_2")
            result = one_call(messages, budget)
            if result is None:
                continue
            prompt, visible, thinking, parsed = result
            prompts.append(prompt)
            outs.append(visible + thinking)      # what actually gets billed as output
            thinks.append(thinking)
            ok += parsed
    n = len(outs)
    print(f"  {label:<12} n={n:<4} in/call={statistics.mean(prompts):.0f}  "
          f"out/call mean={statistics.mean(outs):.0f} median={statistics.median(outs):.0f} "
          f"max={max(outs)}  thinking/call median={statistics.median(thinks):.0f}  "
          f"parsed={ok}/{n}", flush=True)
    return {"n": n, "in": statistics.mean(prompts), "mean": statistics.mean(outs),
            "median": statistics.median(outs), "max": max(outs), "ok": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18, help="dialogues per setting")
    ap.add_argument("--budget", type=int, default=256)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "NegotiationToM.json"), encoding="utf-8") as handle:
        data = json.load(handle)
    random.Random(11).shuffle(data)
    items = data[:args.n]        # identical items for both settings

    print(f"{args.n} dialogues x 2 tasks x 2 settings = {args.n * 4} calls\n")
    results = {}
    for label, budget in (("uncapped", None), (f"budget={args.budget}", args.budget)):
        results[label] = run(items, budget, label)

    base, capped = results["uncapped"], results[f"budget={args.budget}"]
    print()
    print("full-run projection (%d calls):" % FULL_RUN_CALLS)
    for label, r in results.items():
        print(f"  {label:<12} output tokens: mean-based {r['mean'] * FULL_RUN_CALLS / 1e6:.2f}M, "
              f"median-based {r['median'] * FULL_RUN_CALLS / 1e6:.2f}M")
    print()
    print(f"  saving on the mean   : {100 * (1 - capped['mean'] / base['mean']):.0f}%")
    print(f"  saving on the median : {100 * (1 - capped['median'] / base['median']):.0f}%")
    print(f"  worst call           : {base['max']} -> {capped['max']} tokens")
    print(f"  parse success        : {base['ok']}/{base['n']} -> {capped['ok']}/{capped['n']}"
          "   (must not drop)")
    print("\ninput tokens are unaffected by the thinking budget.")


if __name__ == "__main__":
    main()
