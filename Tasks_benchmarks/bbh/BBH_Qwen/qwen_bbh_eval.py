"""Qwen — BIG-Bench Hard runner.

**There is no scorer in this file.** Scoring is `bbh_eval_core.score_response`, the one lenient
matcher every model in this benchmark is judged by, so no runner can score its model more or less
generously than another. Everything model-specific lives below: a client and a `call`.

Hybrid model: reasoning can be disabled with `reasoning={"enabled": False}`, which is measured to help and is NOT set here yet (`.claude/references/model-parameters.md`).
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from together import Together

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import bbh_eval_core as core  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
MODEL = DEFAULT_MODEL
load_dotenv(core.ENV_PATH)
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))


def call(prompt):
    def once():
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=12500,
            stream=False,
        )
        return r.choices[0].message.content
    return core.retry(once, label="qwen")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run BIG-Bench Hard for Qwen.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model id; it is BOTH what is called and what the result files are named "
                         "after, so a copied folder cannot silently relabel another model's numbers")
    ap.add_argument("--task", default="all", help="'all' or a comma-separated list of task names")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between calls")
    args = ap.parse_args()

    MODEL = args.model
    tasks = core.TASKS if args.task == "all" else [t.strip() for t in args.task.split(",")]
    unknown = [t for t in tasks if t not in core.TASKS]
    if unknown:
        raise SystemExit("unknown task(s): %s\nknown: %s" % (unknown, core.TASKS))

    print("Qwen: model=%s tasks=%d" % (MODEL, len(tasks)), flush=True)
    core.run_tasks(MODEL_DIR, MODEL, call, tasks=tasks, sleep_between=args.sleep)
    print("done ->", os.path.join(MODEL_DIR, "results"), flush=True)
