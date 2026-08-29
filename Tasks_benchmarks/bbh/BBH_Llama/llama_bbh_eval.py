"""Llama — BIG-Bench Hard runner.

**There is no scorer in this file.** Scoring is `bbh_eval_core.score_response`, the one lenient
matcher every model in this benchmark is judged by, so no runner can score its model more or less
generously than another. Everything model-specific lives below: a client and a `call`.

Not one of the six reported models — bbh is the only benchmark it was run on.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from together import Together

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import bbh_eval_core as core  # noqa: E402

DEFAULT_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
MODEL = DEFAULT_MODEL
load_dotenv(core.ENV_PATH)
client = Together(api_key=os.getenv("LLAMA_API_KEY"))


# What this runner actually passes, written onto every row it produces
# (`.claude/references/model-parameters.md` rule 8). No seed anywhere here — that is
# the gap, not an omission in this line.
CONFIG = {"temperature": 0, "stream": False}


def call(prompt):
    def once():
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=False,
        )
        return r.choices[0].message.content
    return core.retry(once, label="llama")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run BIG-Bench Hard for Llama.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model id; it is BOTH what is called and what the result files are named "
                         "after, so a copied folder cannot silently relabel another model's numbers")
    ap.add_argument("--task", default="all", help="'all' or a comma-separated list of task names")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N items of each task — a smoke test, not a run")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent request streams. 5 is this project's standing limit and a "
                         "measured fix, not a convention — see quest-cluster.md. Concurrency does "
                         "not change the requests-per-day total, only how fast they are spent")
    args = ap.parse_args()

    MODEL = args.model
    tasks = core.TASKS if args.task == "all" else [t.strip() for t in args.task.split(",")]
    unknown = [t for t in tasks if t not in core.TASKS]
    if unknown:
        raise SystemExit("unknown task(s): %s\nknown: %s" % (unknown, core.TASKS))

    print("Llama: model=%s tasks=%d" % (MODEL, len(tasks)), flush=True)
    core.run_tasks(MODEL_DIR, MODEL, call, tasks=tasks, sleep_between=args.sleep,
                   limit=args.limit, workers=args.workers, config=dict(CONFIG, model=MODEL))
    print("done ->", os.path.join(MODEL_DIR, "results"), flush=True)
