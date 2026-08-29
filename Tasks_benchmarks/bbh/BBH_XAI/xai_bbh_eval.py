"""XAI — BIG-Bench Hard runner.

**There is no scorer in this file.** Scoring is `bbh_eval_core.score_response`, the one lenient
matcher every model in this benchmark is judged by, so no runner can score its model more or less
generously than another. Everything model-specific lives below: a client and a `call`.

The xAI SDK takes no message dicts: `chat.create(...)`, `chat.append(user(...))`, `chat.sample()`. Reasoning is the model and cannot be turned off.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import bbh_eval_core as core  # noqa: E402

DEFAULT_MODEL = "grok-3-mini"
MODEL = DEFAULT_MODEL
load_dotenv(core.ENV_PATH)
client = Client(api_key=os.getenv("GROK_API_KEY"), timeout=3600)


# What this runner actually passes, written onto every row it produces
# (`.claude/references/model-parameters.md` rule 8). No seed anywhere here — that is
# the gap, not an omission in this line.
CONFIG = {"timeout": 3600}


def call(prompt):
    def once():
        chat = client.chat.create(model=MODEL)
        chat.append(user(prompt))
        return chat.sample().content
    return core.retry(once, label="xai")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run BIG-Bench Hard for XAI.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model id; it is BOTH what is called and what the result files are named "
                         "after, so a copied folder cannot silently relabel another model's numbers")
    ap.add_argument("--task", default="all", help="'all' or a comma-separated list of task names")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N items of each task — a smoke test, not a run")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"],
                    help="prompt version. It is part of the run config and is written onto every "
                         "row, so a resume across a change of it is refused rather than silently "
                         "mixing two prompts in one result set")
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

    print("XAI: model=%s tasks=%d" % (MODEL, len(tasks)), flush=True)
    core.run_tasks(MODEL_DIR, MODEL, call, tasks=tasks, sleep_between=args.sleep,
                   limit=args.limit, workers=args.workers, prompt_version=args.prompt, config=dict(CONFIG, model=MODEL))
    print("done ->", os.path.join(MODEL_DIR, "results"), flush=True)
