"""Gemini Flash 2.5 (SUPERSEDED) — BIG-Bench Hard runner.

**There is no scorer in this file.** Scoring is `bbh_eval_core.score_response`, the one lenient
matcher every model in this benchmark is judged by, so no runner can score its model more or less
generously than another. Everything model-specific lives below: a client and a `call`.

**This model is superseded** — the project's Gemini is `gemini-3.5-flash-lite` on OpenRouter. The folder is named after what it actually calls so its numbers are not mistaken for the current model's.\n\n**Its results on disk are a broken run:** 3,002 of 4,833 responses (62%) stop mid-reasoning and never reach `Final Answer:`. No output cap is set here, so raising one is not the fix — diagnose before re-running.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from google import genai

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import bbh_eval_core as core  # noqa: E402

DEFAULT_MODEL = "gemini-2.5-flash"
MODEL = DEFAULT_MODEL
load_dotenv(core.ENV_PATH)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# What this runner actually passes, written onto every row it produces
# (`.claude/references/model-parameters.md` rule 8). No seed anywhere here — that is
# the gap, not an omission in this line.
CONFIG = {}


def call(prompt):
    def once():
        return client.models.generate_content(model=MODEL, contents=prompt).text
    return core.retry(once, label="gemini")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run BIG-Bench Hard for Gemini Flash 2.5 (SUPERSEDED).")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model id; it is BOTH what is called and what the result files are named "
                         "after, so a copied folder cannot silently relabel another model's numbers")
    ap.add_argument("--task", default="all", help="'all' or a comma-separated list of task names")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N items of each task — a smoke test, not a run")
    args = ap.parse_args()

    MODEL = args.model
    tasks = core.TASKS if args.task == "all" else [t.strip() for t in args.task.split(",")]
    unknown = [t for t in tasks if t not in core.TASKS]
    if unknown:
        raise SystemExit("unknown task(s): %s\nknown: %s" % (unknown, core.TASKS))

    print("Gemini Flash 2.5 (SUPERSEDED): model=%s tasks=%d" % (MODEL, len(tasks)), flush=True)
    core.run_tasks(MODEL_DIR, MODEL, call, tasks=tasks, sleep_between=args.sleep,
                   limit=args.limit, config=dict(CONFIG, model=MODEL))
    print("done ->", os.path.join(MODEL_DIR, "results"), flush=True)
