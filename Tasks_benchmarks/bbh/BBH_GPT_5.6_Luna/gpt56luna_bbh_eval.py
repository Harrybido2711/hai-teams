"""GPT-5.6-Luna — BIG-Bench Hard runner. **The project's OpenAI slot.**

**There is no scorer in this file.** Scoring is `bbh_eval_core.score_response`, the one lenient
matcher every model in this benchmark is judged by.

This slot has never been run on bbh. `Final_Result.xlsx`'s OpenAI column is still
`gpt-4o-mini-2024-07-18` (`BBH_GPT_4o_mini/`), which is superseded — re-pointing that column means
running this, not editing a header.

**The parameter surface is negotiated at startup, not hardcoded.** This model refuses `max_tokens`
(it must be `max_completion_tokens`) and refuses the *value* `reasoning_effort="minimal"` while
supporting the parameter perfectly. A rejection is permanent, so discovering it per item costs the
run; and dropping `reasoning_effort` because one value was refused would run all 4,833 items at the
model's default `medium`, uncapped. `core.negotiate` handles both cases and prints what it settled
on. Never set `reasoning_effort="none"` — that is removal, not a cap, and costs 9-11 points
(`.claude/references/model-parameters.md`).
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import bbh_eval_core as core  # noqa: E402

DEFAULT_MODEL = "gpt-5.6-luna"
MODEL = DEFAULT_MODEL
PARAMS = {}

# What we ask for. `max_completion_tokens` counts reasoning tokens as well as visible ones, so it is
# twice the visible budget the other bbh runners use (8192) to leave room for low-effort reasoning
# on top. NOT MEASURED YET — run --limit 20 first and check `no_marker` in the overall CSV before
# committing 4,833 items to it: a cap that truncates returns a billed response with no answer.
WANTED = {"max_completion_tokens": 16384, "reasoning_effort": "low", "seed": 42}

load_dotenv(core.ENV_PATH)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=300)


def call(prompt):
    def once():
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            **PARAMS,
        )
        return r.choices[0].message.content
    return core.retry(once, label="gpt56luna")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run BIG-Bench Hard for GPT-5.6-Luna.")
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

    print("negotiating parameter surface for %s ..." % MODEL, flush=True)
    PARAMS, notes = core.negotiate(client, MODEL, WANTED)
    print("  accepted: %s%s" % (PARAMS, ("  [%s]" % "; ".join(notes)) if notes else ""), flush=True)
    if "reasoning_effort" not in PARAMS:
        raise SystemExit("reasoning_effort was dropped entirely — that runs uncapped at the model "
                         "default. Fix the request rather than proceeding.")
    # write what was actually used beside the results, so a number can be traced to its config
    os.makedirs(os.path.join(MODEL_DIR, "results"), exist_ok=True)
    with open(os.path.join(MODEL_DIR, "results", "negotiated_params.json"), "w") as fh:
        json.dump({"model": MODEL, "asked": WANTED, "accepted": PARAMS, "notes": notes}, fh, indent=2)

    print("GPT-5.6-Luna: model=%s tasks=%d" % (MODEL, len(tasks)), flush=True)
    # rule 8: the config goes on every row, not only into the job script
    core.run_tasks(MODEL_DIR, MODEL, call, tasks=tasks, sleep_between=args.sleep,
                   limit=args.limit, config=dict(PARAMS, model=MODEL))
    print("done ->", os.path.join(MODEL_DIR, "results"), flush=True)
