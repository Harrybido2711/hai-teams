"""Gemma — BIG-Bench Hard runner. Two providers; the provider is part of the run config.

**There is no scorer in this file.** Scoring is `bbh_eval_core.score_response`, the one lenient
matcher every model in this benchmark is judged by.

**The two providers need OPPOSITE handling, and getting it backwards is a silent regression.**
DeepInfra serves this checkpoint with thinking already OFF -- probed 2026-08-29, 98 output tokens
and no reasoning field with no reasoning parameter at all -- and it *accepts* `reasoning_effort`,
which turns thinking back ON. **Pass no reasoning parameter on DeepInfra.** On Together the same
checkpoint has thinking ON by default and it must be disabled. `.claude/references/model-calls.md`.

`--provider deepinfra` exists because this model's 112 unusable rows are being repaired there, one
whole sub-task at a time so no task is split across two providers.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import bbh_eval_core as core  # noqa: E402

DEFAULT_MODEL = "google/gemma-4-31B-it"
MODEL = DEFAULT_MODEL
PROVIDER = "together"
load_dotenv(core.ENV_PATH)

# Together's values are exactly what gemma_finish.py -- the script the job actually submitted --
# passed, kept so the tasks NOT being repaired stay comparable with the rows already on disk.
CONFIG_TOGETHER = {"temperature": 0, "max_tokens": 12000, "stream": False, "timeout": 4800}
# DeepInfra: same max_tokens so the repaired tasks differ from the Together ones by as little as
# possible, plus a seed, which this route accepts and Together's runner never set.
CONFIG_DEEPINFRA = {"temperature": 0, "max_tokens": 12000, "seed": 42, "timeout": 300}

_together = None
_deepinfra = None


def _client():
    global _together, _deepinfra
    if PROVIDER == "deepinfra":
        if _deepinfra is None:
            _deepinfra = OpenAI(api_key=os.getenv("DEEPINFRA_API_KEY"),
                                base_url="https://api.deepinfra.com/v1/openai", timeout=300)
        return _deepinfra
    if _together is None:
        # imported here, not at module scope: the DeepInfra route does not need the Together SDK
        # and should not fail to start because it is absent
        from together import Together
        _together = Together(api_key=os.getenv("TOGETHER_API_KEY"), timeout=4800)
    return _together


def call(prompt):
    def once():
        if PROVIDER == "deepinfra":
            r = _client().chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=12000,
                seed=42,
                # NO reasoning parameter: DeepInfra serves gemma with thinking already off, and
                # adding one helpfully turns it back on.
            )
        else:
            r = _client().chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=12000,
                stream=False,
            )
        return r.choices[0].message.content
    return core.retry(once, label="gemma")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run BIG-Bench Hard for Gemma.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model id; it is BOTH what is called and what the result files are named "
                         "after, so a copied folder cannot silently relabel another model's numbers")
    ap.add_argument("--provider", default="together", choices=["together", "deepinfra"],
                    help="part of the run config; the resume guard refuses to mix two providers "
                         "inside one task")
    ap.add_argument("--task", default="all", help="'all' or a comma-separated list of task names")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between calls")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N items of each task - a smoke test, not a run")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"],
                    help="prompt version. It is part of the run config and is written onto every "
                         "row, so a resume across a change of it is refused rather than silently "
                         "mixing two prompts in one result set")
    ap.add_argument("--shard", type=int, default=0, help="this shard's index, 0-based")
    ap.add_argument("--total-shards", dest="total_shards", type=int, default=1,
                    help="how many shards the work is split across. At 1 the output filename has "
                         "no shard tag, so an unsharded run keeps the name it always had")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent request streams. 5 is this project's standing limit and a "
                         "measured fix, not a convention - see quest-cluster.md")
    args = ap.parse_args()

    MODEL = args.model
    PROVIDER = args.provider
    tasks = core.TASKS if args.task == "all" else [t.strip() for t in args.task.split(",")]
    unknown = [t for t in tasks if t not in core.TASKS]
    if unknown:
        raise SystemExit("unknown task(s): %s\nknown: %s" % (unknown, core.TASKS))

    cfg = dict(CONFIG_DEEPINFRA if PROVIDER == "deepinfra" else CONFIG_TOGETHER,
               model=MODEL, provider=PROVIDER)
    print("Gemma: model=%s provider=%s tasks=%d" % (MODEL, PROVIDER, len(tasks)), flush=True)
    core.run_tasks(MODEL_DIR, MODEL, call, tasks=tasks, sleep_between=args.sleep,
                   limit=args.limit, workers=args.workers, prompt_version=args.prompt,
                   shard=args.shard, total_shards=args.total_shards, config=cfg)
    print("done ->", os.path.join(MODEL_DIR, "results"), flush=True)
