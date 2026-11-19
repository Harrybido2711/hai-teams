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

**`--sample sample10` is the 10% sample rerun (2026-11-15), and nothing else changes without it.**
It runs only the items in `<bbh>/sample10.json` into `results_sample10/` (`smoke_sample10/` under
`--limit`), never `results/`, through `bbh_eval_core.run_sample`: a second client built
`max_retries=0` so `n_attempts` and `latency_s` are true, five attempts in the core's loop, and the
per-row usage fields on every row. Its config is the settled one, PlanBench's
(`PLAN_GPT_5.6_Luna`): the same `WANTED` as below, with `reasoning_effort` and the cap `required` so
negotiation cannot swap them. The cap `max_completion_tokens=16384` is chosen, not measured — bbh's
full run at it had `no_marker=0`, which says it did not bind there.
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


def run_sample(args, tasks):
    """`--sample`: the settled config through the core's guarded loop. Returns the exit code."""
    # max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie
    # (script-skeleton.md §4). Timeouts are PLAN_GPT_5.6_Luna's: the watchdog (600 s) fires first.
    sample_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=900, max_retries=0)

    def sample_once(messages, model, params):
        r = sample_client.chat.completions.create(model=model, messages=messages, **params)
        meta = core.openai_meta(r)        # first: it books the bill before choices[0] can raise
        return r.choices[0].message.content, meta

    return core.run_sample(MODEL_DIR, args.model, client=sample_client, once=sample_once,
                           wanted=WANTED, sample=args.sample, tasks=tasks,
                           required={"reasoning_effort": "low", "max_completion_tokens": 16384},
                           extra_config={"route": "openai"}, call_timeout=600, limit=args.limit,
                           workers=args.workers, prompt_version=args.prompt, shard=args.shard,
                           total_shards=args.total_shards, sleep_between=args.sleep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run BIG-Bench Hard for GPT-5.6-Luna.")
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
    ap.add_argument("--shard", type=int, default=0, help="this shard's index, 0-based")
    ap.add_argument("--total-shards", dest="total_shards", type=int, default=1,
                    help="how many shards the work is split across. At 1 the output filename has "
                         "no shard tag, so an unsharded run keeps the name it always had")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent request streams. 5 is this project's standing limit and a "
                         "measured fix, not a convention — see quest-cluster.md. Concurrency does "
                         "not change the requests-per-day total, only how fast they are spent")
    ap.add_argument("--sample", default="",
                    help="e.g. sample10: only the items of <bbh>/<name>.json, with the per-row usage "
                         "fields, into results_<name>/ — never results/. Omitted, the run is exactly "
                         "what it was before this flag existed")
    args = ap.parse_args()

    MODEL = args.model
    tasks = core.TASKS if args.task == "all" else [t.strip() for t in args.task.split(",")]
    unknown = [t for t in tasks if t not in core.TASKS]
    if unknown:
        raise SystemExit("unknown task(s): %s\nknown: %s" % (unknown, core.TASKS))
    if args.sample:
        sys.exit(run_sample(args, tasks))

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
                   limit=args.limit, workers=args.workers, prompt_version=args.prompt,
                   shard=args.shard, total_shards=args.total_shards, config=dict(PARAMS, model=MODEL))
    print("done ->", os.path.join(MODEL_DIR, "results"), flush=True)
