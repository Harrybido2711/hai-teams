"""GPT-5.6-Luna — MMLU runner. **The project's OpenAI slot.**

**There is no scorer in this file.** Scoring is `mmlu_eval_core.score_response`, the one lenient
matcher every model in this benchmark is judged by.

This slot has never been run on MMLU: `Final_Result.xlsx`'s OpenAI column is blank and its `Model`
row already names `gpt-5.6-luna`. The `MMLU_GPT_4o_mini/` folder holds the superseded model that
produced the old numbers.

**The parameter surface is negotiated at startup, not hardcoded.** This model refuses `max_tokens`
(it must be `max_completion_tokens`) and refuses the *value* `reasoning_effort="minimal"` while
supporting the parameter. A rejection is permanent, so discovering it per item costs the run; and
dropping `reasoning_effort` because one value was refused would run all 3,943 items at the model's
default `medium`, uncapped. Never set it to `none` — that is removal, not a cap, and costs 9-11
points (`.claude/references/model-parameters.md`).

**Prompt v2 by default**: choices labelled `A.`-`D.`, answer as a letter. MMLU shipped with two
prompts and four of its seven runners used this one; a single letter is also unambiguous to score.

**`--sample sample10` runs the 10% sample rerun instead** (`.claude/references/sample10-plan.md`,
WP3): the 401 items of `../sample10.json`, written to `results_sample10/` (a `--limit` smoke to
`smoke_sample10/`), never `results/`, with the per-row usage fields on every row. It branches off
before anything below the argument parser runs, so without `--sample` this file behaves exactly as
it did. Its config is the settled one (`model-parameters.md` § Settled — the project's GPT), the same
three values `WANTED` asks for, all three pinned by `required`: `reasoning_effort="low"`, `seed=42`,
`max_completion_tokens=16384` — **the cap is chosen, not measured** (bbh's and PlanBench's declared
cap for this model); `finish_reason` on the rows says whether it binds. It uses its own client with
`max_retries=0`, so `n_attempts` and `latency_s` are true; the legacy client keeps its settings.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import mmlu_eval_core as core  # noqa: E402

DEFAULT_MODEL = "gpt-5.6-luna"
MODEL = DEFAULT_MODEL
PARAMS = {}

# `max_completion_tokens` counts reasoning tokens as well as visible ones. MMLU answers are one
# letter, so the visible need is tiny and the budget is almost all reasoning headroom.
# NOT MEASURED on MMLU prompts yet - run --limit 20 and check `no_marker` before the full run.
WANTED = {"max_completion_tokens": 16384, "reasoning_effort": "low", "seed": 42}

load_dotenv(core.ENV_PATH)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=300)


def call(prompt):
    def once():
        r = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], **PARAMS)
        return r.choices[0].message.content
    return core.retry(once, label="gpt56luna")


def run_sample10(args):
    """The --sample path. Its own client: max_retries=0, because the SDK's hidden retries would make
    n_attempts and latency_s lie; the core makes up to 5 attempts itself (script-skeleton.md §4)."""
    sample_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=900, max_retries=0)

    def once(messages, model, params):
        r = core.create(sample_client, model=model, messages=messages, **params)
        meta = core.openai_meta(r)       # first: the usage is billed even if the body is malformed
        return r.choices[0].message.content, meta

    return core.run_sample(args, model_dir=MODEL_DIR, client=sample_client, once=once,
                           wanted=WANTED, required=dict(WANTED), call_timeout=600,
                           extra_config={"route": "openai"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run MMLU for GPT-5.6-Luna.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model id; it is BOTH what is called and what the result files are named "
                         "after, so a copied folder cannot relabel another model's numbers")
    ap.add_argument("--subject", default="all",
                    help="'all' or a comma-separated list of subject names")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N items of each subject - a smoke test, not a run")
    ap.add_argument("--prompt", default="v2", choices=["v1", "v2"],
                    help="prompt version; part of the run config, so a resume across a change of "
                         "it is refused rather than silently mixing two prompts")
    ap.add_argument("--shard", type=int, default=0, help="this shard's index, 0-based")
    ap.add_argument("--total-shards", dest="total_shards", type=int, default=1,
                    help="how many shards the work is split across. At 1 the output filename has "
                         "no shard tag, so an unsharded run keeps the name it always had")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent request streams. 5 is this project's standing limit and a "
                         "measured fix, not a convention - see quest-cluster.md")
    core.add_sample_args(ap)
    args = ap.parse_args()
    if args.sample:
        raise SystemExit(run_sample10(args))

    MODEL = args.model
    subjects = core.SUBJECTS if args.subject == "all" else [s.strip() for s in args.subject.split(",")]
    unknown = [s for s in subjects if s not in core.SUBJECTS]
    if unknown:
        raise SystemExit("unknown subject(s): %s\nknown: %s" % (unknown, core.SUBJECTS))

    print("negotiating parameter surface for %s ..." % MODEL, flush=True)
    PARAMS, notes = core.negotiate(client, MODEL, WANTED)
    print("  accepted: %s%s" % (PARAMS, ("  [%s]" % "; ".join(notes)) if notes else ""), flush=True)
    if "reasoning_effort" not in PARAMS:
        raise SystemExit("reasoning_effort was dropped entirely - that runs uncapped at the model "
                         "default. Fix the request rather than proceeding.")
    os.makedirs(os.path.join(MODEL_DIR, "results"), exist_ok=True)
    with open(os.path.join(MODEL_DIR, "results", "negotiated_params.json"), "w") as fh:
        json.dump({"model": MODEL, "asked": WANTED, "accepted": PARAMS, "notes": notes}, fh, indent=2)

    print("GPT-5.6-Luna: model=%s subjects=%d prompt=%s" % (MODEL, len(subjects), args.prompt),
          flush=True)
    core.run_subjects(MODEL_DIR, MODEL, call, subjects=subjects, sleep_between=args.sleep,
                      limit=args.limit, workers=args.workers, prompt_version=args.prompt,
                      shard=args.shard, total_shards=args.total_shards,
                      config=dict(PARAMS, model=MODEL))
    print("done ->", os.path.join(MODEL_DIR, "results"), flush=True)
