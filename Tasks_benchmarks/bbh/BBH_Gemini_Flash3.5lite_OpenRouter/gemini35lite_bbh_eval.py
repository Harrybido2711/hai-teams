"""Gemini 3.5 Flash-Lite via OpenRouter — BIG-Bench Hard runner. **The project's Gemini slot.**

**There is no scorer in this file.** Scoring is `bbh_eval_core.score_response`, the one lenient
matcher every model in this benchmark is judged by.

The folder carries the route because there are two and only one is settled: OpenRouter, set by the
user 2026-08-23 (`.claude/references/model-calls.md`). The native Google AI Studio route runs at the
same speed but takes a different config shape, and Quest's SDK has no `thinking_level` field.

This slot has never been run on bbh. The existing `BBH_Gemini_Flash2.5/` holds `gemini-2.5-flash`,
which is superseded **and whose rows are a broken run** — 62% of them stop mid-reasoning and never
emit `Final Answer:`. That is the column this runner is meant to replace, and the reason
`has_marker` is worth watching from the first pilot.

**`minimal` is the thinking floor; there is no off.** `thinking_budget=0` is rejected 400
INVALID_ARGUMENT on this model, and on a structured task thinking already spends close to zero.
**Omit `temperature`, `top_p`, `top_k`** — 3.x guidance is to remove, not tune.

**OpenRouter switches backend mid-run.** It serves this model from Google AI Studio *and* Google
Vertex with failover; over one 400-item run the split was 110/90. The response carries `provider`,
so this runner counts them and writes the tally beside the results — a score that moved may be a
backend change rather than the model.

**`--sample sample10` is the 10% sample rerun (2026-11-15), and nothing else changes without it.**
It runs only the items in `<bbh>/sample10.json` into `results_sample10/` (`smoke_sample10/` under
`--limit`), never `results/`, through `bbh_eval_core.run_sample`: a second client built
`max_retries=0` so `n_attempts` and `latency_s` are true, five attempts in the core's loop, and the
per-row usage fields on every row — the answering backend among them, read from each response so
no thread-local is needed, and OpenRouter's own dollar figure as `last_call_cost`, summed over every
call of the row as `billed_cost`. Its config is the settled one,
PlanBench's (`PLAN_Gemini_Flash3.5lite_OpenRouter`): `EXTRA_BODY` sent on every call as `fixed`,
`max_tokens` `required`. The cap `max_tokens=8192` is chosen, not measured — bbh's full run at it
had `no_marker=0`, which says it did not bind there.
"""

import argparse
import collections
import json
import threading
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import bbh_eval_core as core  # noqa: E402

DEFAULT_MODEL = "google/gemini-3.5-flash-lite"
MODEL = DEFAULT_MODEL
PARAMS = {}
BACKENDS = collections.Counter()
_BACKEND_LOCK = threading.Lock()
# Thread-local, not a module global: with 5 concurrent request streams a shared "last backend"
# would put whichever thread answered most recently onto the row being written, silently
# misattributing exactly the thing this field exists to track.
_LOCAL = threading.local()

# `reasoning.effort` goes through extra_body, not as a named parameter, so it is not part of the
# negotiation — OpenRouter passes it to the backend rather than validating it here.
EXTRA_BODY = {"reasoning": {"effort": "minimal"}}
# 8192 visible tokens, matching the other bbh runners so the models stay comparable. Thinking is
# billed at the output rate but spends ~0 at `minimal`. NOT MEASURED YET on BBH prompts — run
# --limit 20 and check `no_marker` before committing 4,833 items.
WANTED = {"max_tokens": 8192, "seed": 42}

load_dotenv(core.ENV_PATH)
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"), timeout=300)


def call(prompt):
    def once():
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            extra_body=EXTRA_BODY,
            **PARAMS,
        )
        provider = getattr(r, "provider", None) or "unreported"
        with _BACKEND_LOCK:          # Counter += is not atomic
            BACKENDS[provider] += 1
        _LOCAL.provider = provider
        return r.choices[0].message.content
    return core.retry(once, label="gemini35lite")


def run_sample(args, tasks):
    """`--sample`: the settled config through the core's guarded loop. Returns the exit code."""
    # max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie
    # (script-skeleton.md §4). Timeouts are PLAN_Gemini's: the watchdog (300 s) fires first.
    sample_client = OpenAI(base_url="https://openrouter.ai/api/v1",
                           api_key=os.getenv("OPENROUTER_API_KEY"), timeout=400, max_retries=0)
    fixed = {"extra_body": EXTRA_BODY}

    def sample_once(messages, model, params):
        r = sample_client.chat.completions.create(model=model, messages=messages, **fixed, **params)
        meta = core.openai_meta(r)        # first: it books the bill before choices[0] can raise
        return r.choices[0].message.content, meta

    return core.run_sample(MODEL_DIR, args.model, client=sample_client, once=sample_once,
                           wanted=WANTED, sample=args.sample, tasks=tasks, fixed=fixed,
                           required={"max_tokens": 8192}, extra_config={"route": "openrouter"},
                           call_timeout=300, limit=args.limit, workers=args.workers,
                           prompt_version=args.prompt, shard=args.shard,
                           total_shards=args.total_shards, sleep_between=args.sleep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run BIG-Bench Hard for Gemini 3.5 Flash-Lite (OpenRouter).")
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
    if "seed" not in PARAMS:
        print("  WARNING: seed was dropped — this run is NOT reproducible "
              "(model-parameters.md rule 8)", flush=True)
    os.makedirs(os.path.join(MODEL_DIR, "results"), exist_ok=True)
    with open(os.path.join(MODEL_DIR, "results", "negotiated_params.json"), "w") as fh:
        json.dump({"model": MODEL, "asked": WANTED, "accepted": PARAMS,
                   "extra_body": EXTRA_BODY, "notes": notes}, fh, indent=2)

    print("Gemini 3.5 Flash-Lite (OpenRouter): model=%s tasks=%d" % (MODEL, len(tasks)), flush=True)
    try:
        # rule 8: config on every row. The backend goes on too, per row rather than per run,
        # because a seed alone does not reproduce here — OpenRouter picks the backend and
        # switches mid-run (`.claude/references/provider-gotchas.md`).
        core.run_tasks(MODEL_DIR, MODEL, call, tasks=tasks, sleep_between=args.sleep,
                       limit=args.limit, workers=args.workers, prompt_version=args.prompt,
                       shard=args.shard, total_shards=args.total_shards,
                       config=dict(PARAMS, model=MODEL,
                                   reasoning_effort=EXTRA_BODY["reasoning"]["effort"]),
                       per_row_config=lambda: {"backend": getattr(_LOCAL, "provider", None)})
    finally:
        # written even on a wall-clock kill: which backend answered is not recoverable afterwards
        with open(os.path.join(MODEL_DIR, "results", "backend_providers.json"), "w") as fh:
            json.dump(dict(BACKENDS), fh, indent=2)
        print("backends:", dict(BACKENDS), flush=True)
    print("done ->", os.path.join(MODEL_DIR, "results"), flush=True)
