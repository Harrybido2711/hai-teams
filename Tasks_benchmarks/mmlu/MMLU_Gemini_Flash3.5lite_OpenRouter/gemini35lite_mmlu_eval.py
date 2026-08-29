"""Gemini 3.5 Flash-Lite via OpenRouter — MMLU runner. **The project's Gemini slot.**

**There is no scorer in this file.** Scoring is `mmlu_eval_core.score_response`, the one lenient
matcher every model in this benchmark is judged by.

The folder carries the route because there are two and only OpenRouter is settled (user,
2026-08-23). This slot has never been run on MMLU: `Final_Result.xlsx`'s Gemini column is blank and
its `Model` row already names `gemini-3.5-flash-lite`. `MMLU_Gemini_Flash2.5/` holds the superseded
model that produced the old numbers.

**`minimal` is the thinking floor; there is no off.** `thinking_budget=0` is rejected 400
INVALID_ARGUMENT on this model. **Omit `temperature`, `top_p`, `top_k`** — 3.x guidance is to remove,
not tune.

**OpenRouter switches backend mid-run**, serving this model from Google AI Studio *and* Google
Vertex with failover — measured 2,008/2,038 across one bbh run. A seed alone therefore does not
reproduce, so the answering backend is recorded per row, not per run.

**Prompt v2 by default**: choices labelled `A.`-`D.`, answer as a letter. MMLU shipped with two
prompts and four of its seven runners used this one; a single letter is also unambiguous to score.
"""

import argparse
import collections
import json
import os
import sys
import threading

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import mmlu_eval_core as core  # noqa: E402

DEFAULT_MODEL = "google/gemini-3.5-flash-lite"
MODEL = DEFAULT_MODEL
PARAMS = {}

# reasoning.effort goes through extra_body, not as a named parameter, so it is not negotiated.
EXTRA_BODY = {"reasoning": {"effort": "minimal"}}
# NOT MEASURED on MMLU prompts yet - run --limit 20 and check `no_marker` before the full run.
WANTED = {"max_tokens": 8192, "seed": 42}

load_dotenv(core.ENV_PATH)
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"), timeout=300)
BACKENDS = collections.Counter()
_BACKEND_LOCK = threading.Lock()
# Thread-local, not a shared global: with concurrent streams a shared "last backend" would put
# whichever thread answered most recently onto the row being written.
_LOCAL = threading.local()


def call(prompt):
    def once():
        r = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            extra_body=EXTRA_BODY, **PARAMS)
        provider = getattr(r, "provider", None) or "unreported"
        with _BACKEND_LOCK:          # Counter += is not atomic
            BACKENDS[provider] += 1
        _LOCAL.provider = provider
        return r.choices[0].message.content
    return core.retry(once, label="gemini35lite")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run MMLU for Gemini 3.5 Flash-Lite (OpenRouter).")
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
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent request streams. 5 is this project's standing limit and a "
                         "measured fix, not a convention - see quest-cluster.md")
    args = ap.parse_args()

    MODEL = args.model
    subjects = core.SUBJECTS if args.subject == "all" else [s.strip() for s in args.subject.split(",")]
    unknown = [s for s in subjects if s not in core.SUBJECTS]
    if unknown:
        raise SystemExit("unknown subject(s): %s\nknown: %s" % (unknown, core.SUBJECTS))

    print("negotiating parameter surface for %s ..." % MODEL, flush=True)
    PARAMS, notes = core.negotiate(client, MODEL, WANTED)
    print("  accepted: %s%s" % (PARAMS, ("  [%s]" % "; ".join(notes)) if notes else ""), flush=True)
    if "seed" not in PARAMS:
        print("  WARNING: seed was dropped - this run is NOT reproducible", flush=True)
    os.makedirs(os.path.join(MODEL_DIR, "results"), exist_ok=True)
    with open(os.path.join(MODEL_DIR, "results", "negotiated_params.json"), "w") as fh:
        json.dump({"model": MODEL, "asked": WANTED, "accepted": PARAMS,
                   "extra_body": EXTRA_BODY, "notes": notes}, fh, indent=2)

    print("Gemini 3.5 Flash-Lite: model=%s subjects=%d prompt=%s" % (MODEL, len(subjects), args.prompt),
          flush=True)
    try:
        core.run_subjects(MODEL_DIR, MODEL, call, subjects=subjects, sleep_between=args.sleep,
                          limit=args.limit, workers=args.workers, prompt_version=args.prompt,
                          config=dict(PARAMS, model=MODEL,
                                      reasoning_effort=EXTRA_BODY["reasoning"]["effort"]),
                          per_row_config=lambda: {"backend": getattr(_LOCAL, "provider", None)})
    finally:
        with open(os.path.join(MODEL_DIR, "results", "backend_providers.json"), "w") as fh:
            json.dump(dict(BACKENDS), fh, indent=2)
        print("backends:", dict(BACKENDS), flush=True)
    print("done ->", os.path.join(MODEL_DIR, "results"), flush=True)
