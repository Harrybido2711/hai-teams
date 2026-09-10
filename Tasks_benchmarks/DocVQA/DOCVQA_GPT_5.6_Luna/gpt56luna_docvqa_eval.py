"""DocVQA runner — `gpt-5.6-luna` on the OpenAI platform. **The project's OpenAI slot.**

**There is no scorer in this file.** Scoring is `docvqa_eval_core`, the one matcher every model in
this benchmark is judged by, and the prompt comes from there too — both byte-identical to what
produced the four results already on disk, so this run is comparable with them.

`openai_eval.py` at the DocVQA root is the **superseded** `gpt-4o-mini` run (0.8583 ANLS). It stays
as the record of that model; `Final_Result.xlsx`'s OpenAI column is blank on this sheet precisely
because the selected model had not run, and this closes it.

**The parameter surface is negotiated at startup, not hardcoded.** This model refuses `max_tokens`
(it must be `max_completion_tokens`) and refuses the *value* `reasoning_effort="minimal"` while
accepting the parameter. `core.negotiate` corrects a refused value rather than dropping the cap —
dropping `reasoning_effort` because one value was refused would run all 5,349 images uncapped.
Same `WANTED` as the MMLU and EmoBench runners, so the model's config is the same across benchmarks.

**A daily-cap refusal is fatal here, not retried.** This is the incident this benchmark is famous
for: at 10 shards with 3 retries an item, the retries spent the day's request quota and left 3,021
of 5,349 rows empty (`OPENAI_EVAL_NOTES.md`). Five shards, a 2.5 s sleep, and `fatal` markers on the
cap refusals are the fix — do not raise the shard count.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import docvqa_eval_core as core  # noqa: E402

DEFAULT_MODEL = "gpt-5.6-luna"
MODEL = DEFAULT_MODEL
PARAMS = {}

WANTED = {"max_completion_tokens": 16384, "reasoning_effort": "low", "seed": 42}

# Neither of these clears by waiting a few seconds, and retrying them is what produced the 3,021
# empty rows. Stop instead, so the rows already written stay resumable.
FATAL = (("requests per day", "daily request cap reached — stopping so the remaining quota is "
                              "left for the resume, which will re-ask only the unanswered rows"),
         ("insufficient_quota", "billing quota exhausted — top up at platform.openai.com, then "
                                "resubmit; the checkpoint is intact"))

load_dotenv(core.ENV_PATH)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=300)


def call(prompt, image_path):
    b64 = core.load_image_b64(image_path)   # FileNotFoundError is handled by core.run

    def once():
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 # detail="high" as the gpt-4o-mini run used: a document page is unreadable at low.
                 "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                {"type": "text", "text": prompt}]}],
            **PARAMS)
        return r.choices[0].message.content

    return core.retry(once, label="luna-docvqa", fatal=FATAL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run DocVQA for gpt-5.6-luna (OpenAI platform).")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model id; it is BOTH what is called and what the result files are named "
                         "after, so a copied folder cannot relabel another model's numbers")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N questions — a smoke test, not a run. It does NOT resume "
                         "and writes the untagged filename, so never point it at real rows")
    ap.add_argument("--sleep", type=float, default=2.5,
                    help="seconds between calls. 2.5 is the measured fix from the quota incident: "
                         "5 shards x ~1050 tokens / 2.5 s stays under the token-per-minute limit")
    ap.add_argument("--save-every", dest="save_every", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0, help="this shard's index, 0-based")
    ap.add_argument("--total-shards", dest="total_shards", type=int, default=1,
                    help="how many shards the work is split across. 5 is this benchmark's measured "
                         "ceiling and a fix, not a convention — see OPENAI_EVAL_NOTES.md")
    args = ap.parse_args()

    MODEL = args.model
    print("negotiating parameter surface for %s ..." % MODEL, flush=True)
    PARAMS, notes = core.negotiate(client, MODEL, WANTED)
    print("  accepted: %s%s" % (PARAMS, ("  [%s]" % "; ".join(notes)) if notes else ""), flush=True)
    if "reasoning_effort" not in PARAMS:
        raise SystemExit("reasoning_effort was dropped entirely - that runs uncapped at the model's "
                         "default effort over 5,349 images. Fix the negotiation, do not proceed.")
    if "seed" not in PARAMS:
        print("  WARNING: seed was dropped - this run is NOT reproducible", flush=True)
    with open(os.path.join(core.results_dir(MODEL_DIR), "negotiated_params.json"), "w") as fh:
        json.dump({"model": MODEL, "asked": WANTED, "accepted": PARAMS, "notes": notes}, fh, indent=2)

    core.run(MODEL_DIR, MODEL, call, limit=args.limit, sleep_between=args.sleep,
             save_every=args.save_every, shard=args.shard, total_shards=args.total_shards,
             config=dict(PARAMS, model=MODEL))
