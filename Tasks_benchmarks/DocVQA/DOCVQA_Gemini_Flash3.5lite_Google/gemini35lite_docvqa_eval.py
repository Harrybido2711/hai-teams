"""DocVQA runner — `gemini-3.5-flash-lite` via **Google AI Studio** (native google-genai).

**The project's Gemini slot, on the route the user chose on 2026-09-10.** The same route MMLU's
Gemini column was finished on, so the model's config matches across the two benchmarks:
`thinking_budget=128`, `max_output_tokens=8192`, `seed=42`, no temperature.

**There is no scorer in this file.** Scoring is `docvqa_eval_core`, the one matcher every model here
is judged by, and the prompt comes from there — both byte-identical to what produced the four
results already on disk, so this run is comparable with them.

`gemini_eval.py` at the DocVQA root is the **superseded** `gemini-2.5-flash` run (0.9357 ANLS). It
stays as the record of that model; `Final_Result.xlsx`'s Gemini column is blank on this sheet
because the selected model had not run, and this closes it.

**Use `thinking_budget`, not `thinking_level`.** Quest's google-genai is 1.49.0 and its
`ThinkingConfig` exposes only `include_thoughts` and `thinking_budget`; sending `thinking_level`
there raises a pydantic ValidationError on *every* call, which a `except TypeError` guard never
catches. The field is selected from `model_fields`, never by trying one and catching
(`provider-gotchas.md`). `thinking_budget=0` is rejected 400 on this model — minimal is the floor.

**On MMLU this budget was a request, not a ceiling** — 253 thinking tokens a call against a budget
of 128, measured over 1,755 calls. Expect the same here and do not read the number as a cap.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import docvqa_eval_core as core  # noqa: E402

# OpenRouter's spelling of the id is used for the result filenames on MMLU; here there is no
# carried-over checkpoint to match, so the native id is used as-is and named on every row.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MODEL = DEFAULT_MODEL

THINKING_BUDGET = 128        # 0 is rejected 400 on flash-lite
THINKING_LEVEL = "minimal"   # used only where the SDK exposes thinking_level instead
MAX_OUTPUT_TOKENS = 8192

# Permanent failures. Retrying an auth or request-shape error 3x across 5,349 images spends hours
# proving one fact (`provider-gotchas.md`).
FATAL = (("api key not valid", "auth failure — fix the key and resubmit"),
         ("api_key_invalid", "auth failure — fix the key and resubmit"),
         ("unauthenticated", "auth failure — fix the key and resubmit"),
         ("permission_denied", "auth failure — fix the key and resubmit"),
         ("access_token_type_unsupported", "auth failure — fix the key and resubmit"),
         ("invalid_argument", "permanent request-shape failure, not retried"),
         ("extra inputs are not permitted", "permanent request-shape failure, not retried"),
         ("resource_exhausted", "quota exhausted — stopping so the checkpoint stays resumable"))

load_dotenv(core.ENV_PATH)
# Its own key, not GEMINI_API_KEY: that one is the 2.5 run's and is a different quota.
api_key = os.getenv("GEMINI_FLASH_LITE_API_KEY")
if not api_key:
    sys.exit("GEMINI_FLASH_LITE_API_KEY is not set in %s" % core.ENV_PATH)
# The SDK also reads GOOGLE_API_KEY and GEMINI_API_KEY from the environment and announces which it
# picked — "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY." Locally all three
# hold the same value so it does not matter; **on Quest this benchmark's .env carries the 2.5 run's
# AIzaSy key under GEMINI_API_KEY**, a different key on a different quota. Removing them leaves the
# explicit api_key above as the only one there is, rather than trusting a precedence rule.
for ambient in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
    os.environ.pop(ambient, None)
# A request timeout, because it is the only guard this SDK offers and a job hung inside one call
# reports RUNNING to SLURM with an empty log for as long as it takes (`quest-cluster.md`).
client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=300_000))


def _thinking_config():
    """Chosen from what THIS SDK exposes, never by trying a field and catching an exception."""
    fields = set(getattr(types.ThinkingConfig, "model_fields", None) or {})
    if "thinking_budget" in fields:
        return types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
    if "thinking_level" in fields:
        return types.ThinkingConfig(thinking_level=THINKING_LEVEL)
    sys.exit("This google-genai build exposes no thinking cap (ThinkingConfig fields: %s). Upgrade "
             "the SDK, or decide explicitly to run without one and record that — do not delete "
             "this guard." % sorted(fields))


def visible_text(resp):
    """The answer parts only. `resp.text` warns and concatenates whenever the response carries
    non-text parts, and this model returns a `thought_signature` part on every call."""
    try:
        return "".join(p.text for p in resp.candidates[0].content.parts
                       if getattr(p, "text", None) and not getattr(p, "thought", False))
    except Exception:
        return getattr(resp, "text", "") or ""


def call(prompt, image_path):
    image_bytes = core.load_image_bytes(image_path)   # FileNotFoundError handled by core.run
    cfg = _thinking_config()

    def once():
        resp = client.models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/png"), prompt],
            config=types.GenerateContentConfig(
                thinking_config=cfg, max_output_tokens=MAX_OUTPUT_TOKENS, seed=42),
        )
        finish = ""
        try:
            finish = str(resp.candidates[0].finish_reason or "")
        except Exception:
            pass
        text = visible_text(resp).strip()
        # Thinking ate the whole budget: billed, nothing to score. The fix is a larger cap, not a
        # better prompt, so it is distinguished from a wrong answer.
        if not text and "MAX_TOKENS" in finish.upper():
            print("    empty response, finish_reason=%s — raise the output cap" % finish, flush=True)
        return text

    return core.retry(once, label="gemini35lite-docvqa", fatal=FATAL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run DocVQA for gemini-3.5-flash-lite via Google AI Studio (native SDK).")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model id; it is BOTH what is called and what the result files are named "
                         "after, so a copied folder cannot relabel another model's numbers")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N questions — a smoke test, not a run. It does NOT resume "
                         "and writes the untagged filename, so never point it at real rows")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--save-every", dest="save_every", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0, help="this shard's index, 0-based")
    ap.add_argument("--total-shards", dest="total_shards", type=int, default=1,
                    help="how many shards the work is split across")
    args = ap.parse_args()

    MODEL = args.model
    cap = _thinking_config()
    print("Gemini 3.5 Flash-Lite (Google AI Studio): model=%s thinking=%s max_output_tokens=%d "
          "seed=42 shard=%d/%d" % (MODEL, cap, MAX_OUTPUT_TOKENS, args.shard, args.total_shards),
          flush=True)
    with open(os.path.join(core.results_dir(MODEL_DIR), "google_params.json"), "w") as fh:
        json.dump({"model": MODEL, "route": "google_aistudio", "thinking": str(cap),
                   "max_output_tokens": MAX_OUTPUT_TOKENS, "seed": 42}, fh, indent=2)

    core.run(MODEL_DIR, MODEL, call, limit=args.limit, sleep_between=args.sleep,
             save_every=args.save_every, shard=args.shard, total_shards=args.total_shards,
             config={"model": MODEL, "route": "google_aistudio",
                     "thinking_budget": THINKING_BUDGET,
                     "max_output_tokens": MAX_OUTPUT_TOKENS, "seed": 42})
