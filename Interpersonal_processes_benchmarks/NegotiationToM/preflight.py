"""Preflight gate for the NegotiationToM eval scripts.

Run this on Quest before submitting any pilot or full job. It exercises everything that can only
fail at runtime — SDK imports, key names, call signatures, model ids, data layout, write access —
using one real API call per provider, and exits non-zero if anything fails.

The point is to turn a 24-hour feedback loop ("job finished, every score is 0, log is empty") into
about 30 seconds.

    python preflight.py                  # check everything
    python preflight.py --skip-api       # offline checks only, no API calls
    python preflight.py --only gpt xai   # check a subset of providers
"""

import argparse
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))

# One entry per model folder. `probe` performs a single minimal call and returns the reply text.
PROVIDERS = {}


def register(name, folder, key_names, model, sdk_modules, probe):
    PROVIDERS[name] = {
        "folder": folder, "key_names": key_names, "model": model,
        "sdk_modules": sdk_modules, "probe": probe,
    }


# The probe sends a real prompt built by neg_eval_core, not a synthetic one. A hand-written probe
# ("You are a JSON API." + a toy request) once tripped grok's SAFETY_CHECK_TYPE_BIO filter as a
# false positive while the genuine eval prompts passed — a probe that does not resemble production
# traffic reports failures that do not exist, and can miss ones that do.
TOY_DIALOGUE = [
    "agent_1: Hello! I could really use extra firewood for the cold nights.",
    "agent_2: Hi there! I mostly need food, so maybe we can trade.",
]


def probe_messages():
    import neg_eval_core as core
    return core.desire_messages(TOY_DIALOGUE, "agent_1")


def split_messages(messages):
    system = next(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    return system, user


def probe_openai_compatible(api_key, model, base_url=None, timeout=60):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    resp = client.chat.completions.create(
        model=model, messages=probe_messages(), temperature=0, max_tokens=8192,
    )
    msg = resp.choices[0].message
    return (msg.content or getattr(msg, "reasoning_content", None) or "").strip()


def probe_gpt(api_key, model):
    return probe_openai_compatible(api_key, model)


def probe_together(api_key, model, reasoning=None):
    from together import Together
    client = Together(api_key=api_key, timeout=120)
    kwargs = dict(model=model, messages=probe_messages(), temperature=0, max_tokens=8192)
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def probe_deepseek(api_key, model):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=300)
    response = client.chat.completions.create(
        model=model,
        messages=probe_messages(),
        temperature=0,
        max_tokens=8192,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return (response.choices[0].message.content or "").strip()


def probe_gemini(api_key, model):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    system, user = split_messages(probe_messages())
    resp = client.models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (resp.text or "").strip()


def probe_xai(api_key, model):
    from xai_sdk import Client
    from xai_sdk.chat import system as xai_system
    from xai_sdk.chat import user as xai_user
    client = Client(api_key=api_key, timeout=300)
    system, user = split_messages(probe_messages())
    # Mirrors xai_neg_eval.py exactly: these two kwargs are the ones worth proving out.
    chat = client.chat.create(model=model, max_tokens=8192, temperature=0)
    chat.append(xai_system(system))
    chat.append(xai_user(user))
    return (chat.sample().content or "").strip()


register("gpt", "NEG_GPT", ["OPENAI_API_KEY", "API_KEY"], "gpt-4o-mini",
         ["openai"], probe_gpt)
register("gemini", "NEG_Gemini", ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "gemini-2.5-flash",
         ["google.genai"], probe_gemini)
register("xai", "NEG_XAI", ["GROK_API_KEY", "XAI_API_KEY"], "grok-3-mini",
         ["xai_sdk"], probe_xai)
register("qwen", "NEG_Qwen", ["TOGETHER_API_KEY"], "Qwen/Qwen3.5-9B",
         ["together"], lambda key, model: probe_together(key, model, {"enabled": False}))
register("gemma", "NEG_Gemma", ["TOGETHER_API_KEY"], "google/gemma-4-31B-it",
         ["together"], probe_together)
register("deepseek", "NEG_Deepseek", ["DEEPSEEK_API_KEY"], "deepseek-reasoner",
         ["openai"], probe_deepseek)


# ── reporting ─────────────────────────────────────────────────────────────────

failures = []


# Patterns for "this run cannot possibly work". The obvious two were not enough: the earlier set
# (insufficient_quota / requests per day / billing / rate_limit) missed both failures that actually
# occurred. xAI says "used all available credits or reached its monthly spending limit" and matched
# nothing at all, so 36,124 occurrences went unreported while a job burned five hours; Gemini says
# "generate_requests_per_model_per_day" with underscores, which the spaced pattern missed and only
# "billing" caught by luck. Match on how providers really word it, not on how we assume they do.
BILLING_PATTERNS = [
    ("credits exhausted",
     ["used all available credits", "spending limit", "insufficient_quota",
      "insufficient credit", "billing", "payment required", "402"]),
    ("daily/rate quota",
     ["requests per day", "requests_per_day", "per_model_per_day", "resource_exhausted",
      "quota exceeded", "rate_limit", "ratelimit", "429"]),
    ("permission denied",
     ["permission_denied", "permissiondenied"]),
]


def classify_billing(error):
    """Name the billing/quota condition in an exception, or None if it is something else."""
    text = str(error).lower()
    for label, needles in BILLING_PATTERNS:
        if any(n in text for n in needles):
            return label
    return None


def check(label, ok, detail="", fatal=True):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""),
          flush=True)
    if not ok and fatal:
        failures.append(label)
    return ok


def section(title):
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


# ── checks ────────────────────────────────────────────────────────────────────

def check_shared_imports():
    section("Shared dependencies")
    for mod in ("pandas", "sklearn.metrics", "dotenv", "yaml"):
        try:
            __import__(mod)
            check(f"import {mod}", True)
        except Exception as error:
            check(f"import {mod}", False, f"{type(error).__name__}: {error}")


def check_core():
    section("Shared evaluator")
    try:
        sys.path.insert(0, ROOT)
        import neg_eval_core as core
        check("import neg_eval_core", True)
        for fn in ("run_desire", "run_belief", "run_intention", "run_cli",
                   "record_call", "usage_from"):
            check(f"neg_eval_core.{fn}", hasattr(core, fn))
        return core
    except Exception as error:
        check("import neg_eval_core", False, f"{type(error).__name__}: {error}")
        return None


def check_env():
    section("Environment file")
    path = os.path.join(ROOT, ".env")
    if not check(f".env exists at {path}", os.path.isfile(path)):
        return {}
    from dotenv import dotenv_values
    values = {k: v for k, v in dotenv_values(path).items() if v}
    check(".env has non-empty values", bool(values), f"{len(values)} keys")
    return values


def check_data(core):
    section("Dataset")
    path = os.path.join(ROOT, "NegotiationToM.json")
    if not check("NegotiationToM.json exists", os.path.isfile(path)):
        return
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    check("2380 dialogues", len(data) == 2380, f"found {len(data)}")

    required = ["dialogue_id", "dialogue", "utterance1_intent", "utterance2_intent",
                "agent1_desire_high", "agent1_belief_high"]
    missing = [f for f in required if f not in data[0]]
    check("required fields present", not missing, f"missing: {missing}" if missing else "")

    if core is None:
        return
    # Row counts must match the fixed odd/even branch — a regression here silently changes scores.
    intention = sum(1 if s["utterance2_intent"] == "None" else 2 for s in data)
    check("intention rows == 4618", intention == 4618, f"computed {intention}")

    flat = set()
    for s in data:
        flat |= {s["agent1_desire_high"], s["agent1_desire_medium"], s["agent1_desire_low"]}
    check("desire gold includes Not Given/None",
          {"Not Given", "None"} <= flat, f"values: {sorted(flat)}")


def check_writable():
    section("Output directories")
    for name, spec in PROVIDERS.items():
        folder = os.path.join(ROOT, spec["folder"])
        if not check(f"{spec['folder']}/ exists", os.path.isdir(folder)):
            continue
        probe_path = os.path.join(folder, "results", ".preflight")
        try:
            os.makedirs(os.path.dirname(probe_path), exist_ok=True)
            with open(probe_path, "w") as handle:
                handle.write("ok")
            os.remove(probe_path)
            check(f"{spec['folder']}/results writable", True)
        except Exception as error:
            check(f"{spec['folder']}/results writable", False,
                  f"{type(error).__name__}: {error}")


def check_providers(env, selected, skip_api):
    section("Providers")
    for name in selected:
        spec = PROVIDERS[name]
        print(f"\n[{name}]  {spec['model']}", flush=True)

        for mod in spec["sdk_modules"]:
            try:
                __import__(mod)
                check(f"import {mod}", True)
            except Exception as error:
                check(f"import {mod}", False, f"{type(error).__name__}: {error}")
                continue

        key = next((env[k] for k in spec["key_names"] if env.get(k)), None)
        found = next((k for k in spec["key_names"] if env.get(k)), None)
        if not check(f"key present ({' or '.join(spec['key_names'])})", bool(key),
                     f"using {found}" if found else "none of these are set in .env"):
            continue

        if skip_api:
            print("  SKIP  live API call (--skip-api)", flush=True)
            continue

        try:
            reply = spec["probe"](key, spec["model"])
            if check("live API call returns non-empty", bool(reply)):
                print(f"        reply: {reply[:80]}", flush=True)
        except Exception as error:
            kind = classify_billing(error)
            if kind:
                check(f"live API call — {kind}", False, str(error)[:160])
            else:
                check("live API call", False, f"{type(error).__name__}: {error}")
            traceback.print_exc(limit=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-api", action="store_true",
                        help="run offline checks only, no API calls")
    parser.add_argument("--only", nargs="+", choices=sorted(PROVIDERS),
                        help="check only these providers")
    args = parser.parse_args()
    selected = args.only or sorted(PROVIDERS)

    print("=" * 68)
    print(" NegotiationToM preflight")
    print(f" root: {ROOT}")
    print(f" python: {sys.version.split()[0]}  ({sys.executable})")
    print("=" * 68)

    check_shared_imports()
    core = check_core()
    env = check_env()
    check_data(core)
    check_writable()
    check_providers(env, selected, args.skip_api)

    print("\n" + "=" * 68)
    if failures:
        print(f" PREFLIGHT FAILED — {len(failures)} problem(s):")
        for item in failures:
            print(f"   - {item}")
        print("=" * 68)
        return 1
    print(" PREFLIGHT PASSED — safe to submit jobs")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
