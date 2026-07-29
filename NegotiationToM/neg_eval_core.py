"""Shared objective evaluator for the six NegotiationToM model runners."""

import argparse
import json
import os
import random
import re
import signal
import time

import pandas as pd
from sklearn.metrics import f1_score


# Per-1M-token USD prices, filled in by whoever runs the pilot. Left empty on purpose so the
# pilot never reports an invented cost — with no entry it reports token counts only.
PRICE_PER_1M = {
    # "gpt-4o-mini": {"in": 0.0, "out": 0.0},
}

# Runtime counters, reported by the pilot. Runners update these through record_* below.
STATS = {
    "calls": 0, "empty": 0, "errors": 0, "parse_fail": 0,
    "prompt_tokens": 0, "completion_tokens": 0, "usage_seen": 0,
}


def record_call(prompt_tokens=0, completion_tokens=0, usage_seen=False):
    """Called by each runner after a successful API call."""
    STATS["calls"] += 1
    STATS["prompt_tokens"] += prompt_tokens or 0
    STATS["completion_tokens"] += completion_tokens or 0
    if usage_seen:
        STATS["usage_seen"] += 1


def record_usage(prompt_tokens=0, completion_tokens=0, usage_seen=False):
    """Record billed tokens from an API response that did not yield a usable answer.

    Empty responses are especially expensive for reasoning models: Qwen can consume its entire
    output budget and return empty content. They must count toward cost without being reported as
    successful calls.
    """
    STATS["prompt_tokens"] += prompt_tokens or 0
    STATS["completion_tokens"] += completion_tokens or 0
    if usage_seen:
        STATS["usage_seen"] += 1


def record_empty():
    STATS["empty"] += 1


def record_error():
    STATS["errors"] += 1


# A run in which every call fails should stop, not spend hours producing empty rows that look like
# a completed job. Grok's credits ran out mid-run and the job carried on for five more hours, wrote
# 9,378 empty rows, and exited COMPLETED with 0:0 — Belief_EM 0.0 was an unpaid invoice, not a
# result. Nothing in the pipeline noticed.
CONSECUTIVE_FAILURE_LIMIT = 40
_consecutive_failures = 0


def note_outcome(ok):
    """Track consecutive total failures; abort the run when the provider is clearly not serving."""
    global _consecutive_failures
    if ok:
        _consecutive_failures = 0
        return
    _consecutive_failures += 1
    if _consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
        raise SystemExit(
            f"aborting: {_consecutive_failures} consecutive items returned nothing. "
            "The provider is almost certainly refusing every call (credits, quota or an outage). "
            "Fix that, prune the failed rows with prune_failed_rows.py, then resubmit."
        )


def usage_from(response):
    """Best-effort token extraction; SDKs disagree on the shape, so tolerate all of them.

    **Reasoning tokens must be counted as output.** They are billed at the output rate but reported
    in a separate field, so reading only the visible-answer count understates the bill badly: one
    measured gemini-2.5-flash call was 16 visible tokens against 143 thinking tokens — a 10x
    undercount, which is why an early cost projection for Gemini came in an order of magnitude
    below the actual invoice.
    """
    usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0, False
    prompt = (getattr(usage, "prompt_tokens", None)
              or getattr(usage, "prompt_token_count", None)
              or getattr(usage, "input_tokens", None) or 0)
    completion_total = (getattr(usage, "completion_tokens", None)
                        or getattr(usage, "output_tokens", None))
    if completion_total is not None:
        # OpenAI-compatible APIs include reasoning tokens inside completion_tokens. Their
        # completion_tokens_details.reasoning_tokens value is a subset, not an additional charge.
        completion = completion_total
    else:
        # Gemini reports visible candidates and hidden thoughts separately.
        completion = getattr(usage, "candidates_token_count", None) or 0
        completion += getattr(usage, "thoughts_token_count", None) or 0
        if not completion:
            completion = (getattr(usage, "reasoning_tokens", None)
                          or getattr(usage, "reasoning_token_count", None) or 0)
    return prompt, completion, True


# Surface-form aliases, in the spirit of bbh/xai_eval.py::score_response: be generous about how an
# answer is written, strict about what it says. Keys are matched after the cleanup in norm_item().
ITEM_NORM = {
    "food": "Food", "foods": "Food",
    "water": "Water", "waters": "Water",
    "firewood": "Firewood", "fire wood": "Firewood", "fire-wood": "Firewood",
    "wood": "Firewood", "woods": "Firewood",
    # "not revealed yet" — the model has several ways of saying it
    "not given": "Not Given", "notgiven": "Not Given", "not_given": "Not Given",
    "not-given": "Not Given", "not specified": "Not Given", "unspecified": "Not Given",
    "unknown": "Not Given", "n/a": "Not Given", "na": "Not Given",
    # The sentinel. In gold it marks an unannotated sample and those rows are excluded from the
    # metrics, so in any scored row gold is one of Food/Water/Firewood/Not Given and a model
    # answering "None" is simply wrong — which is correct, since the prompt no longer offers it.
    "none": "None", "null": "None",
}
INTENT_LABELS = [
    "Build-Rapport", "Callout-Fairness", "Describe-Need", "Discover-Preference",
    "No-Intention", "No-Need", "Promote-Coordination", "Show-Empathy",
    "Undermine-Requirements",
]
INTENT_NORM = {label.lower(): label for label in INTENT_LABELS}


def parse_json(text):
    """Parse bare or fenced JSON; return None instead of raising."""
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidate = match.group(1) if match else text
    try:
        return json.loads(candidate.strip())
    except (TypeError, ValueError):
        return None


def retry_delay(error, default=5.0):
    match = re.search(r"try again in ([\d.]+)(ms|s)", str(error), re.IGNORECASE)
    if not match:
        return default
    delay = float(match.group(1))
    if match.group(2).lower() == "ms":
        delay /= 1000
    return max(delay + 1, 1)


class CallTimeout(BaseException):
    """Derives from BaseException on purpose, like KeyboardInterrupt and SystemExit.

    Every runner's call_api ends in `except Exception`, which would otherwise swallow the watchdog
    and treat it as an ordinary API error. The alarm has already fired and been cleared by then, so
    the remaining attempts inside that same call_api run with no watchdog at all — the protection
    silently evaporates after its first use. Deriving from BaseException lets it propagate past
    those handlers to call_and_parse, which retries deliberately.
    """


# Hard ceiling on one call_api invocation, enforced with SIGALRM so it does not depend on the SDK
# honouring its own timeout= argument. Together's client did not: Gemma sat inside a single request
# for over two hours with timeout=18000, and again produced nothing for ~70 minutes with
# timeout=300, in both cases with SLURM still reporting RUNNING and an empty log. SIGALRM
# interrupts the blocking read regardless of what the library does.
# This is a backstop for a hung connection, not the mechanism for bounding work. Bound work with
# max_tokens: the server enforces it, it returns promptly, and finish_reason says what happened.
# A wall-clock kill tells you nothing about why.
#
# Sizing: the ceiling must sit above the slowest *legitimate* call, or it silently truncates good
# work. Measured worst case is a full max_tokens=8192 generation at ~90 tok/s, about 110s, so 200
# leaves real margin. An earlier 150 was too tight against a 32768 budget — every runaway was cut
# by the clock before it could report Length, which destroyed the diagnostic signal.
HARD_CALL_TIMEOUT = 200


def _raise_timeout(signum, frame):
    raise CallTimeout(f"call exceeded {HARD_CALL_TIMEOUT}s hard limit")


def guarded_call(call_api, messages, model):
    """Run call_api under a SIGALRM watchdog. Raises CallTimeout if it overruns."""
    if not hasattr(signal, "SIGALRM"):
        return call_api(messages, model)          # not POSIX; fall back
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(HARD_CALL_TIMEOUT)
    try:
        return call_api(messages, model)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def call_and_parse(call_api, messages, model, max_parse_retries=3):
    last_raw = None
    for attempt in range(max_parse_retries):
        try:
            raw = guarded_call(call_api, messages, model)
        except CallTimeout as error:
            # A hang is usually transient, so it is worth another attempt rather than losing the
            # item — unlike call_api returning None, which means its own retries are exhausted.
            record_error()
            print(f"[{model}] {error} ({attempt + 1}/{max_parse_retries}), retrying", flush=True)
            continue
        if raw is None:
            note_outcome(False)
            return None, None
        last_raw = raw
        parsed = parse_json(raw)
        if isinstance(parsed, dict):
            note_outcome(True)
            return raw, parsed
        print(f"[{model}] JSON parse failed ({attempt + 1}/{max_parse_retries}); calling again",
              flush=True)
    STATS["parse_fail"] += 1
    note_outcome(False)
    return last_raw, None


def format_dialogue(turns):
    return "\n".join(turns)


def desire_messages(dialogue, agent):
    # Gold labels are cutoff-aware: a priority the dialogue has not revealed yet is "Not Given",
    # and "None" means the agent expressed no preference at that level. The prompt has to offer
    # both, otherwise the model can never match those labels.
    system = (
        "Infer an agent's desires in a negotiation over Food, Water, and Firewood, using only "
        "what the dialogue so far reveals. Use 'Not Given' when a priority has not been revealed. "
        "Return only valid JSON with no markdown: {\"high\":\"<item or Not Given>\","
        '"medium":"<item or Not Given>","low":"<item or Not Given>"}'
    )
    user = (
        f"Negotiation dialogue:\n{format_dialogue(dialogue)}\n\n"
        f"What are {agent}'s High, Medium, and Low item preferences?"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def belief_messages(dialogue, agent, opponent):
    system = (
        "Infer one negotiator's belief about the other negotiator's preferences over Food, "
        "Water, and Firewood. Use 'Not Given' when a priority is not revealed. Return only "
        'valid JSON with no markdown: {"high":"<item or Not Given>",'
        '"medium":"<item or Not Given>","low":"<item or Not Given>"}'
    )
    user = (
        f"Negotiation dialogue:\n{format_dialogue(dialogue)}\n\n"
        f"What does {agent} believe about {opponent}'s High, Medium, and Low preferences?"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def intention_messages(dialogue, target):
    labels = "\n".join(f"- {label}" for label in INTENT_LABELS)
    system = (
        "Classify all intents expressed by the target negotiation utterance. Return only valid "
        'JSON with no markdown: {"intents":["label1","label2"]}'
    )
    user = (
        f"Negotiation dialogue:\n{format_dialogue(dialogue)}\n\n"
        f"Target utterance: {target}\n\nChoose one or more labels:\n{labels}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_surface(value):
    """Strip the packaging a model puts around an answer, following the bbh scripts.

    Quotes and backticks (`.strip("\\"'`")` in bbh/xai_eval.py), runs of whitespace, and trailing
    punctuation are all noise. Applied to model output only — gold labels are already canonical and
    must never be rewritten.
    """
    if not isinstance(value, str):
        return ""
    text = value.strip().strip("\"'`").strip()
    text = re.sub(r"\s+", " ", text)          # "Fire   wood" -> "Fire wood"
    return text.strip(" .,;:!")


def norm_item(value):
    text = clean_surface(value)
    if not text:
        return ""
    return ITEM_NORM.get(text.lower(), text.title())


def norm_intent(value):
    """Normalise one intent label: surface cleanup, then space/underscore -> hyphen."""
    text = clean_surface(value)
    if not text:
        return ""
    key = re.sub(r"[\s_]+", "-", text.lower())
    return INTENT_NORM.get(key, text)


def pred_item(prediction, key):
    if not isinstance(prediction, dict):
        return ""
    return norm_item(prediction.get(key) or prediction.get(key.title()) or "")


def is_unannotated(values):
    """True when every priority slot is the sentinel string "None".

    The dataset writes "None" in all three slots at once — 84 samples for agent_1 and 72 for
    agent_2, identically in the desire and belief fields, and never mixed with a real label — while
    the agent{N}_desire dict still holds real values. So "None" marks an unannotated sample rather
    than "the agent wants nothing", exactly like utterance2_intent == "None" on the intention side.
    Such rows are unanswerable and are excluded from the metrics (the rows are still written out).
    """
    return all(str(v).strip() == "None" for v in values)


def desire_em(prediction, gold):
    return int(bool(prediction) and all(
        pred_item(prediction, key.lower()) == gold.get(key, "")
        for key in ("High", "Medium", "Low")
    ))


def belief_em(prediction, high, medium, low):
    return int(bool(prediction) and (
        pred_item(prediction, "high") == high
        and pred_item(prediction, "medium") == medium
        and pred_item(prediction, "low") == low
    ))


def intent_bitmask(labels):
    if isinstance(labels, str):
        labels = labels.split(",")
    normalized = {
        norm_intent(label)
        for label in (labels or []) if isinstance(label, str) and label.strip()
    }
    return [int(label in normalized) for label in INTENT_LABELS]


def model_slug(model):
    return model.split("/")[-1].replace(".", "_").replace("/", "-")


def shard_slice(rows, shard, total_shards):
    size = (len(rows) + total_shards - 1) // total_shards
    return rows[shard * size:min((shard + 1) * size, len(rows))]


def load_checkpoint(path):
    done, rows = set(), []
    if not os.path.exists(path):
        return done, rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add(row["uid"])
            rows.append(row)
    return done, rows


def save_checkpoint(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_paths(script_dir, task, model, shard, total_shards, pilot=False):
    # Pilot output is kept under results/pilot/ so it can never overwrite a full run.
    parts = [script_dir, "results"] + (["pilot"] if pilot else []) + [task]
    out_dir = os.path.join(*parts)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_shard{shard}of{total_shards}" if total_shards > 1 else ""
    stem = model_slug(model) + tag
    return out_dir, stem, os.path.join(out_dir, stem + ".jsonl")


def scorable(task, rows):
    """Drop rows the dataset never annotated — they are unanswerable, not wrong answers.

    See is_unannotated(). Returns (kept_rows, dropped_count).
    """
    if task == "desire":
        keep = [r for r in rows if not is_unannotated((r["gold_desire"] or {}).values())]
    elif task == "belief":
        keep = [r for r in rows
                if not is_unannotated((r["gold_high"], r["gold_med"], r["gold_low"]))]
    else:
        keep = rows
    return keep, len(rows) - len(keep)


def write_task_outputs(task, rows, out_dir, stem):
    """Write columns in exactly the same order as Output_template."""
    df = pd.DataFrame(rows)
    scored, dropped = scorable(task, rows)
    if dropped:
        print(f"[{task}] excluded {dropped}/{len(rows)} unannotated rows from the metrics",
              flush=True)
    sdf = pd.DataFrame(scored)

    if task == "desire":
        columns = ["uid", "dialogue_id", "agent", "gold_desire", "pred", "raw_response", "desire_em"]
        # Desire_EM is the headline number and excludes unannotated rows. Desire_EM_all keeps them
        # (where the model scores 0 by construction) purely so the figure can be lined up against a
        # published table that may have used the other convention. Both come from the same rows —
        # no extra API calls, and the raw output is unchanged either way.
        metrics = [{"metric": "Desire_EM", "score": sdf["desire_em"].mean()},
                   {"metric": "Desire_EM_all", "score": df["desire_em"].mean()}]
    elif task == "belief":
        columns = ["uid", "dialogue_id", "agent", "opponent", "gold_high", "gold_med",
                   "gold_low", "pred", "belief_em", "raw_response"]
        metrics = [{"metric": "Belief_EM", "score": sdf["belief_em"].mean()},
                   {"metric": "Belief_EM_all", "score": df["belief_em"].mean()}]
    else:
        columns = ["uid", "dialogue_id", "utt_idx", "target_utterance", "gold_intent",
                   "gold_bitmask", "pred_intents", "pred_bitmask", "raw_response"]
        gold = list(sdf["gold_bitmask"])
        pred = list(sdf["pred_bitmask"])
        metrics = [
            {"metric": "Intent_Micro_F1", "score": f1_score(gold, pred, average="micro", zero_division=0)},
            {"metric": "Intent_Macro_F1", "score": f1_score(gold, pred, average="macro", zero_division=0)},
        ]
    metrics.append({"metric": f"{task}_scored_rows", "score": len(scored)})
    # every row is still written out; only the metrics exclude the unannotated ones
    df.reindex(columns=columns).to_csv(os.path.join(out_dir, stem + ".csv"), index=False)
    pd.DataFrame(metrics, columns=["metric", "score"]).to_csv(
        os.path.join(out_dir, stem + "_overall.csv"), index=False
    )


def run_desire(data, call_api, model, script_dir, shard, total_shards, save_every, pilot=False):
    rows = []
    for sample in data:
        for agent, prefix in (("agent_1", "agent1_desire"), ("agent_2", "agent2_desire")):
            # Use the cutoff-aware flat fields, not the sample["agent{N}_desire"] dict: the dict is
            # always a complete Food/Water/Firewood permutation, so it can never express the
            # "Not Given" / "None" labels the benchmark's Desire label set requires.
            gold_desire = {
                "High": sample[prefix + "_high"],
                "Medium": sample[prefix + "_medium"],
                "Low": sample[prefix + "_low"],
            }
            rows.append({"dialogue_id": sample["dialogue_id"], "dialogue": sample["dialogue"],
                         "agent": agent, "gold_desire": gold_desire})
    rows = shard_slice(rows, shard, total_shards)
    out_dir, stem, checkpoint = output_paths(script_dir, "desire", model, shard, total_shards, pilot)
    done, results = load_checkpoint(checkpoint)
    for item in rows:
        uid = f"{item['dialogue_id']}_{item['agent']}_desire"
        if uid in done:
            continue
        raw, pred = call_and_parse(call_api, desire_messages(item["dialogue"], item["agent"]), model)
        results.append({"uid": uid, "dialogue_id": item["dialogue_id"], "agent": item["agent"],
                        "gold_desire": item["gold_desire"], "pred": pred,
                        "raw_response": raw or "", "desire_em": desire_em(pred, item["gold_desire"])})
        done.add(uid)
        if len(results) % save_every == 0:
            save_checkpoint(checkpoint, results)
    save_checkpoint(checkpoint, results)
    write_task_outputs("desire", results, out_dir, stem)


def run_belief(data, call_api, model, script_dir, shard, total_shards, save_every, pilot=False):
    rows = []
    for sample in data:
        for agent, opponent, prefix in (
            ("agent_1", "agent_2", "agent1_belief"), ("agent_2", "agent_1", "agent2_belief")
        ):
            rows.append({"dialogue_id": sample["dialogue_id"], "dialogue": sample["dialogue"],
                         "agent": agent, "opponent": opponent,
                         "gold_high": sample[prefix + "_high"],
                         "gold_med": sample[prefix + "_medium"],
                         "gold_low": sample[prefix + "_low"]})
    rows = shard_slice(rows, shard, total_shards)
    out_dir, stem, checkpoint = output_paths(script_dir, "belief", model, shard, total_shards, pilot)
    done, results = load_checkpoint(checkpoint)
    for item in rows:
        uid = f"{item['dialogue_id']}_{item['agent']}_belief"
        if uid in done:
            continue
        messages = belief_messages(item["dialogue"], item["agent"], item["opponent"])
        raw, pred = call_and_parse(call_api, messages, model)
        results.append({"uid": uid, "dialogue_id": item["dialogue_id"], "agent": item["agent"],
                        "opponent": item["opponent"], "gold_high": item["gold_high"],
                        "gold_med": item["gold_med"], "gold_low": item["gold_low"], "pred": pred,
                        "belief_em": belief_em(pred, item["gold_high"], item["gold_med"], item["gold_low"]),
                        "raw_response": raw or ""})
        done.add(uid)
        if len(results) % save_every == 0:
            save_checkpoint(checkpoint, results)
    save_checkpoint(checkpoint, results)
    write_task_outputs("belief", results, out_dir, stem)


def run_intention(data, call_api, model, script_dir, shard, total_shards, save_every, pilot=False):
    rows = []
    for sample in data:
        turns = sample["dialogue"]
        # Only even-length dialogues end on a complete exchange and carry two annotated targets.
        # Odd-length ones (the final, uncut sample of a dialogue) annotate the last utterance only
        # and set utterance2_intent to the string "None" — which is not one of the 9 labels, so
        # scoring it would silently produce an all-zero gold bitmask.
        if sample["utterance2_intent"] == "None":
            rows.append({"dialogue_id": sample["dialogue_id"], "dialogue": turns, "utt_idx": 1,
                         "target": turns[-1], "gold_intent": sample["utterance1_intent"]})
        else:
            rows.append({"dialogue_id": sample["dialogue_id"], "dialogue": turns, "utt_idx": 1,
                         "target": turns[-2], "gold_intent": sample["utterance1_intent"]})
            rows.append({"dialogue_id": sample["dialogue_id"], "dialogue": turns, "utt_idx": 2,
                         "target": turns[-1], "gold_intent": sample["utterance2_intent"]})
    rows = shard_slice(rows, shard, total_shards)
    out_dir, stem, checkpoint = output_paths(script_dir, "intention", model, shard, total_shards, pilot)
    done, results = load_checkpoint(checkpoint)
    for item in rows:
        uid = f"{item['dialogue_id']}_utt{item['utt_idx']}_intention"
        if uid in done:
            continue
        raw, pred = call_and_parse(call_api, intention_messages(item["dialogue"], item["target"]), model)
        pred_intents = (pred or {}).get("intents", [])
        if isinstance(pred_intents, str):
            pred_intents = [part.strip() for part in pred_intents.split(",") if part.strip()]
        results.append({"uid": uid, "dialogue_id": item["dialogue_id"], "utt_idx": item["utt_idx"],
                        "target_utterance": item["target"], "gold_intent": item["gold_intent"],
                        "gold_bitmask": intent_bitmask(item["gold_intent"]),
                        "pred_intents": pred_intents, "pred_bitmask": intent_bitmask(pred_intents),
                        "raw_response": raw or ""})
        done.add(uid)
        if len(results) % save_every == 0:
            save_checkpoint(checkpoint, results)
    save_checkpoint(checkpoint, results)
    write_task_outputs("intention", results, out_dir, stem)


def pilot_report(model, tasks, script_dir, stem, n_samples, n_full, elapsed):
    """Print the go/no-go summary for a pilot run: health counters, scores, and projections."""
    line = "=" * 68
    print(f"\n{line}\n PILOT REPORT — {model}\n{line}")

    print(f"\n[sample]  {n_samples} / {n_full} dialogues "
          f"({n_samples / n_full * 100:.1f}%), seed-fixed so every model sees the same subset")

    calls = STATS["calls"]
    print(f"\n[health]  successful calls : {calls}")
    print(f"          empty responses  : {STATS['empty']}")
    print(f"          API errors       : {STATS['errors']}")
    print(f"          JSON parse fails : {STATS['parse_fail']}")
    if calls == 0:
        print("\n  *** NO SUCCESSFUL CALLS — check the key name, SDK and model id before scaling up.")
    else:
        bad = STATS["empty"] + STATS["errors"] + STATS["parse_fail"]
        rate = bad / (calls + bad) * 100
        verdict = "looks healthy" if rate < 5 else "HIGH FAILURE RATE — investigate before scaling"
        print(f"          failure rate     : {rate:.1f}%  ({verdict})")

    print("\n[scores]")
    for task in tasks:
        path = os.path.join(script_dir, "results", "pilot", task, stem + "_overall.csv")
        if os.path.exists(path):
            for row in pd.read_csv(path).to_dict("records"):
                print(f"          {row['metric']:<18} {row['score']:.4f}")

    scale = n_full / n_samples if n_samples else 0
    print(f"\n[cost]    elapsed          : {elapsed / 60:.1f} min for {calls} calls")
    print(f"          projected full   : {elapsed * scale / 3600:.1f} h single-threaded, "
          f"{elapsed * scale / 3600 / 5:.1f} h across 5 shards")
    if STATS["usage_seen"]:
        pin, pout = STATS["prompt_tokens"], STATS["completion_tokens"]
        print(f"          tokens           : {pin:,} in / {pout:,} out "
              f"({STATS['usage_seen']} API responses reported usage)")
        print(f"          output/response  : {pout / STATS['usage_seen']:.1f} tokens average")
        print(f"          projected full   : {pin * scale:,.0f} in / {pout * scale:,.0f} out")
        price = PRICE_PER_1M.get(model)
        if price:
            cost = (pin * price["in"] + pout * price["out"]) / 1e6
            print(f"          projected cost   : ${cost * scale:.2f} USD")
        else:
            print(f"          projected cost   : add '{model}' to PRICE_PER_1M to get a figure")
    else:
        print("          tokens           : this SDK did not report usage")
    print(f"{line}\n")


def run_cli(call_api, default_model, script_file):
    script_dir = os.path.dirname(os.path.abspath(script_file))
    benchmark_root = os.path.dirname(script_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--task", default="all", choices=["desire", "belief", "intention", "all"])
    parser.add_argument("--data", default=os.path.join(benchmark_root, "NegotiationToM.json"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--pilot", action="store_true",
                        help="run a fraction of the data, write to results/pilot/, print a report")
    parser.add_argument("--pilot-frac", type=float, default=0.10)
    parser.add_argument("--pilot-seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.shard < args.total_shards:
        parser.error("--shard must be in [0, --total-shards)")
    if args.pilot and not 0 < args.pilot_frac <= 1:
        parser.error("--pilot-frac must be in (0, 1]")

    with open(args.data, encoding="utf-8") as handle:
        data = json.load(handle)
    n_full = len(data)

    if args.pilot:
        k = max(1, round(n_full * args.pilot_frac))
        # Shuffle once with a fixed seed, then take a prefix — so a smaller --pilot-frac yields a
        # strict subset of a larger one. (random.sample would return an unrelated set instead,
        # making a cheaper pilot for a slow model incomparable with the others.)
        order = list(range(n_full))
        random.Random(args.pilot_seed).shuffle(order)
        data = [data[i] for i in order[:k]]
        print(f"[pilot] {k}/{n_full} dialogues ({k / n_full * 100:.1f}%), seed={args.pilot_seed}")

    tasks = ["desire", "belief", "intention"] if args.task == "all" else [args.task]
    started = time.time()
    for task in tasks:
        globals()[f"run_{task}"](
            data, call_api, args.model, script_dir, args.shard, args.total_shards,
            args.save_every, args.pilot,
        )
    elapsed = time.time() - started

    tag = f"_shard{args.shard}of{args.total_shards}" if args.total_shards > 1 else ""
    stem = model_slug(args.model) + tag
    if args.task == "all":
        results_root = os.path.join(script_dir, "results", "pilot" if args.pilot else "")
        summary = []
        for task in ("desire", "belief", "intention"):
            path = os.path.join(results_root, task, stem + "_overall.csv")
            if os.path.exists(path):
                summary.extend(pd.read_csv(path).to_dict("records"))
        summary_path = os.path.join(results_root, stem + "_negotiation_overall.csv")
        pd.DataFrame(summary, columns=["metric", "score"]).to_csv(summary_path, index=False)

    if args.pilot:
        pilot_report(args.model, tasks, script_dir, stem, len(data), n_full, elapsed)
