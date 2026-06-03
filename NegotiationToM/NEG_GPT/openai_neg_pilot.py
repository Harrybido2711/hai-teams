"""
Pilot script: run NegotiationToM eval on 50 random samples to estimate
time, cost, and correctness before launching the full Quest job.

Usage:
    python openai_neg_pilot.py --data ../NegotiationToM.json --model gpt-4o-mini --seed 42
"""
import argparse
import json
import os
import random
import re
import time

import pandas as pd
from sklearn.metrics import f1_score
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
client = OpenAI(api_key=api_key)

# gpt-4o-mini pricing (USD per 1M tokens) — update if OpenAI changes rates
PRICE_INPUT  = 0.15
PRICE_OUTPUT = 0.60

INTENT_LABELS = [
    "Build-Rapport", "Callout-Fairness", "Describe-Need", "Discover-Preference",
    "No-Intention", "No-Need", "Promote-Coordination", "Show-Empathy", "Undermine-Requirements",
]

_ITEM_NORM = {
    "food": "Food", "water": "Water", "firewood": "Firewood",
    "not given": "Not Given", "none": "None",
}
_INTENT_NORM = {lbl.lower(): lbl for lbl in INTENT_LABELS}


def norm_item(s):
    if not isinstance(s, str):
        return ""
    return _ITEM_NORM.get(s.strip().lower(), s.strip().title())


def norm_intent(s):
    if not isinstance(s, str):
        return ""
    return _INTENT_NORM.get(s.strip().lower(), s.strip())


def _pred_item(pred, key):
    val = pred.get(key) or pred.get(key.title()) or ""
    return norm_item(val)


# ── prompt builders (identical to main script) ────────────────────────────────

def format_dialogue(turns):
    return "\n".join(turns)


def build_desire_messages(dialogue, agent):
    sys = (
        "You are evaluating an agent's desires in a negotiation over three items: Food, Water, Firewood. "
        "Each item has exactly one priority level (High, Medium, Low) and each level maps to exactly one item. "
        'Respond ONLY with valid JSON: {"high": "<item>", "medium": "<item>", "low": "<item>"}'
    )
    user = (
        f"Negotiation dialogue:\n{format_dialogue(dialogue)}\n\n"
        f"Based on the dialogue, what are {agent}'s preference priorities?\n"
        "Assign exactly one priority (High, Medium, Low) to each of Food, Water, Firewood."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def build_belief_messages(dialogue, agent, opponent):
    sys = (
        f"You are evaluating what {agent} believes about {opponent}'s preferences "
        "in a negotiation over Food, Water, Firewood. "
        "Each slot (high/medium/low) maps to an item or 'Not Given' if the dialogue does not reveal that belief. "
        'Respond ONLY with valid JSON: {"high": "<item or Not Given>", "medium": "<item or Not Given>", "low": "<item or Not Given>"}'
    )
    user = (
        f"Negotiation dialogue:\n{format_dialogue(dialogue)}\n\n"
        f"Based on the dialogue, what does {agent} believe {opponent}'s preference priorities are "
        "for Food, Water, and Firewood? Use 'Not Given' where the belief cannot be determined."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def build_intention_messages(dialogue, target_utterance):
    intent_list = "\n".join(f"- {lbl}" for lbl in INTENT_LABELS)
    sys = (
        "You are classifying the intent(s) of a negotiation utterance. "
        "An utterance may have one or more intents from the provided list. "
        'Respond ONLY with valid JSON: {"intents": ["label1", "label2"]}'
    )
    user = (
        f"Negotiation dialogue:\n{format_dialogue(dialogue)}\n\n"
        f'Target utterance: "{target_utterance}"\n\n'
        f"Classify the intent(s) of the target utterance. Choose one or more from:\n{intent_list}"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ── API call — returns (text, prompt_tokens, completion_tokens) ───────────────

def parse_json(text):
    try:
        if "```json" in text:
            text = re.search(r"```json\s*([\s\S]*?)```", text).group(1)
        return json.loads(text.strip())
    except Exception:
        return None


def call_api(messages, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.6,
                max_tokens=256,
                timeout=60,
            )
            time.sleep(2.0)
            text = resp.choices[0].message.content.strip()
            if not text:
                print(f"  Empty response (attempt {attempt+1}/{max_retries}), retrying...")
                continue
            usage = resp.usage
            return text, usage.prompt_tokens, usage.completion_tokens
        except Exception as e:
            err = str(e)
            print(f"  API error (attempt {attempt+1}/{max_retries}): {e}")
            if "insufficient_quota" in err:
                raise SystemExit("OpenAI quota exhausted — top up billing and retry.")
            if "requests per day" in err:
                print("  RPD limit hit — stopping.")
                return None, 0, 0
            wait = 5.0
            m = re.search(r"try again in ([\d.]+)(ms|s)", err)
            if m:
                wait = float(m.group(1)) / 1000 if m.group(2) == "ms" else float(m.group(1))
                wait = max(wait + 1, 1)
            print(f"  Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return None, 0, 0


def call_and_parse(messages, model, max_parse_retries=3):
    """Call API and parse JSON; retries the full call if parse fails or response is empty.
    Returns (raw_text, parsed_dict, total_input_tokens, total_output_tokens)."""
    total_in = total_out = 0
    last_raw = None
    for attempt in range(max_parse_retries):
        raw, tok_in, tok_out = call_api(messages, model)
        total_in += tok_in
        total_out += tok_out
        if raw is None:
            return None, None, total_in, total_out  # hard stop
        last_raw = raw
        parsed = parse_json(raw)
        if parsed is not None:
            return raw, parsed, total_in, total_out
        print(f"  JSON parse failed (attempt {attempt+1}/{max_parse_retries}), retrying call...")
    print("  Max parse retries reached — recording as unparseable (score 0).")
    return last_raw, None, total_in, total_out


# ── scoring ───────────────────────────────────────────────────────────────────

def desire_em(pred, gold_desire):
    if not pred:
        return 0
    return int(
        _pred_item(pred, "high")   == gold_desire.get("High", "") and
        _pred_item(pred, "medium") == gold_desire.get("Medium", "") and
        _pred_item(pred, "low")    == gold_desire.get("Low", "")
    )


def belief_em(pred, gold_high, gold_med, gold_low):
    if not pred:
        return 0
    return int(
        _pred_item(pred, "high")   == gold_high and
        _pred_item(pred, "medium") == gold_med and
        _pred_item(pred, "low")    == gold_low
    )


def intent_bitmask(intent_str):
    labels = {norm_intent(l) for l in intent_str.split(",") if l.strip()}
    return [1 if lbl in labels else 0 for lbl in INTENT_LABELS]


def pred_intent_bitmask(pred_intents):
    normalized = {norm_intent(i) for i in (pred_intents or [])}
    return [1 if lbl in normalized else 0 for lbl in INTENT_LABELS]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="gpt-4o-mini")
    parser.add_argument("--data",    default="../NegotiationToM.json")
    parser.add_argument("--n",       type=int, default=50, help="Number of samples to pilot")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", default="results/pilot")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)

    random.seed(args.seed)
    pilot_data = random.sample(data, min(args.n, len(data)))
    os.makedirs(args.out_dir, exist_ok=True)

    total_input_tokens = 0
    total_output_tokens = 0
    call_count = 0
    t_start = time.time()

    desire_rows, belief_rows, intention_rows = [], [], []

    print(f"\n{'='*60}")
    print(f"NegotiationToM Pilot — {args.n} samples, model={args.model}")
    print(f"{'='*60}\n")

    for i, s in enumerate(pilot_data):
        dlg  = s["dialogue"]
        did  = s["dialogue_id"]
        print(f"[{i+1}/{len(pilot_data)}] dialogue_id={did}")

        # ── desire ───────────────────────────────────────────────────────────
        for agent, gold in [("agent_1", s["agent1_desire"]), ("agent_2", s["agent2_desire"])]:
            raw, pred, tok_in, tok_out = call_and_parse(build_desire_messages(dlg, agent), args.model)
            total_input_tokens += tok_in; total_output_tokens += tok_out; call_count += 1
            desire_rows.append({
                "dialogue_id": did, "agent": agent,
                "gold_desire": gold, "pred": pred,
                "desire_em": desire_em(pred, gold),
                "raw_response": raw or "",
            })

        # ── belief ────────────────────────────────────────────────────────────
        for agent, opp, gh, gm, gl in [
            ("agent_1", "agent_2", s["agent1_belief_high"], s["agent1_belief_medium"], s["agent1_belief_low"]),
            ("agent_2", "agent_1", s["agent2_belief_high"], s["agent2_belief_medium"], s["agent2_belief_low"]),
        ]:
            raw, pred, tok_in, tok_out = call_and_parse(build_belief_messages(dlg, agent, opp), args.model)
            total_input_tokens += tok_in; total_output_tokens += tok_out; call_count += 1
            belief_rows.append({
                "dialogue_id": did, "agent": agent, "opponent": opp,
                "gold_high": gh, "gold_med": gm, "gold_low": gl,
                "pred": pred,
                "belief_em": belief_em(pred, gh, gm, gl),
                "raw_response": raw or "",
            })

        # ── intention ─────────────────────────────────────────────────────────
        utt_pairs = []
        if len(dlg) >= 2:
            utt_pairs.append((1, dlg[-2], s["utterance1_intent"]))
        utt_pairs.append((2, dlg[-1], s["utterance2_intent"]))

        for utt_idx, target, gold_intent in utt_pairs:
            raw, pred, tok_in, tok_out = call_and_parse(build_intention_messages(dlg, target), args.model)
            total_input_tokens += tok_in; total_output_tokens += tok_out; call_count += 1
            pred_intents = (pred or {}).get("intents", [])
            intention_rows.append({
                "dialogue_id": did, "utt_idx": utt_idx,
                "target_utterance": target,
                "gold_intent": gold_intent,
                "gold_bitmask": intent_bitmask(gold_intent),
                "pred_intents": pred_intents,
                "pred_bitmask": pred_intent_bitmask(pred_intents),
                "raw_response": raw or "",
            })

    elapsed = time.time() - t_start

    # ── save CSVs ─────────────────────────────────────────────────────────────
    pd.DataFrame(desire_rows).to_csv(f"{args.out_dir}/pilot_desire.csv", index=False)
    pd.DataFrame(belief_rows).to_csv(f"{args.out_dir}/pilot_belief.csv", index=False)
    pd.DataFrame(intention_rows).to_csv(f"{args.out_dir}/pilot_intention.csv", index=False)

    # ── score ─────────────────────────────────────────────────────────────────
    df_d = pd.DataFrame(desire_rows)
    df_b = pd.DataFrame(belief_rows)
    df_i = pd.DataFrame(intention_rows)

    desire_score = df_d["desire_em"].mean()
    belief_score = df_b["belief_em"].mean()
    micro_f1 = f1_score(list(df_i["gold_bitmask"]), list(df_i["pred_bitmask"]), average="micro", zero_division=0)
    macro_f1 = f1_score(list(df_i["gold_bitmask"]), list(df_i["pred_bitmask"]), average="macro", zero_division=0)

    # ── cost estimate ─────────────────────────────────────────────────────────
    est_cost = (total_input_tokens * PRICE_INPUT + total_output_tokens * PRICE_OUTPUT) / 1_000_000
    full_scale = len(data) / len(pilot_data)
    est_full_cost = est_cost * full_scale
    est_full_time_hr = (elapsed / len(pilot_data)) * len(data) / 3600

    print(f"\n{'='*60}")
    print(f"PILOT RESULTS  ({len(pilot_data)} samples, {call_count} API calls)")
    print(f"{'='*60}")
    print(f"  Desire EM:        {desire_score:.4f}  ({df_d['desire_em'].sum()}/{len(df_d)})")
    print(f"  Belief EM:        {belief_score:.4f}  ({df_b['belief_em'].sum()}/{len(df_b)})")
    print(f"  Intent Micro F1:  {micro_f1:.4f}")
    print(f"  Intent Macro F1:  {macro_f1:.4f}")
    print(f"\n  Tokens used:      {total_input_tokens:,} input  /  {total_output_tokens:,} output")
    print(f"  Pilot cost:       ${est_cost:.4f}")
    print(f"\n  --- Full-dataset projection ({len(data)} samples) ---")
    print(f"  Estimated cost:   ${est_full_cost:.2f}  (at current token ratios)")
    print(f"  Estimated time:   {est_full_time_hr:.1f} hrs  (single-threaded, ~2s/call)")
    print(f"  Elapsed (pilot):  {elapsed:.0f}s")
    print(f"\n  Saved → {args.out_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
