import argparse
import json
import os
import re
import time

import pandas as pd
from sklearn.metrics import f1_score
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
client = OpenAI(api_key=api_key)

ITEMS = ["Food", "Water", "Firewood"]
NOT_GIVEN = "Not Given"
INTENT_LABELS = [
    "Build-Rapport", "Callout-Fairness", "Describe-Need", "Discover-Preference",
    "No-Intention", "No-Need", "Promote-Coordination", "Show-Empathy", "Undermine-Requirements",
]

# canonical lowercase → canonical form (handles any model casing/spacing variation)
_ITEM_NORM = {
    "food": "Food", "water": "Water", "firewood": "Firewood",
    "not given": "Not Given", "none": "None",
}
_INTENT_NORM = {lbl.lower(): lbl for lbl in INTENT_LABELS}


def norm_item(s):
    """Normalize a model-output item value: strip, lowercase-lookup, else title-case."""
    if not isinstance(s, str):
        return ""
    return _ITEM_NORM.get(s.strip().lower(), s.strip().title())


def norm_intent(s):
    """Normalize a model-output intent label: strip, lowercase-lookup, else as-is."""
    if not isinstance(s, str):
        return ""
    return _INTENT_NORM.get(s.strip().lower(), s.strip())


# ── prompt builders ───────────────────────────────────────────────────────────

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


# ── API call ──────────────────────────────────────────────────────────────────

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
                print(f"Empty response (attempt {attempt+1}/{max_retries}), retrying...")
                continue
            return text
        except Exception as e:
            err = str(e)
            print(f"API error (attempt {attempt+1}/{max_retries}): {e}")
            if "insufficient_quota" in err:
                raise SystemExit("OpenAI quota exhausted — top up billing and retry.")
            if "requests per day" in err:
                print("RPD limit hit — stopping.")
                return None
            wait = 5.0
            m = re.search(r"try again in ([\d.]+)(ms|s)", err)
            if m:
                wait = float(m.group(1)) / 1000 if m.group(2) == "ms" else float(m.group(1))
                wait = max(wait + 1, 1)
            print(f"Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return None


def call_and_parse(messages, model, max_parse_retries=3):
    """Call API and parse JSON response; retries the full call if parse fails or response is empty."""
    last_raw = None
    for attempt in range(max_parse_retries):
        raw = call_api(messages, model)
        if raw is None:
            return None, None  # hard stop (quota exhausted or RPD limit)
        last_raw = raw
        parsed = parse_json(raw)
        if parsed is not None:
            return raw, parsed
        print(f"JSON parse failed (attempt {attempt+1}/{max_parse_retries}), retrying call...")
    print("Max parse retries reached — recording as unparseable (score 0).")
    return last_raw, None


# ── scoring helpers ───────────────────────────────────────────────────────────

def _pred_item(pred, key):
    """Get a normalized item value from pred, checking both lower and title-case keys."""
    val = pred.get(key) or pred.get(key.title()) or ""
    return norm_item(val)


def desire_em(pred, gold_desire):
    """Exact match: all 3 priority slots must be correct simultaneously."""
    if not pred:
        return 0
    return int(
        _pred_item(pred, "high")   == gold_desire.get("High", "") and
        _pred_item(pred, "medium") == gold_desire.get("Medium", "") and
        _pred_item(pred, "low")    == gold_desire.get("Low", "")
    )


def belief_em(pred, gold_high, gold_med, gold_low):
    """Exact match for belief slots (may include 'Not Given')."""
    if not pred:
        return 0
    return int(
        _pred_item(pred, "high")   == gold_high and
        _pred_item(pred, "medium") == gold_med and
        _pred_item(pred, "low")    == gold_low
    )


def intent_bitmask(intent_str):
    """Bitmask for a gold intent string like 'Build-Rapport,Describe-Need'."""
    labels = {norm_intent(l) for l in intent_str.split(",") if l.strip()}
    return [1 if lbl in labels else 0 for lbl in INTENT_LABELS]


def pred_intent_bitmask(pred_intents):
    """Bitmask for model-output intent list; handles casing/spacing variants."""
    normalized = {norm_intent(i) for i in (pred_intents or [])}
    return [1 if lbl in normalized else 0 for lbl in INTENT_LABELS]


# ── evaluate + save ───────────────────────────────────────────────────────────

def evaluate_desire(results, model_name, shard_tag):
    df = pd.DataFrame(results)
    overall = df["desire_em"].mean()
    out_dir = "results/desire"
    df.to_csv(f"{out_dir}/{model_name}{shard_tag}.csv", index=False)
    pd.DataFrame([{"metric": "Desire_EM", "score": overall}]).to_csv(
        f"{out_dir}/{model_name}{shard_tag}_overall.csv", index=False
    )
    print(f"\nDesire EM: {overall:.4f}  ({df['desire_em'].sum()}/{len(df)})")


def evaluate_belief(results, model_name, shard_tag):
    df = pd.DataFrame(results)
    overall = df["belief_em"].mean()
    out_dir = "results/belief"
    df.to_csv(f"{out_dir}/{model_name}{shard_tag}.csv", index=False)
    pd.DataFrame([{"metric": "Belief_EM", "score": overall}]).to_csv(
        f"{out_dir}/{model_name}{shard_tag}_overall.csv", index=False
    )
    print(f"\nBelief EM: {overall:.4f}  ({df['belief_em'].sum()}/{len(df)})")


def evaluate_intention(results, model_name, shard_tag):
    df = pd.DataFrame(results)
    golds = list(df["gold_bitmask"])
    preds = list(df["pred_bitmask"])
    micro = f1_score(golds, preds, average="micro", zero_division=0)
    macro = f1_score(golds, preds, average="macro", zero_division=0)
    out_dir = "results/intention"
    df.to_csv(f"{out_dir}/{model_name}{shard_tag}.csv", index=False)
    pd.DataFrame([
        {"metric": "Intent_Micro_F1", "score": micro},
        {"metric": "Intent_Macro_F1", "score": macro},
    ]).to_csv(f"{out_dir}/{model_name}{shard_tag}_overall.csv", index=False)
    print(f"\nIntent Micro F1: {micro:.4f} | Macro F1: {macro:.4f}")


# ── task runners ──────────────────────────────────────────────────────────────

def shard_slice(flat, shard, total):
    size = (len(flat) + total - 1) // total
    return flat[shard * size: min((shard + 1) * size, len(flat))]


def load_checkpoint(path):
    done, results = set(), []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add(r["uid"])
                results.append(r)
    return done, results


def save_checkpoint(path, results):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")


def run_desire(data, model, shard, total_shards, save_every):
    model_name = model.split("/")[-1].replace(".", "_")
    shard_tag = f"_shard{shard}of{total_shards}" if total_shards > 1 else ""
    os.makedirs("results/desire", exist_ok=True)
    out_path = f"results/desire/{model_name}{shard_tag}.jsonl"

    flat = []
    for s in data:
        flat.append({"dialogue_id": s["dialogue_id"], "dialogue": s["dialogue"],
                     "agent": "agent_1", "gold_desire": s["agent1_desire"]})
        flat.append({"dialogue_id": s["dialogue_id"], "dialogue": s["dialogue"],
                     "agent": "agent_2", "gold_desire": s["agent2_desire"]})
    flat = shard_slice(flat, shard, total_shards)

    done, results = load_checkpoint(out_path)
    if done:
        print(f"[desire shard {shard}] Resuming — {len(done)} done, {len(flat)-len(done)} remaining")
    else:
        print(f"[desire shard {shard}] Processing {len(flat)} items")

    for item in flat:
        uid = f"{item['dialogue_id']}_{item['agent']}_desire"
        if uid in done:
            continue
        raw, pred = call_and_parse(build_desire_messages(item["dialogue"], item["agent"]), model)
        results.append({
            "uid": uid,
            "dialogue_id": item["dialogue_id"],
            "agent": item["agent"],
            "gold_desire": item["gold_desire"],
            "pred": pred,
            "desire_em": desire_em(pred, item["gold_desire"]),
            "raw_response": raw or "",
        })
        done.add(uid)
        if len(results) % save_every == 0:
            save_checkpoint(out_path, results)
            print(f"[desire] {len(results)}/{len(flat)} checkpoint saved")

    save_checkpoint(out_path, results)
    print(f"[desire shard {shard}] Done → {out_path}")
    evaluate_desire(results, model_name, shard_tag)


def run_belief(data, model, shard, total_shards, save_every):
    model_name = model.split("/")[-1].replace(".", "_")
    shard_tag = f"_shard{shard}of{total_shards}" if total_shards > 1 else ""
    os.makedirs("results/belief", exist_ok=True)
    out_path = f"results/belief/{model_name}{shard_tag}.jsonl"

    flat = []
    for s in data:
        flat.append({"dialogue_id": s["dialogue_id"], "dialogue": s["dialogue"],
                     "agent": "agent_1", "opponent": "agent_2",
                     "gold_high": s["agent1_belief_high"],
                     "gold_med": s["agent1_belief_medium"],
                     "gold_low": s["agent1_belief_low"]})
        flat.append({"dialogue_id": s["dialogue_id"], "dialogue": s["dialogue"],
                     "agent": "agent_2", "opponent": "agent_1",
                     "gold_high": s["agent2_belief_high"],
                     "gold_med": s["agent2_belief_medium"],
                     "gold_low": s["agent2_belief_low"]})
    flat = shard_slice(flat, shard, total_shards)

    done, results = load_checkpoint(out_path)
    if done:
        print(f"[belief shard {shard}] Resuming — {len(done)} done, {len(flat)-len(done)} remaining")
    else:
        print(f"[belief shard {shard}] Processing {len(flat)} items")

    for item in flat:
        uid = f"{item['dialogue_id']}_{item['agent']}_belief"
        if uid in done:
            continue
        raw, pred = call_and_parse(build_belief_messages(item["dialogue"], item["agent"], item["opponent"]), model)
        results.append({
            "uid": uid,
            "dialogue_id": item["dialogue_id"],
            "agent": item["agent"],
            "opponent": item["opponent"],
            "gold_high": item["gold_high"],
            "gold_med": item["gold_med"],
            "gold_low": item["gold_low"],
            "pred": pred,
            "belief_em": belief_em(pred, item["gold_high"], item["gold_med"], item["gold_low"]),
            "raw_response": raw or "",
        })
        done.add(uid)
        if len(results) % save_every == 0:
            save_checkpoint(out_path, results)
            print(f"[belief] {len(results)}/{len(flat)} checkpoint saved")

    save_checkpoint(out_path, results)
    print(f"[belief shard {shard}] Done → {out_path}")
    evaluate_belief(results, model_name, shard_tag)


def run_intention(data, model, shard, total_shards, save_every):
    model_name = model.split("/")[-1].replace(".", "_")
    shard_tag = f"_shard{shard}of{total_shards}" if total_shards > 1 else ""
    os.makedirs("results/intention", exist_ok=True)
    out_path = f"results/intention/{model_name}{shard_tag}.jsonl"

    flat = []
    for s in data:
        turns = s["dialogue"]
        if len(turns) >= 2:
            flat.append({"dialogue_id": s["dialogue_id"], "dialogue": turns,
                         "utt_idx": 1, "target": turns[-2], "gold_intent": s["utterance1_intent"]})
        flat.append({"dialogue_id": s["dialogue_id"], "dialogue": turns,
                     "utt_idx": 2, "target": turns[-1], "gold_intent": s["utterance2_intent"]})
    flat = shard_slice(flat, shard, total_shards)

    done, results = load_checkpoint(out_path)
    if done:
        print(f"[intention shard {shard}] Resuming — {len(done)} done, {len(flat)-len(done)} remaining")
    else:
        print(f"[intention shard {shard}] Processing {len(flat)} items")

    for item in flat:
        uid = f"{item['dialogue_id']}_utt{item['utt_idx']}_intention"
        if uid in done:
            continue
        raw, pred = call_and_parse(build_intention_messages(item["dialogue"], item["target"]), model)
        pred_intents = (pred or {}).get("intents", [])
        results.append({
            "uid": uid,
            "dialogue_id": item["dialogue_id"],
            "utt_idx": item["utt_idx"],
            "target_utterance": item["target"],
            "gold_intent": item["gold_intent"],
            "gold_bitmask": intent_bitmask(item["gold_intent"]),
            "pred_intents": pred_intents,
            "pred_bitmask": pred_intent_bitmask(pred_intents),
            "raw_response": raw or "",
        })
        done.add(uid)
        if len(results) % save_every == 0:
            save_checkpoint(out_path, results)
            print(f"[intention] {len(results)}/{len(flat)} checkpoint saved")

    save_checkpoint(out_path, results)
    print(f"[intention shard {shard}] Done → {out_path}")
    evaluate_intention(results, model_name, shard_tag)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--task", default="all", choices=["desire", "belief", "intention", "all"])
    parser.add_argument("--data", default="../NegotiationToM.json",
                        help="Path to the extracted NegotiationToM.json")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)

    tasks = ["desire", "belief", "intention"] if args.task == "all" else [args.task]
    for task in tasks:
        if task == "desire":
            run_desire(data, args.model, args.shard, args.total_shards, args.save_every)
        elif task == "belief":
            run_belief(data, args.model, args.shard, args.total_shards, args.save_every)
        elif task == "intention":
            run_intention(data, args.model, args.shard, args.total_shards, args.save_every)
