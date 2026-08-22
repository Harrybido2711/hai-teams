"""EmoBench runner — gemini-3.5-flash-lite via OpenRouter (OpenAI-compatible).

The same model as EMO_Gemini_Flash3.5lite_Google, reached a different way. Everything that decides a
score is identical to that runner and to EMO_Gemini_Flash2.5, so the three stay comparable. What
differs here is only the transport:

  * OpenAI client against https://openrouter.ai/api/v1, so the system prompt is a system *message*
    rather than google-genai's system_instruction. Same text either way.
  * The thinking cap is OpenRouter's unified parameter, reasoning={"effort": "minimal"}, which maps
    to Gemini's thinking level. exclude:true is NOT used — excluded reasoning is still billed as
    output, so it hides the trace without saving anything.
  * max_tokens caps visible output. Unlike the native route it does not bound thinking, so the two
    runners bound cost differently and their bills are not directly comparable even though their
    scores are.
  * OpenRouter routes between Google AI Studio and Google Vertex (US) with failover, so the serving
    path can change mid-run. The provider that answered is recorded per item.

temperature is not set, matching the Google-route runner and following Google's 3.x guidance to
remove it rather than tune it. The 2.5 runner uses 0.6, which is a real difference between the old
numbers and these.

The prompt is NOT modified — see the sibling runner's header for why EmoBench does not take the
show-your-reasoning rule.
"""
import argparse
import json
import os
import re
import string
import sys
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("OPENROUTER_API_KEY is not set in EmoBench/.env")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=300)
LETTERS = string.ascii_uppercase

REASONING = {"effort": "minimal"}   # maps to Gemini's thinking level; "none" is not offered here
AUTH_MARKERS = ("No auth credentials", "invalid_api_key", "Unauthorized", "401")


# ── helpers — identical to the 2.5 runner ─────────────────────────────────────

def load_yaml(rel_path):
    with open(os.path.join(ROOT, rel_path)) as f:
        return yaml.safe_load(f)


def rank_choices(choices):
    return "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))


def build_system_prompt(task):
    prompts = load_yaml("src/configs/prompts.yaml")
    response = load_yaml("src/configs/response.yaml")
    statement = response["base"]["en"]
    conditions = response[task]["en"]
    fmt = f"\n{statement}\n```json\n    {{\n    {conditions}\n    }}\n```"
    return prompts["sys"]["en"] + fmt


def build_user_prompt(task, sample):
    template = load_yaml("src/configs/prompts.yaml")[task]["en"]
    if task == "EU":
        return template.format(
            scenario=sample["scenario"],
            subject=sample["subject"],
            emo_choices=rank_choices(sample["emotion_choices"]),
            cause_choices=rank_choices(sample["cause_choices"]),
        )
    else:
        return template.format(
            scenario=sample["scenario"],
            subject=sample["subject"],
            choices=rank_choices(sample["choices"]),
            q_type=sample["question type"],
        )


def parse_json(text):
    try:
        if "```json" in text:
            text = re.search(r"```json\s*([\s\S]*?)```", text).group(1)
        return json.loads(text.strip())
    except Exception:
        return None


# ── API call ──────────────────────────────────────────────────────────────────

def call_api(sys_prompt, user_prompt, model, max_tokens, max_retries=3):
    """Returns (text, finish_reason, reasoning_tokens, served_by)."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
                extra_body={"reasoning": REASONING},
            )

            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            finish = str(choice.finish_reason or "")
            served = getattr(resp, "provider", "") or ""

            reasoning = None
            try:
                reasoning = resp.usage.completion_tokens_details.reasoning_tokens
            except Exception:
                pass

            if not text:
                # Billed with nothing to score. finish_reason distinguishes "hit the cap" from
                # "the provider returned nothing", and those need different fixes.
                print(f"    empty response, finish_reason={finish}, provider={served}, "
                      f"reasoning_tokens={reasoning}", flush=True)
                return "", finish, reasoning, served

            time.sleep(2.0)
            return text, finish, reasoning, served

        except Exception as e:
            err = str(e)
            if any(m in err for m in AUTH_MARKERS):
                sys.exit(f"Authentication failed and will not be retried: {e}")
            print(f"API error (attempt {attempt+1}/{max_retries}): {e}")
            wait = 5.0
            m = re.search(r'try again in ([\d.]+)(ms|s)', err)
            if m:
                wait = float(m.group(1)) / 1000 if m.group(2) == "ms" else float(m.group(1))
                wait = max(wait + 1, 1)
            print(f"Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return None, "", None, ""


# ── evaluation + CSV output — identical to the 2.5 runner ─────────────────────

def evaluate(results, task, model_name):
    if not results:
        return

    df = pd.DataFrame(results)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", task)

    for col in ["answer", "emo_answer", "cause_answer", "label", "emo_label", "cause_label"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    if task == "EA":
        df["score"] = (df["label"] == df["answer"]).astype(int)
        cat_acc = df.groupby("category")["score"].mean()
        overall = df["score"].mean()
    else:
        df["score"] = ((df["emo_label"] == df["emo_answer"]) & (df["cause_label"] == df["cause_answer"])).astype(int)
        cat_acc = df.groupby("coarse_category")["score"].mean()
        overall = df["score"].mean()

    csv_path = os.path.join(out_dir, f"{model_name}_en.csv")
    df.to_csv(csv_path, index=False)

    overall_path = os.path.join(out_dir, f"{model_name}_en_overall.csv")
    rows = [{"category": cat, "accuracy": acc} for cat, acc in cat_acc.items()]
    rows.append({"category": "Overall", "accuracy": overall})
    pd.DataFrame(rows).to_csv(overall_path, index=False)

    empty = int((df["model_response"].astype(str).str.strip() == "").sum())
    unparsed = int((df["model_response"].astype(str).str.strip() != "").sum()
                   - df["model_response"].astype(str).str.strip().apply(
                       lambda t: bool(parse_json(t))).sum())
    thinking = df["thinking_tokens"].dropna().sum() if "thinking_tokens" in df.columns else 0
    providers = df["served_by"].value_counts().to_dict() if "served_by" in df.columns else {}

    print(f"\n{'='*50}")
    print(f"[{task}-en] Evaluation for {model_name}")
    print(f"  Overall accuracy: {overall:.4f} ({df['score'].sum()}/{len(df)})")
    for cat, acc in cat_acc.items():
        print(f"  {cat}: {acc:.4f}")
    print(f"  Empty responses: {empty}   Unparseable JSON: {unparsed}")
    print(f"  Reasoning tokens billed: {int(thinking)}")
    print(f"  Served by: {providers}")
    print(f"  Saved → {csv_path}")
    print(f"  Saved → {overall_path}")
    print(f"{'='*50}\n")


# ── main loop ─────────────────────────────────────────────────────────────────

def run_task(task, model, save_every, max_tokens):
    data = []
    data_path = os.path.join(ROOT, "data", f"{task}.jsonl")
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["language"] == "en":
                data.append(s)

    model_name = model.replace(".", "_").replace("/", "-")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", task)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{model_name}_en.jsonl")

    done_ids, results = set(), []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(str(r["qid"]))
                results.append(r)
        print(f"[{task}-en] Resuming — {len(done_ids)} done, {len(data)-len(done_ids)} remaining")
    else:
        print(f"[{task}-en] Processing {len(data)} samples")

    sys_prompt = build_system_prompt(task)

    def save_checkpoint():
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")

    for sample in data:
        qid = str(sample["qid"])
        if qid in done_ids:
            continue

        user_prompt = build_user_prompt(task, sample)
        raw, finish, thinking, served = call_api(sys_prompt, user_prompt, model, max_tokens)
        parsed = parse_json(raw) if raw else None

        common = {
            "qid": sample["qid"],
            "lang": "en",
            "scenario": sample["scenario"],
            "subject": sample["subject"],
            "model_response": raw or "",
            "finish_reason": finish,
            "thinking_tokens": thinking,
            "served_by": served,
        }
        if task == "EU":
            res = {
                **common,
                "coarse_category": sample["coarse_category"],
                "finegrained_category": sample["finegrained_category"],
                "emo_label": LETTERS[sample["emotion_choices"].index(sample["emotion_label"])],
                "emo_answer": (parsed or {}).get("answer_q1", ""),
                "cause_label": LETTERS[sample["cause_choices"].index(sample["cause_label"])],
                "cause_answer": (parsed or {}).get("answer_q2", ""),
            }
        else:
            res = {
                **common,
                "category": sample["category"],
                "question_type": sample["question type"],
                "label": LETTERS[sample["choices"].index(sample["label"])],
                "answer": (parsed or {}).get("answer", ""),
            }

        results.append(res)
        done_ids.add(qid)

        if len(results) % save_every == 0:
            save_checkpoint()
            print(f"[{task}-en] [{len(results)}/{len(data)}] checkpoint saved")

    save_checkpoint()
    print(f"[{task}-en] Done → {out_path}")
    evaluate(results, task, model_name)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemini-3.5-flash-lite")
    parser.add_argument("--task", type=str, default="all", choices=["EU", "EA", "all"])
    parser.add_argument("--save-every", type=int, default=20)
    # Visible output only on this route — thinking is not bounded by it.
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    tasks = ["EU", "EA"] if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, args.model, args.save_every, args.max_tokens)
