"""EmoBench runner — gemini-3.5-flash-lite via Google AI Studio (native google-genai).

Modelled on EMO_Gemini_Flash2.5/gemini_emo_eval.py. Everything that decides a score — the prompts
built from src/configs, the JSON parse, the record shape, the CSV outputs, the resume-by-qid
checkpoint — is deliberately identical, so this run stays comparable with the five providers already
finished. Four things differ, and each is a decision rather than an accident:

  1. thinking_level="minimal". Flash-Lite's default is already minimal AND minimal is its floor, so
     this buys no reduction; it pins the value against a provider-side default that can move. There
     is no way to turn thinking off on this model.
  2. temperature is NOT set. The 2.5 runner uses 0.6; Google's 3.x guidance is to remove temperature,
     top_p and top_k rather than tune them. This is a real difference from the 2.5 numbers and must
     be stated wherever the two are compared.
  3. max_output_tokens is set, and it INCLUDES thinking tokens. Too low and the model is billed for
     thinking and returns nothing — see the finish_reason handling below.
  4. Auth failures are fatal, not retried. This key is an AQ. auth key, and those are reported to
     401 against generativelanguage.googleapis.com with ACCESS_TOKEN_TYPE_UNSUPPORTED. Retrying an
     auth failure 3x per item for 800 items wastes an hour to learn one fact.

The prompt is NOT modified. EmoBench's own contract says "Do not provide any additional information
or explanations", which is the opposite of the show-your-reasoning rule that applies to BBH; the
upstream cot variant exists for that condition and switching to it would be a different experiment.
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
from google import genai
from google.genai import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

# Its own key, not GEMINI_API_KEY: that one is the 2.5 run's and is a different quota.
api_key = os.getenv("GEMINI_FLASH_LITE_API_KEY")
if not api_key:
    sys.exit("GEMINI_FLASH_LITE_API_KEY is not set in EmoBench/.env")
client = genai.Client(api_key=api_key)
LETTERS = string.ascii_uppercase

THINKING_LEVEL = "minimal"          # the floor for Flash-Lite; there is no off
AUTH_MARKERS = ("API key not valid", "ACCESS_TOKEN_TYPE_UNSUPPORTED", "PERMISSION_DENIED",
                "API_KEY_INVALID", "UNAUTHENTICATED")


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

def _thinking_config():
    """thinking_level moved from a numeric budget to a string level in 3.x.

    Older google-genai builds have ThinkingConfig without the field, and passing an unknown kwarg
    raises rather than being ignored. Failing loudly here is deliberate: silently dropping the cap
    would let the run proceed at whatever the model chooses to think, which is the thing the cap
    exists to prevent.
    """
    try:
        return types.ThinkingConfig(thinking_level=THINKING_LEVEL)
    except TypeError as e:
        sys.exit(
            f"This google-genai build does not accept thinking_level ({e}). Upgrade the SDK, or "
            f"decide explicitly to run without a thinking cap and record that decision — do not "
            f"remove this guard."
        )


def call_api(sys_prompt, user_prompt, model, max_output_tokens, max_retries=3):
    """Returns (text, finish_reason, reasoning_tokens). text is None when every attempt failed."""
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    thinking_config=_thinking_config(),
                    max_output_tokens=max_output_tokens,
                ),
            )

            finish, reasoning = "", None
            try:
                finish = str(resp.candidates[0].finish_reason or "")
            except Exception:
                pass
            try:
                reasoning = resp.usage_metadata.thoughts_token_count
            except Exception:
                pass

            text = (resp.text or "").strip() if getattr(resp, "text", None) else ""

            # Thinking consumed the whole budget: billed, and nothing to score. Distinguish it from
            # a wrong answer, because the fix is a larger cap rather than a better prompt.
            if not text and "MAX_TOKENS" in finish.upper():
                print(f"    empty response, finish_reason={finish}, "
                      f"thinking_tokens={reasoning} — raise --max-output-tokens", flush=True)
                return "", finish, reasoning

            time.sleep(2.0)
            return text, finish, reasoning

        except Exception as e:
            err = str(e)
            if any(m in err for m in AUTH_MARKERS):
                sys.exit(
                    f"Authentication failed and will not be retried: {e}\n"
                    f"AQ. auth keys are reported to fail against generativelanguage.googleapis.com. "
                    f"Probe this route before rerunning, or use the OpenRouter runner instead."
                )
            print(f"API error (attempt {attempt+1}/{max_retries}): {e}")
            wait = 5.0
            m = re.search(r'retry_delay.*?seconds:\s*([\d.]+)', err)
            if m:
                wait = float(m.group(1)) + 1
            else:
                m2 = re.search(r'try again in ([\d.]+)(ms|s)', err)
                if m2:
                    wait = float(m2.group(1)) / 1000 if m2.group(2) == "ms" else float(m2.group(1))
                    wait = max(wait + 1, 1)
            print(f"Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return None, "", None


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

    # Reported separately from the score, never folded into it: an unparseable or empty response is
    # a different failure from a wrong answer, and this project's rule is to count it as its own.
    empty = int((df["model_response"].astype(str).str.strip() == "").sum())
    unparsed = int((df["model_response"].astype(str).str.strip() != "").sum()
                   - df["model_response"].astype(str).str.strip().apply(
                       lambda t: bool(parse_json(t))).sum())
    thinking = df["thinking_tokens"].dropna().sum() if "thinking_tokens" in df.columns else 0

    print(f"\n{'='*50}")
    print(f"[{task}-en] Evaluation for {model_name}")
    print(f"  Overall accuracy: {overall:.4f} ({df['score'].sum()}/{len(df)})")
    for cat, acc in cat_acc.items():
        print(f"  {cat}: {acc:.4f}")
    print(f"  Empty responses: {empty}   Unparseable JSON: {unparsed}")
    print(f"  Thinking tokens billed: {int(thinking)}")
    print(f"  Saved → {csv_path}")
    print(f"  Saved → {overall_path}")
    print(f"{'='*50}\n")


# ── main loop ─────────────────────────────────────────────────────────────────

def run_task(task, model, save_every, max_output_tokens, limit=0):
    data = []
    data_path = os.path.join(ROOT, "data", f"{task}.jsonl")
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["language"] == "en":
                data.append(s)
    # Smoke-test lever. Truncates the work list, so the checkpoint it writes is a partial run of the
    # same config — resuming a real run on top of it is correct, but delete it first if the config
    # changed in between.
    if limit:
        data = data[:limit]
        print(f"[{task}-en] --limit {limit}: smoke test, not a run")

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
        raw, finish, thinking = call_api(sys_prompt, user_prompt, model, max_output_tokens)
        parsed = parse_json(raw) if raw else None

        common = {
            "qid": sample["qid"],
            "lang": "en",
            "scenario": sample["scenario"],
            "subject": sample["subject"],
            "model_response": raw or "",
            "finish_reason": finish,
            "thinking_tokens": thinking,
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
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite")
    parser.add_argument("--task", type=str, default="all", choices=["EU", "EA", "all"])
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N items per task; 0 means all")
    # Includes thinking tokens. 2048 is a starting point, not a measured value: the expected answer
    # is a few tokens of JSON, and the rest is headroom for minimal thinking. Watch the empty-response
    # count on the pilot before trusting it.
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    args = parser.parse_args()

    tasks = ["EU", "EA"] if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, args.model, args.save_every, args.max_output_tokens, args.limit)
