"""EmoBench runner — gemini-3.5-flash-lite via OpenRouter (OpenAI-compatible).

The same model as EMO_Gemini_Flash3.5lite_Google, reached a different way. Everything that decides a
score is identical to that runner and to EMO_Gemini_Flash2.5, so the three stay comparable. What
differs here is only the transport:

  * OpenAI client against https://openrouter.ai/api/v1, so the system prompt is a system *message*
    rather than google-genai's system_instruction. Same text either way.
  * Hidden thinking is capped unconditionally. Visible reasoning is not decided here: it is read
    from EmoBench's own README at start-up (reasoning_visibility.resolve), which today answers
    False. --use-cot / --no-use-cot override it, and how it was decided is written onto every row.
  * max_tokens caps visible output. Unlike the native route it does not bound thinking, so the two
    runners bound cost differently and their bills are not directly comparable even though their
    scores are.
  * OpenRouter routes between Google AI Studio and Google Vertex (US) with failover, so the serving
    path can change mid-run. The provider that answered is recorded per item.

temperature is not set, matching the Google-route runner and following Google's 3.x guidance to
remove it rather than tune it. The 2.5 runner uses 0.6, which is a real difference between the old
numbers and these.

Nothing in the prompt is invented here — see the sibling runner's header.
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

# ROOT is the benchmark folder — the one holding upstream's README and the resolver that
# reads it. Nothing in this file decides whether reasoning is shown.
sys.path.insert(0, ROOT)
import reasoning_visibility  # noqa: E402 — needs ROOT on the path first

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("OPENROUTER_API_KEY is not set in EmoBench/.env")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=300)
LETTERS = string.ascii_uppercase

REASONING_EFFORT = "minimal"   # API-side ceiling; --reasoning-effort or --reasoning-max-tokens

AUTH_MARKERS = ("No auth credentials", "invalid_api_key", "Unauthorized", "401")


# ── helpers — identical to the 2.5 runner ─────────────────────────────────────

def load_yaml(rel_path):
    with open(os.path.join(ROOT, rel_path)) as f:
        return yaml.safe_load(f)


def rank_choices(choices):
    return "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))


def build_system_prompt(task, use_cot):
    """Upstream's own construction, both branches — src/utils.py::get_response_format.

    With CoT the statement changes and a "reasoning" key is prepended to the JSON conditions, so the
    model returns its reasoning as data rather than as prose the parser has to survive.
    """
    prompts = load_yaml("src/configs/prompts.yaml")
    response = load_yaml("src/configs/response.yaml")
    statement = response["cot"]["en"] if use_cot else response["base"]["en"]
    conditions = response[task]["en"]
    if use_cot:
        conditions = response["reasoning"]["en"] + ",\n" + conditions
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

def call_api(sys_prompt, user_prompt, model, max_tokens, reasoning_cfg, max_retries=3):
    """Returns (text, finish_reason, reasoning_tokens, served_by, reasoning_text, usage)."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
                extra_body={"reasoning": reasoning_cfg},
            )

            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            # OpenRouter returns the reasoning as readable text in its own field, so unlike the
            # native route it never contaminates the answer — but it is only there if asked for.
            thought = getattr(choice.message, "reasoning", None) or ""
            finish = str(choice.finish_reason or "")
            served = getattr(resp, "provider", "") or ""

            reasoning = None
            # Metered per call, not derived from a price list. The first run of this runner could
            # not be costed at all: it recorded reasoning tokens, which are neither prompt nor
            # completion tokens, so there was nothing to multiply a price by. OpenRouter also
            # returns the actual dollar cost of the call, which was being discarded.
            usage = {"prompt_tokens": None, "output_tokens": None, "thinking_tokens": None,
                     "call_cost": None}
            try:
                u = resp.usage
                usage["prompt_tokens"] = u.prompt_tokens
                usage["output_tokens"] = u.completion_tokens
                usage["call_cost"] = getattr(u, "cost", None)
                try:
                    reasoning = u.completion_tokens_details.reasoning_tokens
                except Exception:
                    pass
                # 0 rather than None when nothing was thought, so the column stays summable. The
                # native runner writes None here and the two will not aggregate together.
                usage["thinking_tokens"] = reasoning if reasoning is not None else 0
            except Exception:
                pass

            if not text:
                # Billed with nothing to score. finish_reason distinguishes "hit the cap" from
                # "the provider returned nothing", and those need different fixes.
                print(f"    empty response, finish_reason={finish}, provider={served}, "
                      f"reasoning_tokens={reasoning}", flush=True)
                return "", finish, reasoning, served, thought, usage

            time.sleep(2.0)
            return text, finish, reasoning, served, thought, usage

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
    return None, "", None, "", "", {"prompt_tokens": None, "output_tokens": None,
                                    "thinking_tokens": None, "call_cost": None}


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

def run_task(task, model, save_every, max_tokens, reasoning_cfg, use_cot, cot_source,
             limit=0):
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

    sys_prompt = build_system_prompt(task, use_cot)
    print(f"[{task}-en] reasoning={reasoning_cfg}  use_cot={use_cot}")

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
        raw, finish, thinking, served, thought, usage = call_api(
            sys_prompt, user_prompt, model, max_tokens, reasoning_cfg)
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
            "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "call_cost": usage.get("call_cost"),
            # The model's own reasoning, as BBH keeps its visible chain.
            "reasoning": thought,
            # The two ceilings are part of the condition, so they travel with the row.
            "reasoning_cfg": json.dumps(reasoning_cfg),
            "use_cot": use_cot,
            "use_cot_source": cot_source,
            # Visible reasoning, returned as a JSON field in CoT mode. Distinct from the hidden
            # thought summary above: one is what the model showed, the other what it was billed for.
            "reasoning_visible": (parsed or {}).get("reasoning", ""),
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
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N items per task; 0 means all")
    # Visible output only on this route — thinking is not bounded by it. Upstream uses 50 for
    # non-CoT and 2048 for CoT (src/model.py:75); 2048 covers both without risking truncation.
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--reasoning-effort", type=str, default=REASONING_EFFORT,
                        choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--reasoning-max-tokens", type=int, default=0,
                        help="use a numeric reasoning budget instead of an effort level")
    # Deliberately no default. Rule 3 says the benchmark decides whether reasoning is shown, so
    # the value is read from EmoBench's own README at run time rather than frozen here — a
    # literal True or False would be this project overruling the benchmark, and the next runner
    # copied from this one would inherit the wrong answer silently. Either flag overrides, and
    # whichever way it was decided is written onto every row.
    parser.add_argument("--use-cot", dest="use_cot", action="store_true", default=None,
                        help="force upstream's opt-in visible-reasoning branch (src/main.py:32)")
    parser.add_argument("--no-use-cot", dest="use_cot", action="store_false",
                        help="force the non-CoT branch")
    args = parser.parse_args()

    # What the benchmark says, or what the operator overrode it with — recorded either way.
    if args.use_cot is None:
        args.use_cot, cot_source = reasoning_visibility.resolve(ROOT)
        cot_source = "README " + cot_source
    else:
        cot_source = "--use-cot" if args.use_cot else "--no-use-cot"
    print(f"visible reasoning: use_cot={args.use_cot}  ({cot_source})")

    reasoning_cfg = ({"max_tokens": args.reasoning_max_tokens} if args.reasoning_max_tokens
                     else {"effort": args.reasoning_effort})

    tasks = ["EU", "EA"] if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, args.model, args.save_every, args.max_tokens,
                 reasoning_cfg, args.use_cot, cot_source, args.limit)
