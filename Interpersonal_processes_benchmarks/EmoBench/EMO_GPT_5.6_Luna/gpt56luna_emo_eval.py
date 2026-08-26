"""EmoBench runner — gpt-5.6-luna via the OpenAI platform.

Built to the same shape as EMO_Gemini_Flash3.5lite_OpenRouter. Everything that decides a score —
the prompts from src/configs, the JSON parse, the record, the CSVs, the resume-by-qid checkpoint —
is deliberately identical, so this run is comparable with the providers already finished.

What is specific to this runner, and why:

  1. **Its parameter surface is negotiated once at startup, not assumed.** This model is newer than
     anything else called here and the accepted parameter names are not established. The previous
     runner in this project hardcoded a parameter its SDK did not accept, and because the error was
     caught by a generic retry handler it produced 20 empty rows over 11 minutes before anyone
     looked. negotiate() spends ONE call finding out which of max_completion_tokens/max_tokens,
     reasoning_effort and seed this model takes, prints the answer, and writes it onto every row.
  2. **A rejected parameter is fatal, not retried.** Retrying an invalid request 400 times to learn
     one fact is the failure above.
  3. **Hidden reasoning is bounded at reasoning_effort="low"** — median 40 tokens a row, against
     the model's own default of "medium". Rule 1 of references/model-parameters.md is unconditional,
     but it bounds what thinking may spend rather than switching it off: "none" measured 9-11
     accuracy points worse over 200 items (p<=0.003) to save two cents. "minimal" — the value the
     Gemini runners use — is refused outright here, which is why negotiate() distinguishes a refused
     value from a refused parameter instead of dropping the cap and running uncapped.
  4. **temperature is not set.** Consistent with the Gemini runners, where unset, 0.0 and 0.6 were
     measured identical on all 200 items, and with newer OpenAI reasoning models that reject any
     value but the default.
  5. **seed is pinned** where accepted — rule 6. Without one, 22.5% of EmoBench items changed
     between two runs of the same Gemini config, which is larger than any score gap we interpret.
  6. **Usage is recorded per call**: prompt, completion and reasoning tokens. Cost could not be
     compared between two routes here until the tokens were metered rather than derived.

Visible reasoning is not decided in this file. It is read from EmoBench's own README at start-up
(reasoning_visibility.resolve), which today answers False. --use-cot / --no-use-cot override it, and
how it was decided is written onto every row.
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

# ROOT is the benchmark folder — the one holding upstream's README and the resolver that reads it.
# Nothing in this file decides whether reasoning is shown.
sys.path.insert(0, ROOT)
import reasoning_visibility  # noqa: E402 — needs ROOT on the path first

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("OPENAI_API_KEY is not set in EmoBench/.env")
client = OpenAI(api_key=api_key, timeout=300)
LETTERS = string.ascii_uppercase

# Measured on this model, not copied from the Gemini runners. "minimal" is rejected outright; the
# accepted set is none/low/medium/high/xhigh (the docs also list "max", which this model refuses).
#
# "low", not "none". Over 200 EU items with the seed pinned: none 0.565, low 0.650, medium 0.675;
# none-vs-low p=0.0033 and none-vs-medium p=0.0003, both significant, while low-vs-medium p=0.332
# is not. Switching thinking off costs 9-11 accuracy points to save two cents. Rule 1 bounds what
# thinking may SPEND — "none" is removal, not a cap, and it was adopted here only by copying the
# Gemini shape onto a model where thinking actually earns its tokens.
REASONING_EFFORT = "low"
EFFORT_FALLBACKS = ("low", "medium", "none")   # preference order if a value is refused
RESULTS_TAG = ""               # set from --tag; keeps a sweep arm out of the baseline directory

AUTH_MARKERS = ("invalid_api_key", "Incorrect API key", "Unauthorized", "401",
                "insufficient_quota", "billing")
# Permanent request-shape failures. Retrying these three times per item across 400 items is how a
# run spends hours proving the same thing over and over.
FATAL_MARKERS = ("model_not_found", "does not exist", "invalid_request_error",
                 "Unsupported parameter", "Unrecognized request argument")


# ── helpers — byte-identical to the other EmoBench runners ────────────────────

def load_yaml(rel_path):
    with open(os.path.join(ROOT, rel_path)) as f:
        return yaml.safe_load(f)


def rank_choices(choices):
    return "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))


def build_system_prompt(task, use_cot):
    """Upstream's own construction, both branches — src/utils.py::get_response_format."""
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


def _results_dir(task):
    name = "results" + (("_" + RESULTS_TAG) if RESULTS_TAG else "")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name, task)


# ── capability negotiation ────────────────────────────────────────────────────

def negotiate(model, wanted):
    """Find which of `wanted` this model actually accepts, in one probe call.

    Chosen over hardcoding because the accepted names for this model are not established here, and
    over try/except-per-item because a rejected parameter is permanent: discovering it 400 times
    costs a run. OpenAI names the offending parameter in the error and often names its replacement
    ("Use 'max_completion_tokens' instead"), so a rejection is actionable rather than just fatal.

    Returns (params, notes). Raises if the failure is not about a parameter — an unknown model or a
    bad key is not something to negotiate around.
    """
    params, notes = dict(wanted), []
    for _ in range(len(wanted) + 2):
        try:
            client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Reply with: ok"}], **params)
            return params, notes
        except Exception as e:
            err = str(e)
            if any(m in err for m in AUTH_MARKERS):
                sys.exit(f"Authentication or quota failure, not retried: {e}")
            offender = next((k for k in params if f"'{k}'" in err), None)
            if offender is None:
                raise

            # A refused *value* is not a refused *parameter*, and conflating them is how a cap gets
            # dropped instead of corrected. gpt-5.6-luna refuses reasoning_effort="minimal" while
            # supporting the parameter perfectly well — dropping it there would have run the whole
            # benchmark at the model's default effort, uncapped, in breach of rule 1.
            if "Unsupported value" in err or "does not support" in err:
                supported = re.search(r"[Ss]upported values are:?\s*([^.}]+)", err)
                options = re.findall(r"'([^']+)'", supported.group(1)) if supported else []
                pick = next((v for v in EFFORT_FALLBACKS if v in options), options[0] if options else None)
                if pick is not None:
                    params[offender] = pick
                    notes.append(f"{offender}={pick} (asked {params.get(offender)!r} refused)")
                    print(f"  negotiate: {offender} refused that value; supported {options}, "
                          f"using {pick!r}", flush=True)
                    continue
                # no alternatives named — fall through and drop it, loudly
            value = params.pop(offender)
            rename = re.search(r"[Uu]se '([A-Za-z_]+)' instead", err)
            if rename:
                params[rename.group(1)] = value
                notes.append(f"{offender}→{rename.group(1)}")
                print(f"  negotiate: {offender} rejected, using {rename.group(1)}", flush=True)
            else:
                notes.append(f"{offender} unsupported")
                print(f"  negotiate: {offender} unsupported, dropped", flush=True)
    raise SystemExit(f"could not find a working parameter set for {model}; last tried {params}")


# ── API call ──────────────────────────────────────────────────────────────────

def call_api(sys_prompt, user_prompt, model, params, max_retries=3):
    """Returns (text, finish_reason, reasoning_tokens, reasoning_text, usage)."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user_prompt}],
                **params,
            )
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            thought = getattr(choice.message, "reasoning", None) or ""
            finish = str(choice.finish_reason or "")

            reasoning = None
            usage = {"prompt_tokens": None, "output_tokens": None, "thinking_tokens": None,
                     "call_cost": None}
            try:
                u = resp.usage
                usage["prompt_tokens"] = u.prompt_tokens
                usage["output_tokens"] = u.completion_tokens
                try:
                    reasoning = u.completion_tokens_details.reasoning_tokens
                except Exception:
                    pass
                # 0 rather than None when nothing was thought, so the column stays summable across
                # models; the native Gemini runner writes None and the two would not aggregate.
                usage["thinking_tokens"] = reasoning if reasoning is not None else 0
            except Exception:
                pass

            if not text:
                # Billed with nothing to score. finish_reason separates "hit the cap" from "returned
                # nothing", and those need different fixes.
                print(f"    empty response, finish_reason={finish}, "
                      f"reasoning_tokens={reasoning}", flush=True)
                return "", finish, reasoning, thought, usage

            time.sleep(2.0)
            return text, finish, reasoning, thought, usage

        except Exception as e:
            err = str(e)
            if any(m in err for m in AUTH_MARKERS):
                sys.exit(f"Authentication or quota failure, not retried: {e}")
            if any(m in err for m in FATAL_MARKERS):
                sys.exit(
                    f"Request rejected as invalid and will not be retried: {type(e).__name__}: {e}\n"
                    f"This is a configuration error, not a transient one. Parameters in use: {params}"
                )
            print(f"API error (attempt {attempt+1}/{max_retries}): {e}", flush=True)
            wait = 5.0
            m = re.search(r"try again in ([\d.]+)(ms|s)", err)
            if m:
                wait = float(m.group(1)) / 1000 if m.group(2) == "ms" else float(m.group(1))
                wait = max(wait + 1, 1)
            print(f"Retrying in {wait:.1f}s...", flush=True)
            time.sleep(wait)
    return None, "", None, "", {"prompt_tokens": None, "output_tokens": None,
                                "thinking_tokens": None, "call_cost": None}


# ── evaluation + CSV output — identical to the other runners ──────────────────

def evaluate(results, task, model_name):
    if not results:
        return

    df = pd.DataFrame(results)
    out_dir = _results_dir(task)

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

    print(f"\n{'='*50}")
    print(f"[{task}-en] Evaluation for {model_name}")
    print(f"  Overall accuracy: {overall:.4f} ({df['score'].sum()}/{len(df)})")
    for cat, acc in cat_acc.items():
        print(f"  {cat}: {acc:.4f}")
    print(f"  Empty responses: {empty}   Unparseable JSON: {unparsed}")
    print(f"  Reasoning tokens billed: {int(thinking)}")
    print(f"  Saved → {csv_path}")
    print(f"  Saved → {overall_path}")
    print(f"{'='*50}\n")


# ── main loop ─────────────────────────────────────────────────────────────────

def run_task(task, model, save_every, params, negotiated, use_cot, cot_source, limit=0):
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
    results_dir = _results_dir(task)
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
    print(f"[{task}-en] params={params}  use_cot={use_cot}", flush=True)

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
        raw, finish, thinking, thought, usage = call_api(sys_prompt, user_prompt, model, params)
        parsed = parse_json(raw) if raw else None

        common = {
            "qid": sample["qid"],
            "lang": "en",
            "scenario": sample["scenario"],
            "subject": sample["subject"],
            "model_response": raw or "",
            "finish_reason": finish,
            "thinking_tokens": usage.get("thinking_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "call_cost": usage.get("call_cost"),
            # The condition travels with the data: what was sent, and what the model would not take.
            "params": json.dumps(params),
            "negotiated": ";".join(negotiated) if negotiated else "",
            "seed": params.get("seed"),
            "temperature": params.get("temperature"),
            # The model's own reasoning, as BBH keeps its visible chain.
            "reasoning": thought,
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
            print(f"[{task}-en] [{len(results)}/{len(data)}] checkpoint saved", flush=True)

    save_checkpoint()
    print(f"[{task}-en] Done → {out_path}")
    evaluate(results, task, model_name)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-5.6-luna")
    parser.add_argument("--task", type=str, default="all", choices=["EU", "EA", "all"])
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N items per task; 0 means all")
    # Upstream caps non-CoT output at 50 (src/model.py:75). Not copied: on a reasoning model the
    # cap counts thinking, so thinking would exhaust it and return a billed empty response.
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--reasoning-effort", type=str, default=REASONING_EFFORT,
                        help="rule 1 cap. none/low/medium/high/xhigh on this model; the model's\n"
                             "own default is medium, which is NOT free")
    parser.add_argument("--seed", type=int, default=42,
                        help="makes the run reproducible; dropped automatically if unsupported")
    parser.add_argument("--tag", type=str, default="",
                        help="write to results_<tag>/ instead of results/, for sweeps")
    # Deliberately no default. Rule 3: the benchmark decides whether reasoning is shown, so the
    # value is read from EmoBench's own README at run time rather than frozen here.
    parser.add_argument("--use-cot", dest="use_cot", action="store_true", default=None,
                        help="force upstream's opt-in visible-reasoning branch (src/main.py:32)")
    parser.add_argument("--no-use-cot", dest="use_cot", action="store_false",
                        help="force the non-CoT branch")
    args = parser.parse_args()

    RESULTS_TAG = args.tag

    # What the benchmark says, or what the operator overrode it with — recorded either way.
    if args.use_cot is None:
        args.use_cot, cot_source = reasoning_visibility.resolve(ROOT)
        cot_source = "README " + cot_source
    else:
        cot_source = "--use-cot" if args.use_cot else "--no-use-cot"
    print(f"visible reasoning: use_cot={args.use_cot}  ({cot_source})", flush=True)

    # One call, before 400 items depend on the answer.
    print(f"negotiating parameters for {args.model}...", flush=True)
    wanted = {"max_completion_tokens": args.max_tokens,
              "reasoning_effort": args.reasoning_effort,
              "seed": args.seed}
    params, negotiated = negotiate(args.model, wanted)
    print(f"accepted: {params}", flush=True)
    if "reasoning_effort" not in params:
        print("WARNING: this model exposes no reasoning_effort. Rule 1 still applies — cap hidden "
              "reasoning with a prompt ceiling and record the wording with the score "
              "(references/prompt-ceiling.md).", flush=True)
    if "seed" not in params:
        print("WARNING: no seed accepted — this run is not reproducible (rule 6).", flush=True)

    tasks = ["EU", "EA"] if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, args.model, args.save_every, params, negotiated, args.use_cot, cot_source,
                 args.limit)
