"""Shared core for every DocVQA runner. Modelled on `Tasks_benchmarks/mmlu/mmlu_eval_core.py`.

**One scorer.** `score_response` and `anls_score` are the only scorers in this benchmark. They are
**copied verbatim** from the four runners that already produced results — `openai_eval.py`,
`gemini_eval.py`, `qwen_DocVQA/qwen_eval.py`, `gemma_DocVQA/gemma_eval_half*.py` — rather than
improved, because those four results stay valid only if a new model is judged by the same matcher
they were. Changing anything here means rescoring all six, not just scoring the new two.

**The prompt is byte-identical to those four runners too**, for the same reason: a different prompt
is a different condition, and DocVQA answers are verbatim spans, so wording moves the score.

**The task is defined on the document image.** The challenge's Task 1 page: "answer questions asked
on a document image", answers are "short text spans taken verbatim from the document". The OCR the
dataset ships is auxiliary — "participants are free to use any OCR" — and this project does not use
it (settled 2026-09-09; the reasoning is on the benchmark's page). So every runner here sends the
page image, and `load_image_bytes` is the only input path.

**ANLS takes the maximum over the answer list.** Each question carries several acceptable answers.
Scoring against one of them under-reports systematically.
"""

import base64
import csv
import json
import os
import re
import time

# Paths resolve from THIS FILE, never the cwd: a runner lives in <DocVQA>/DOCVQA_<Slot>/ while the
# data lives in <DocVQA>/docvqa_output/.
DOCVQA_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DOCVQA_ROOT, "docvqa_output", "docvqa_validation.json")


def _find_env():
    """This benchmark's own `.env` if it has one, else the nearest one above it.

    Quest keeps a `.env` per benchmark directory; the local tree keeps a single one at the repo
    root and DocVQA has none of its own. The old runners called `load_dotenv()` with no argument
    and got the same walk-up for free — this makes it explicit rather than dependent on the cwd
    the job happened to start in.
    """
    d = DOCVQA_ROOT
    while True:
        candidate = os.path.join(d, ".env")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.join(DOCVQA_ROOT, ".env")   # nothing found; report this path in the error
        d = parent


ENV_PATH = _find_env()

# Bumped whenever the matcher changes behaviour, and written onto every result row. A v1 number
# cannot be compared with a v2 one without rescoring — v2 added two branches on 2026-09-11, the
# history is on the benchmark's page.
SCORER_VERSION = "docvqa_lenient_v2"

# Byte-for-byte what the four finished runners sent. Do not tidy it.
PROMPT = (
    "You are reading a document image. Answer the question below using only "
    "information visible in the document.\n\n"
    "Question: {question}\n\n"
    "Give a short, direct answer — a word, number, or brief phrase. "
    'End your response with: "Final Answer: <your answer here>"'
)

AUTH_MARKERS = ("invalid_api_key", "incorrect api key", "unauthorized", "permission_denied",
                "api key not valid", "unauthenticated", "insufficient_quota")
EFFORT_FALLBACKS = ("low", "minimal", "medium")


def model_slug(model_id):
    return model_id.replace("/", "-").replace(".", "_")


def load_data():
    """The 5,349 validation questions. Each carries questionId, question, answers, image_path."""
    with open(DATA_PATH) as fh:
        return json.load(fh)


def build_prompt(example):
    return PROMPT.format(question=example["question"])


def load_image_bytes(image_path):
    """`image_path` in the data is repo-relative — './docvqa_output/images/49153.png'."""
    return open(os.path.join(DOCVQA_ROOT, image_path.lstrip("./")), "rb").read()


def load_image_b64(image_path):
    return base64.standard_b64encode(load_image_bytes(image_path)).decode("utf-8")


# ---------------------------------------------------------------- scoring


def extract_final_answer(model_output):
    if not isinstance(model_output, str):
        return ""
    m = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    return m.group(1).strip() if m else model_output.strip()


def normalize(text):
    return re.sub(r"\s+", " ", str(text).lower().strip())


def strip_punct(text):
    return re.sub(r"[^\w\s]", "", text)


def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def anls_score(prediction, gold_answers, threshold=0.5):
    """Average Normalized Levenshtein Similarity — DocVQA's official metric.

    The `max` over `gold_answers` is not a nicety: every question has several acceptable answers and
    the challenge's own metric takes the best of them. Scoring against one under-reports.
    The 0.5 threshold is the official one — below it the score is 0, not a small number.
    """
    pred = normalize(prediction)
    best = 0.0
    for g in gold_answers:
        g_norm = normalize(g)
        max_len = max(len(pred), len(g_norm))
        nl = 0.0 if max_len == 0 else levenshtein(pred, g_norm) / max_len
        sim = 0.0 if nl > threshold else 1.0 - nl
        best = max(best, sim)
    return best


def squash(text):
    """Every space and punctuation mark removed — the written form stripped to its characters.

    `"JAN 17 '68"` and `"jan17'68"` are the same answer written two ways, and so are `DR.W.J.DARBY`
    and `Dr. W. J. Darby`, `3. 28-2001` and `3-28-2001`. The existing branches normalise spacing OR
    punctuation; this normalises both at once, which is where those pairs fall through.
    """
    return re.sub(r"[^\w]", "", normalize(text))


def one_transposition(a, b, floor=8):
    """True when `a` and `b` differ by exactly one swap of two ADJACENT characters.

    An adjacent swap is the signature of a typing slip, and in this dataset it is usually the GOLD
    that has it: `Grocrey` for `Grocery`, `laways` for `always`, `Cigfil` for `Cigifl`. It is a
    different thing from a misreading — `Paul Saliman` for `Paul Saltman` is one substitution, not a
    swap, and is a wrong answer.

    **`floor` is not decoration.** Without it this branch credits `14` for `41`, which is a genuinely
    different number; measured over all 32,094 stored rows that was the single false gain it made,
    and 8 characters removes it while keeping all twelve real ones.
    """
    if len(a) != len(b) or a == b or len(a) < floor:
        return False
    d = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]


def score_response(model_output, gold_answers):
    """Lenient exact match. Branches are additive and tried in order, so one can only gain rows.

    Six branches. The first four are verbatim from the four standalone runners that produced this
    benchmark's earlier results; branches 5 and 6 were added on 2026-09-11 and each was measured
    over every stored row before being kept. **Every branch normalises how an answer is written.
    None of them tolerates an answer being slightly wrong** — see the page for the fuzzy branch that
    was measured and rejected for crediting `Paul Saliman` as `Paul Saltman`.
    """
    fa = extract_final_answer(model_output)
    if not fa:
        return 0
    if any(normalize(fa) == normalize(g) for g in gold_answers):
        return 1

    def tokenize(t):
        return normalize(t).replace(",", " ").split()

    if any(tokenize(fa) == tokenize(g) for g in gold_answers):
        return 1
    if any(normalize(strip_punct(fa)) == normalize(strip_punct(g)) for g in gold_answers):
        return 1
    fa_norm = normalize(fa)
    if any(fa_norm in normalize(g) or normalize(g) in fa_norm for g in gold_answers):
        return 1
    # 5 — spacing and punctuation removed together, +311 rows measured
    sq = squash(fa)
    if any(sq == squash(g) for g in gold_answers):
        return 1
    # 6 — one adjacent transposition, +12 rows measured, every one of them a typo in the gold
    if any(one_transposition(sq, squash(g)) for g in gold_answers):
        return 1
    return 0


# ---------------------------------------------------------------- output

FIELDS = ["questionId", "question", "gold_answers", "image_path", "model_response",
          "final_answer", "score", "anls", "config"]


def config_string(config):
    """The generation config, flattened onto every row — model-parameters.md rule 8. A config that
    lives only in a job script cannot be recovered from a result file later."""
    if not config:
        return ""
    return ";".join(f"{k}={v}" for k, v in sorted(config.items()))


def parse_config(text):
    out = {}
    for part in (text or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def results_dir(model_dir):
    d = os.path.join(model_dir, "results")
    os.makedirs(d, exist_ok=True)
    return d


def shard_tag(shard, total_shards):
    """`_shard2of5`, or empty when total_shards is 1 — the empty case keeps an unsharded run writing
    the filename it always did. Without the tag every shard overwrites the last (`.claude/INDEX.md`).
    """
    return f"_shard{shard}of{total_shards}" if total_shards > 1 else ""


def shard_slice(items, shard, total_shards):
    """Contiguous block split, so a shard's rows stay in one range and a missing shard shows up as
    a gap rather than scattered holes."""
    if total_shards <= 1:
        return list(items)
    size = (len(items) + total_shards - 1) // total_shards
    return list(items)[shard * size: min((shard + 1) * size, len(items))]


def retry(fn, tries=3, base_sleep=2.0, label="", fatal=()):
    """Retries on exception AND on an empty string, returning "" when every attempt fails.

    **`fatal` is the reason this is not a generic retry.** DocVQA's one recorded incident was a
    daily-request-cap refusal retried three times an item: for every question answered, six were
    skipped because the retries had spent the quota, leaving 3,021 of 5,349 rows empty
    (`OPENAI_EVAL_NOTES.md`). A cap refusal is not transient — matching one of `fatal` stops the run
    at once rather than burning what is left of the day.
    """
    for attempt in range(tries):
        try:
            out = fn()
            if isinstance(out, str) and out.strip():
                return out.strip()
        except Exception as e:
            err = str(e)
            for marker, why in fatal:
                if marker in err.lower():
                    raise SystemExit(f"[{label}] {why}\n  {err[:400]}")
            print(f"[{label}] attempt {attempt + 1}/{tries} failed: {err[:200]}", flush=True)
        if attempt < tries - 1:
            time.sleep(base_sleep * (2 ** attempt))
    # An empty string at HTTP 200 raises nothing, so without this line every attempt is silent and
    # the run's only trace of the failure is a blank cell. gemini-3.5-flash-lite returned exactly
    # that on 5 of 5,349 images on 2026-09-10 — finish_reason MALFORMED_RESPONSE, thought tokens
    # spent, no output part — and the logs held not one word about it.
    print(f"[{label}] gave up after {tries} attempt(s), writing an empty row", flush=True)
    return ""


def load_checkpoint(out_path, config, verbose=True):
    """Rows already done, keyed by questionId — or a refusal if the config changed.

    **An empty row is not a done row.** The 3,021 empty rows the quota incident left behind had to
    be re-asked, and a plain resume would have skipped them forever.
    """
    if not os.path.exists(out_path):
        return {}
    want = {k: str(v) for k, v in (config or {}).items()}
    done, empty = {}, 0
    for line in open(out_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        have = parse_config(r.get("config", ""))
        diff = {k: (v, have.get(k)) for k, v in want.items() if have.get(k) != v}
        if diff:
            raise SystemExit(
                f"{out_path}\n  refusing to resume: stored rows have a different config.\n"
                f"  differences (wanted, stored): {diff}\n"
                f"  Archive the results directory and start clean.")
        if str(r.get("model_response", "")).strip():
            done[str(r["questionId"])] = r
        else:
            empty += 1
    if verbose and (done or empty):
        print(f"  resume: {len(done)} done, {empty} empty row(s) will be retried", flush=True)
    return done


def write_results(model_dir, model_id, records, config=None, tag=""):
    d = results_dir(model_dir)
    slug = model_slug(model_id) + tag
    ordered = sorted(records, key=lambda r: int(r["questionId"]))
    with open(os.path.join(d, f"{slug}.jsonl"), "w") as fh:
        for r in ordered:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(d, f"{slug}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    n = len(ordered)
    acc = sum(r["score"] for r in ordered) / n if n else ""
    anls = sum(r["anls"] for r in ordered) / n if n else ""
    blank = sum(1 for r in ordered if not str(r["model_response"]).strip())
    no_marker = sum(1 for r in ordered
                    if not re.search(r"Final Answer:", str(r["model_response"]), re.IGNORECASE))
    with open(os.path.join(d, f"{slug}_overall.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n", "overall_accuracy", "anls", "no_marker", "empty_response",
                    "scorer", "config"])
        w.writerow([model_id, n, round(acc, 4) if n else "", round(anls, 4) if n else "",
                    no_marker, blank, SCORER_VERSION, config_string(config)])
    return {"model": model_id, "n": n, "overall_accuracy": round(acc, 4) if n else "",
            "anls": round(anls, 4) if n else "", "no_marker": no_marker, "empty_response": blank}


# ---------------------------------------------------------------- the loop


def run(model_dir, model_id, call, limit=0, config=None, save_every=20, resume=True,
        sleep_between=0.0, shard=0, total_shards=1, verbose=True):
    """`call(prompt, image_path)` returns the model's text, or "" when every attempt failed."""
    data = load_data()
    if limit:
        data = data[:limit]
        if verbose:
            print(f"--limit {limit}: smoke test, not a run", flush=True)
    config = dict(config or {})
    tag = shard_tag(shard, total_shards)
    out_path = os.path.join(results_dir(model_dir), f"{model_slug(model_id)}{tag}.jsonl")
    done = load_checkpoint(out_path, config, verbose) if (resume and not limit) else {}
    records = list(done.values())

    def flush(recs):
        with open(out_path, "w") as fh:
            for r in sorted(recs, key=lambda x: int(x["questionId"])):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    todo = [ex for ex in shard_slice(data, shard, total_shards)
            if str(ex["questionId"]) not in done]
    if verbose:
        print(f"shard {shard}/{total_shards}: {len(todo)} to ask, {len(done)} already done",
              flush=True)

    for k, ex in enumerate(todo, 1):
        gold = ex["answers"]
        if isinstance(gold, str):          # the JSON stores this as a stringified list
            gold = json.loads(gold.replace("'", '"'))
        try:
            resp = call(build_prompt(ex), ex["image_path"])
        except FileNotFoundError:
            print(f"[{ex['questionId']}] image missing: {ex['image_path']}", flush=True)
            resp = ""
        resp = resp if isinstance(resp, str) else ""
        records.append({
            "questionId": ex["questionId"], "question": ex["question"],
            "gold_answers": json.dumps(gold, ensure_ascii=False), "image_path": ex["image_path"],
            "model_response": resp, "final_answer": extract_final_answer(resp),
            "score": score_response(resp, gold), "anls": anls_score(extract_final_answer(resp), gold),
            "config": config_string(config)})
        if sleep_between:
            time.sleep(sleep_between)
        if save_every and k % save_every == 0:
            flush(records)
            if verbose:
                print(f"  {len(records)}/{len(todo) + len(done)}", flush=True)

    summary = write_results(model_dir, model_id, records, config=config, tag=tag)
    if verbose:
        print(f"done: n={summary['n']} acc={summary['overall_accuracy']} "
              f"anls={summary['anls']} no_marker={summary['no_marker']} "
              f"empty={summary['empty_response']}", flush=True)
    return summary


def negotiate(client, model, wanted, verbose=True):
    """Find which of `wanted` the model actually accepts, in one probe call. A refused VALUE is not
    a refused PARAMETER — conflating them drops a cap instead of correcting it. Ported verbatim
    from `mmlu_eval_core.negotiate`."""
    params, notes = dict(wanted), []
    for _ in range(len(wanted) + 2):
        try:
            client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Reply with: ok"}], **params)
            return params, notes
        except Exception as e:
            err = str(e)
            if any(m in err.lower() for m in AUTH_MARKERS):
                raise SystemExit(f"Authentication or quota failure, not negotiated around: {e}")
            offender = next((k for k in params if f"'{k}'" in err), None)
            if offender is None:
                raise
            if "Unsupported value" in err or "does not support" in err:
                m = re.search(r"[Ss]upported values are:?\s*([^.}]+)", err)
                options = re.findall(r"'([^']+)'", m.group(1)) if m else []
                pick = next((v for v in EFFORT_FALLBACKS if v in options),
                            options[0] if options else None)
                if pick is not None:
                    asked = params[offender]
                    params[offender] = pick
                    notes.append(f"{offender}={pick} (asked {asked!r}, refused)")
                    if verbose:
                        print(f"  negotiate: {offender} refused {asked!r}; using {pick!r}",
                              flush=True)
                    continue
            value = params.pop(offender)
            rename = re.search(r"[Uu]se '([A-Za-z_]+)' instead", err)
            if rename:
                params[rename.group(1)] = value
                notes.append(f"{offender} -> {rename.group(1)}")
                if verbose:
                    print(f"  negotiate: {offender} rejected, using {rename.group(1)}", flush=True)
            else:
                notes.append(f"{offender} unsupported, DROPPED")
                if verbose:
                    print(f"  negotiate: {offender} unsupported, dropped", flush=True)
    raise SystemExit(f"no working parameter set for {model}; last tried {params}")
