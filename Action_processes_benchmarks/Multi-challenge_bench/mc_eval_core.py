"""MultiChallenge — shared core for the judge pilot: data, the 10% sample, the upstream judge prompt
and its scorer. Identical for every judge; the runners hold only the client.

What upstream does, and what this keeps (LLM_as_judge/JUDGE_SELECTION.md §7.2):
- judge input: the tested model's FINAL response and the item's `TARGET_QUESTION` (the rubric) —
  no conversation (D5: follow the code). Prompt `src/evaluator.py::JUDGE_PROMPT`, read from the
  source text so not a character can differ, including "Be VERY STRICT!";
- output: structured `{reasoning, verdict ∈ YES/NO}` — the `JudgeResponse` pydantic model, sent as
  the equivalent strict JSON schema because the call goes through OpenRouter, not OpenAI's
  `beta.parse`;
- scoring: an item passes on `verdict == PASS_CRITERIA` (kept although all 273 rows say "YES" —
  D6); per-axis pass rate; headline = MACRO mean over axes (B4). Micro is reported beside it;
- `--attempts 1` (one response per item, B2).
Fixed rather than copied (§7.2 D1, D2): no `max_tokens` kwarg crash; a judge error is its own
status and never becomes a fifth axis 'NA' at 0%; a generation error is retried, never stored as
the response text "FAIL THIS QUESTION" and sent to the judge (C1).
"""

import ast
import collections
import json
import os
import sys

MC_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MC_ROOT))
import judge_core as jc  # noqa: E402

DATA_PATH = os.path.join(MC_ROOT, "data", "benchmark_questions.jsonl")
SAMPLE_PATH = os.path.join(MC_ROOT, "sample_uids.json")
ENV_PATH = os.path.join(MC_ROOT, ".env")
EXPECTED_ITEMS = 273
EXPECTED_AXES = {"INFERENCE_MEMORY": 113, "INSTRUCTION_RETENTION": 69, "SELF_COHERENCE": 50,
                 "RELIABLE_VERSION_EDITING": 41}
SCORER_VERSION = "mc_upstream_v1"


def _upstream_judge_prompt():
    """`JUDGE_PROMPT` from the vendored evaluator, by parsing its source — importing it would pull in
    openai, tqdm and pydantic for one string constant."""
    path = os.path.join(MC_ROOT, "src", "evaluator.py")
    with open(path) as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "JUDGE_PROMPT":
            return node.value.value
    raise SystemExit(f"JUDGE_PROMPT not found in {path}")


JUDGE_PROMPT = _upstream_judge_prompt()

# `class JudgeResponse(BaseModel): reasoning: str; verdict: Literal["YES", "NO"]`
# (src/evaluator.py:7-9), as the strict schema OpenAI's structured output derives from it.
JUDGE_RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {
    "name": "JudgeResponse", "strict": True, "schema": {
        "type": "object",
        "properties": {"reasoning": {"type": "string"},
                       "verdict": {"type": "string", "enum": ["YES", "NO"]}},
        "required": ["reasoning", "verdict"], "additionalProperties": False}}}


def load_items():
    """Every item, keyed by QUESTION_ID, with the counts checked against JUDGE_SELECTION.md §7.2."""
    items = {}
    with open(DATA_PATH) as fh:
        for line in fh:
            d = json.loads(line)
            items[d["QUESTION_ID"]] = d
    axes = collections.Counter(d["AXIS"] for d in items.values())
    if len(items) != EXPECTED_ITEMS or dict(axes) != EXPECTED_AXES:
        raise SystemExit(f"{DATA_PATH}: expected {EXPECTED_ITEMS} items {EXPECTED_AXES}, "
                         f"found {len(items)} {dict(axes)}")
    return items


def build_sample():
    """Stratified by axis, 10%, seeded; written once and committed — every judge sees these items."""
    items = load_items()
    picked = jc.stratified_sample(list(items), lambda q: items[q]["AXIS"])
    sizes = collections.Counter(d["AXIS"] for d in items.values())
    jc.write_sample_file(SAMPLE_PATH, picked, sizes,
                         "MultiChallenge judge pilot: QUESTION_IDs, stratified by AXIS")
    return picked


def load_sample():
    if not os.path.exists(SAMPLE_PATH):
        raise SystemExit(f"{SAMPLE_PATH} missing — run `python mc_eval_core.py --build-sample` once")
    items = load_items()
    picked = jc.read_sample_file(SAMPLE_PATH)
    uids = [u for axis in sorted(picked) for u in picked[axis]]
    missing = [u for u in uids if u not in items]
    if missing:
        raise SystemExit(f"sample names {len(missing)} unknown QUESTION_IDs, e.g. {missing[:3]}")
    return [items[u] for u in uids]


def judge_messages(response, item):
    """Exactly what upstream sends: one user turn, no system message (src/evaluator.py:40-43)."""
    return [{"role": "user", "content": JUDGE_PROMPT.format(response, item["TARGET_QUESTION"])}]


def parse_verdict(text):
    """The structured answer's verdict, or None. Lenient about packaging (fences, case, a stray
    prefix before the object), strict about meaning: exactly YES or NO."""
    obj = jc.first_json_object(text)
    if obj is None or not isinstance(obj.get("verdict"), str):
        return None
    verdict = obj["verdict"].strip().strip(".").upper()
    return verdict if verdict in ("YES", "NO") else None


def majority(values):
    """The label a STRICT majority of the replicates gave — `values` is one entry per replicate,
    None where that replicate produced nothing usable. [True, None, None] is None: one surviving
    replicate is not a majority of three."""
    votes = collections.Counter(v for v in values if v is not None)
    if not votes:
        return None
    top, n = votes.most_common(1)[0]
    return top if n > len(values) / 2 else None


def score(judged):
    """`judged`: {QUESTION_ID: passed (bool) or None}, with axes from the dataset. Upstream's macro
    headline, plus micro, over the items that have a verdict; `n_unjudged` counts the rest. If an
    axis present in `judged` has no verdict at all, the macro is None — averaging the remaining axes
    would be a different quantity that looks the same — and the axis is named in `axes_missing`."""
    items = load_items()
    per_axis = collections.defaultdict(lambda: [0, 0])
    unjudged = 0
    for qid, passed in judged.items():
        a = per_axis[items[qid]["AXIS"]]
        if passed is None:
            unjudged += 1
            continue
        a[0] += int(passed)
        a[1] += 1
    axis_scores = {axis: 100.0 * p / n for axis, (p, n) in sorted(per_axis.items()) if n}
    missing = sorted(axis for axis, (_, n) in per_axis.items() if not n)
    n = sum(v[1] for v in per_axis.values())
    return {"overall_macro": (sum(axis_scores.values()) / len(axis_scores))
            if axis_scores and not missing else None,
            "overall_micro": (100.0 * sum(v[0] for v in per_axis.values()) / n) if n else None,
            "axis_scores": axis_scores, "axes_missing": missing, "n_judged": n, "n_unjudged": unjudged}


if __name__ == "__main__":
    if "--build-sample" in sys.argv:
        if os.path.exists(SAMPLE_PATH) and "--force" not in sys.argv:
            raise SystemExit(f"{SAMPLE_PATH} exists; the sample is drawn once. --force to redraw.")
        for axis, uids in build_sample().items():
            print(f"{axis:26s} {len(uids):3d} of {EXPECTED_AXES[axis]}")
    else:
        print(__doc__)
