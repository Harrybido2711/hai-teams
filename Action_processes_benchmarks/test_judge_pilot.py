"""Pins the judge pilot's parsers, samples, verbatim prompts and upstream scorer. Calls no API.
Run after any change to judge_core, mc_eval_core or wb_eval_core:

    python test_judge_pilot.py
"""

import os
import sys

GROUP_ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "Multi-challenge_bench", "Wonderbread_bench"):
    sys.path.insert(0, os.path.join(GROUP_ROOT, sub))
import compare_judges as cj  # noqa: E402
import judge_core as jc  # noqa: E402
import mc_eval_core as mc  # noqa: E402
import wb_eval_core as wb  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r}, want {want!r}")


# --- samples: drawn once, committed, 10% per stratum
mc_sample = mc.load_sample()
check("mc sample size", len(mc_sample), 27)
check("mc per axis", {a: sum(i["AXIS"] == a for i in mc_sample) for a in mc.EXPECTED_AXES},
      {"INFERENCE_MEMORY": 11, "INSTRUCTION_RETENTION": 7, "SELF_COHERENCE": 5,
       "RELIABLE_VERSION_EDITING": 4})
check("mc redraw identical", jc.stratified_sample(list(mc.load_items()),
                                                  lambda q: mc.load_items()[q]["AXIS"]),
      jc.read_sample_file(mc.SAMPLE_PATH))
check("qa sample size", len(wb.load_sample("qa")), 12)
check("sop sample size", len(wb.load_sample("sop")), 16)
check("demos needed", len(wb.demos_needed()), 31)

# --- verbatim prompts
check("mc prompt strictness line", "Be VERY STRICT!" in mc.JUDGE_PROMPT, True)
check("mc prompt slots", mc.JUDGE_PROMPT.count("{}"), 2)
msgs = mc.judge_messages("RESP", {"TARGET_QUESTION": "RUBRIC?"})
check("mc one user turn", [m["role"] for m in msgs], ["user"])
check("mc slot order", msgs[0]["content"].index("RESP") < msgs[0]["content"].index("RUBRIC?"), True)
qa_msgs = wb.qa_judge_messages("completeness", "Q?", "HUMAN LABEL", "RESP")
check("qa system prompt", qa_msgs[0], {"role": "system",
                                       "content": "You are a helpful assistant that automates digital workflows."})
check("qa reference shown to completeness", "HUMAN LABEL" in qa_msgs[1]["content"][0]["text"], True)
check("qa reference withheld from clarity",
      "HUMAN LABEL" in wb.qa_judge_messages("clarity", "Q?", "HUMAN LABEL", "RESP")[1]["content"][0]["text"],
      False)
sop_msgs = wb.sop_judge_messages("Click Save", ["Open page", "Click the Save button"])
check("sop list indexed", "1 - Click the Save button" in sop_msgs[1]["content"], True)
check("sop asks for json", "json" in sop_msgs[1]["content"].lower(), True)

# --- parsers: lenient about packaging, strict about meaning
for text, want in [('{"reasoning": "r", "verdict": "YES"}', "YES"),
                   ('```json\n{"reasoning": "r", "verdict": "no"}\n```', "NO"),
                   ('Here: {"reasoning": "r", "verdict": "NO"} done', "NO"),
                   ('{"reasoning": "r", "verdict": "MAYBE"}', None), ("YES", None), ("", None)]:
    check(f"verdict {text[:30]!r}", mc.parse_verdict(text), want)
for text, want in [("2", 2), ("**3**", 3), ("Score: 1", 1), ("2 - mostly complete", 2),
                   ("Rating: 2\n\nExplanation: see step #5", 2), ("The response scores 1.", 1),
                   ("1 or 2", None), ("4", None), ("2.5", None), ("", None)]:
    check(f"qa score {text!r}", wb.parse_qa_score(text), want)
parse3 = wb.make_index_parser(3)
for text, want in [('{"scratchpad": "s", "index": 2}', 2), ('{"index": -1}', -1), ('{"index": 0}', 0),
                   ('```json\n{"index": "1"}\n```', 1), ('{"index": 3}', None), ('{"index": 1.5}', None),
                   ('{"scratchpad": "no index"}', None), ("-1", None)]:
    check(f"sop index {text[:30]!r}", parse3(text), want)

# --- upstream scorer, fed parsed indices
m = wb.sop_metrics([0, -1, 1], [0, 2])
check("sop precision", m["precision"], 2 / 3)
check("sop recall", m["recall"], 1.0)
check("sop ordering", m["ordering"], 2 / 3)
check("sop f1", round(m["f1"], 6), 0.8)
check("sop out of order", wb.sop_metrics([1, 0], [0, 1])["ordering"], 0.5)
check("sop missing decision", wb.sop_metrics([0, None], [0]), None)
check("preprocess strips numbering", wb.sop_lines("1. Click Save. Then confirm.\n\n2. Done"),
      ["Click Save. Then confirm.", "Done"])
# the anchor pairs reproduce upstream's own line counts — all but anchor20, whose stored pred_sop
# holds 10 numbered steps while upstream recorded n_lines_pred_sop=11: the CSV kept a different
# text from the one upstream scored. Its human labels are over that unrecorded text; noted, kept.
anchor = wb.load_anchor_sop()
import pandas as pd  # noqa: E402
counts = pd.read_csv(wb.ANCHOR_SOP)[["n_lines_pred_sop", "n_lines_gold_sop"]].values.tolist()
counts[20][0] = 10
check("anchor line counts", [[len(wb.sop_lines(a["pred_sop"])), len(wb.sop_lines(a["gold_sop"]))]
                             for a in anchor.values()], counts)

# --- MultiChallenge score: macro over axes, failures excluded rather than scored NO
items = mc.load_items()
ids = {a: [q for q in items if items[q]["AXIS"] == a][:2] for a in mc.EXPECTED_AXES}
judged = {ids["INFERENCE_MEMORY"][0]: True, ids["INFERENCE_MEMORY"][1]: False,
          ids["SELF_COHERENCE"][0]: True, ids["SELF_COHERENCE"][1]: None}
s = mc.score(judged)
check("mc macro", s["overall_macro"], 75.0)
check("mc micro", round(s["overall_micro"], 4), round(200 / 3, 4))
check("mc unjudged", s["n_unjudged"], 1)
check("mc no missing axis", s["axes_missing"], [])
s = mc.score({ids["INFERENCE_MEMORY"][0]: True, ids["SELF_COHERENCE"][0]: None})
check("mc axis with no verdict voids the macro", (s["overall_macro"], s["axes_missing"]),
      (None, ["SELF_COHERENCE"]))
check("majority tie", mc.majority([True, False]), None)
check("majority", mc.majority([True, None, True]), True)
check("one surviving replicate is not a majority", mc.majority([True, None, None]), None)
check("qa label: median when all answered", cj.qa_label([1, 3, 3]), 3)
check("qa label: [1, None, 3] is unresolved, not 'best'", cj.qa_label([1, None, 3]), None)
check("qa label: agreeing majority", cj.qa_label([2, None, 2]), 2)
# replicates that agree a line matched but name different lines: precision/recall kept, ordering not
m = cj.sop_pair_metrics("u", 2, 1, {("u", "pred", 0): True, ("u", "pred", 1): False, ("u", "gold", 0): True},
                        {("u", "pred", 0): None, ("u", "pred", 1): -1, ("u", "gold", 0): 0})
check("sop pair: match decided, index split", (m["precision"], m["recall"], m["ordering"]), (0.5, 1.0, None))
check("sop pair: a line with no majority match voids the pair",
      cj.sop_pair_metrics("u", 1, 1, {("u", "pred", 0): None, ("u", "gold", 0): True}, {}), None)
# an axis outside the common set must void the macro, not shrink it to three axes
qids = [q for q in items if items[q]["AXIS"] in ("INFERENCE_MEMORY", "SELF_COHERENCE")][:1] + \
       [q for q in items if items[q]["AXIS"] == "SELF_COHERENCE"][:1]
common = qids[:1]
check("macro over the common set names the dropped axis",
      mc.score({q: (True if q in common else None) for q in qids})["overall_macro"], None)

# --- production paths, against mock clients (no API, no sleeping)
import tempfile  # noqa: E402
import types  # noqa: E402
jc.time.sleep = lambda seconds: None
TMP = tempfile.mkdtemp(prefix="judge_pilot_test_")


def reset_streaks():
    for k in jc._streak:
        jc._streak[k] = 0
    jc._recent.clear()


def fake_response(content, cost, finish="stop", refusal=None):
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, cost=cost,
                                  completion_tokens_details=types.SimpleNamespace(reasoning_tokens=0))
    message = types.SimpleNamespace(content=content, refusal=refusal)
    return types.SimpleNamespace(usage=usage, model="m", provider="P",
                                 choices=[types.SimpleNamespace(message=message, finish_reason=finish)])


def sequence(*responses):
    it = iter(responses)
    return lambda: jc._answer(next(it))


reset_streaks()
text, meta, status = jc.call_with_retries(sequence(fake_response("", 0.5), fake_response("", 0.5),
                                                   fake_response("ok", 1.0)), "m", TMP)
check("empty bodies retried", (text, status, meta["n_attempts"]), ("ok", "ok", 3))
check("every attempt's bill summed", (meta["cost"], meta["prompt_tokens"]), (2.0, 30))
text, parsed, meta, status = jc.call_and_parse(
    sequence(fake_response("junk", 1.0), fake_response("2", 1.0)), wb.parse_qa_score, "m", TMP)
check("re-asked answer summed", (parsed, status, meta["cost"], meta["n_parse_attempts"]), (2, "ok", 2.0, 2))
_, meta, status = jc.call_with_retries(sequence(fake_response(None, 0.0, "content_filter",
                                                                 refusal="blocked")), "m", TMP)
check("message.refusal is a refusal", (status, meta["refusal"], meta["n_attempts"]), ("refused", "blocked", 1))
_, _, status = jc.call_with_retries(sequence(fake_response(None, 0.0, "content_filter")), "m", TMP)
check("content_filter with no content is a refusal", status, "refused")
reset_streaks()
try:
    for _ in range(jc.CONSECUTIVE_PARSE_FAIL_LIMIT):
        st = jc.call_and_parse(lambda: jc._answer(fake_response("junk", 0.1)), wb.parse_qa_score, "m", TMP)[3]
        jc.note_outcome(st, "m", TMP)
    check("parse-fail streak halts", "no halt", "SystemExit")
except SystemExit:
    check("parse-fail marker written", os.path.exists(os.path.join(TMP, "PARSE_HALT.txt")), True)
reset_streaks()
check("exit code: parse failures over the rate",
      jc.judge_exit_code([{"status": "ok"}] * 9 + [{"status": "parse_fail"}], 10), 1)
check("exit code: refusals are a finding, not a crash",
      jc.judge_exit_code([{"status": "ok"}] * 9 + [{"status": "refused"}], 10), 0)
check("exit code: missing rows", jc.judge_exit_code([{"status": "ok"}], 2), 1)
check("exit code: a mostly refused set fails", jc.judge_exit_code([{"status": "refused"}] * 3, 3), 1)

path = os.path.join(TMP, "rows.jsonl")
cfg = {"a": 1}
jc.append_row(path, {"uid": "u1", "status": "failed", "config": cfg})
jc.append_row(path, {"uid": "u2", "status": "ok", "judged_sha": "x", "config": cfg})
check("failed rows retried on resume", sorted(jc.load_done(path, cfg)), ["u2"])
try:
    jc.load_done(path, {"a": 2})
    check("config change refused", "resumed", "SystemExit")
except SystemExit:
    pass
try:
    jc.check_judged(jc.load_done(path, cfg), {"u2": "y"}, path)
    check("changed judged text refused", "resumed", "SystemExit")
except SystemExit:
    pass
jc.lock_output(path)
import subprocess  # noqa: E402
second = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {GROUP_ROOT!r}); "
                         f"import judge_core as jc; jc.lock_output({path!r})"], capture_output=True)
check("a second writer is refused", second.returncode != 0, True)

os.environ.pop("OPENROUTER_API_KEY", None)
try:
    jc._key("OPENROUTER_API_KEY")
    check("missing key refused", "used a fallback", "SystemExit")
except SystemExit:
    pass
import openai  # noqa: E402 — the real package, or wb_eval_core's stand-in for it
os.environ["OPENROUTER_API_KEY"] = os.environ["OPENAI_API_KEY"] = "test-key"
if getattr(openai, "__absent__", False):
    try:
        jc.judge_client()
        check("a stand-in openai is never used for a client", "client built", "SystemExit")
    except SystemExit:
        print("(openai not installed here: client retry checks skipped)")
else:
    check("judge client: no hidden SDK retries", jc.judge_client().max_retries, 0)
    check("generation client: no hidden SDK retries", jc.gen_client().max_retries, 0)
check("every judge pinned, no fallback", {j: jc.judge_routing(j)["allow_fallbacks"] for j in jc.JUDGES},
      {j: False for j in jc.JUDGES})

# --- the transport patch: upstream's own builder reaches our transport (needs data/demos)
if not [p for d in wb.demos_needed() for p in wb.check_demo(d)]:
    up_ = wb.upstream()
    seen = []
    up_.qa_gen._fetch_completion = lambda messages, model, **kw: seen.append(messages) or "ANSWER"
    qa = wb.load_qa()
    uid = wb.load_sample("qa")[0]
    out = up_.qa_gen.generate_response(wb.DEMOS_DIR, qa[uid]["Task ID(s)"], qa[uid]["Question Instantiation"],
                                       qa[uid]["Evidence"], "GPT4")
    check("upstream QA builder calls the patched transport", (out, len(seen)), ("ANSWER", 1))
else:
    print("(data/demos not fetched here: transport check skipped)")

# --- statistics
check("kappa perfect", cj.cohen_kappa([1, 2, 3, 1], [1, 2, 3, 1]), 1.0)
check("kappa chance", cj.cohen_kappa([True, False, True, False], [True, True, False, False]), 0.0)
check("qwk partial", round(cj.cohen_kappa([1, 2, 3], [1, 3, 3], "quadratic", [1, 2, 3]), 4), 0.8)
check("mcnemar none discordant", cj.mcnemar_p(0, 0), 1.0)

# --- the stand-ins for absent client libraries are never called
up = wb.upstream()
check("helpers imported", hasattr(up.helpers, "create_merged_trace"), True)

if FAILED:
    print("\n".join(FAILED))
    raise SystemExit(f"{len(FAILED)} check(s) failed")
print("all checks passed")
