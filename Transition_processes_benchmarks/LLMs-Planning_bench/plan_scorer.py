"""Offline scorer for PlanBench rows — upstream's own parsers and VAL, behind ONE lenient extraction.

`plan_eval_core.py` makes the calls and stores `raw_response`; this file turns stored rows into scores.
It runs wherever VAL and `tarski` exist — locally after `setup_scorer.sh`, never inside a Quest
generation job (Quest's shared Python has no `tarski`). Rescoring stored rows is local work.

What is upstream's and what is ours:

- **Upstream's, imported unchanged:** the text→PDDL plan parsers and text→state parsers
  (`plan-bench/utils/text_to_pddl.py`, loaded by file path so `utils/__init__.py` — which imports
  `openai` and reads `OPENAI_API_KEY` at import — never runs), `model_parser.ModelWriter` for the
  re-planning instances of tasks 4 and 6, tarski's PDDL reader for the action set, and the VAL binary.
- **Upstream's, ported:** `ResponseEvaluator.parse_output` and the verification verdicts (task 3) —
  the class lives in a module that imports `transformers` at the top, so it cannot be imported here.
  The port is line-for-line except that a response with no valid/invalid verdict is a parse failure
  where upstream raised KeyError.
- **Ours, by design (script-skeleton.md rule 7):**
  1. `extract()` — the lenient step, applied identically to every model before upstream's parsers.
  2. The optimal cost for task 2 is the length of the shipped ground-truth plan, not a fresh Fast
     Downward run: both are `astar(lmcut())`, which is optimal, and every domain is unit-cost.
  3. Empty responses stay in the denominator (`acc`); upstream skipped them. `acc_answered` is
     upstream's denominator, reported beside it.
  4. A VAL output with no recognisable verdict is a `scorer_error`, never a silent "wrong" — upstream
     returned False, so a missing binary scored every plan 0.

`python plan_scorer.py --selftest` re-scores upstream's own stored GPT-4 responses (blocksworld and
blocksworld_3, all ten tasks) and reports agreement with upstream's stored verdicts, in both
extraction modes. Run it after any change here.
"""

import argparse
import collections
import csv
import glob
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile

import yaml

PB_ROOT = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(PB_ROOT, "plan-bench")
sys.path.insert(0, PB_ROOT)
sys.path.insert(0, UPSTREAM)                        # model_parser
import plan_eval_core as core  # noqa: E402
from model_parser.writer_new import ModelWriter  # noqa: E402  (needs tarski)
from tarski.io import PDDLReader  # noqa: E402

_spec = importlib.util.spec_from_file_location("pb_text_to_pddl",
                                               os.path.join(UPSTREAM, "utils", "text_to_pddl.py"))
ttp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ttp)

SCORER_VERSION = "pb_lenient_v1"
EXTRACTION_MODES = ("lenient", "upstream")


class ScorerInfraError(Exception):
    """The scorer itself cannot run (no VAL, unreadable domain). Stops scoring; never a score.
    Deliberately NOT a RuntimeError: score_results turns a RuntimeError into one row's scorer_error."""


# Slices whose upstream parser crashes on every well-formed answer (planbench.md § Traps), so they
# score 0 for every model by construction. Kept in the by-slice table, where the 0 is visible, and
# left out of the by-task pooling, where it would drag each task's headline down by up to ~12 points.
BROKEN_UPSTREAM_SLICES = ("obfuscated_deceptive_logistics/task_1_plan_generation",
                          "obfuscated_deceptive_logistics/task_2_plan_optimality")


# ---------------------------------------------------------------- VAL


def val_binary():
    """$VAL_BIN, else the build for this platform: upstream's shipped Linux binary on Linux, the
    binary `setup_scorer.sh` compiles from the vendored source on macOS."""
    path = os.getenv("VAL_BIN")
    if not path:
        val_dir = os.path.join(PB_ROOT, "planner_tools", "VAL")
        path = (os.path.join(val_dir, "validate") if platform.system() == "Linux"
                else os.path.join(val_dir, "build", platform.system().lower(), "validate"))
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        raise ScorerInfraError(f"no executable VAL at {path} — run setup_scorer.sh or set VAL_BIN")
    return path


_VALID = "Plan valid"
_INVALID = ("Plan invalid", "Plan failed to execute", "Bad operator in plan", "Bad plan description",
            "Goal not satisfied")


class PlanUnvalidatable(RuntimeError):
    """VAL died on this plan (a signal exit) or reported `Error in type-checking!`: the plan is
    malformed — unbalanced parentheses, wrong arity, an object the problem does not have."""


# A crash rate above this share of the plans that reached VAL means the binary is broken, not the plans.
MAX_VAL_CRASH_SHARE = 0.05


def validate(domain, problem, plan_file):
    """True / False from VAL's own verdict line. A malformed plan makes VAL crash or print a
    type-checking error with no verdict; that raises `PlanUnvalidatable`, which `score_row` records as
    an invalid plan (upstream's `"Plan valid" in output` scores it wrong too). Any other silence raises
    a plain RuntimeError: upstream returned False for an empty output, which is how a missing binary
    would score every plan 0 (utils/__init__.py:255)."""
    out = subprocess.run([val_binary(), domain, problem, plan_file], capture_output=True,
                         text=True, timeout=120)
    text = out.stdout + out.stderr
    if "Problem in domain" in text:
        raise ScorerInfraError(f"VAL: problem in domain {domain}")
    if _VALID in text:
        return True
    if any(s in text for s in _INVALID):
        return False
    if out.returncode < 0 or "Error in type-checking" in text:
        raise PlanUnvalidatable(f"VAL exit {out.returncode}: {text.strip()[-120:]!r}")
    raise RuntimeError(f"VAL gave no verdict (exit {out.returncode}): {text.strip()[:200]!r}")


# ---------------------------------------------------------------- upstream data, cached


_CONFIGS, _PROMPTS, _ACTIONS = {}, {}, {}


def config(domain):
    if domain not in _CONFIGS:
        with open(os.path.join(UPSTREAM, "configs", f"{domain}.yaml")) as fh:
            _CONFIGS[domain] = yaml.safe_load(fh)
    return _CONFIGS[domain]


def prompt_instance(domain, task, instance_id):
    key = (domain, task)
    if key not in _PROMPTS:
        with open(os.path.join(core.PROMPTS_DIR, domain, f"{task}.json")) as fh:
            _PROMPTS[key] = {r["instance_id"]: r for r in json.load(fh)["instances"]}
    return _PROMPTS[key][instance_id]


def paths(domain, task, instance_id):
    data = config(domain)
    inst_dir = data["generalized_instance_dir"] if "plan_generalization" in task else data["instance_dir"]
    domain_pddl = os.path.join(UPSTREAM, "instances", data["domain_file"])
    problem = os.path.join(UPSTREAM, "instances", inst_dir, data["instances_template"].format(instance_id))
    for p in (domain_pddl, problem):
        if not os.path.exists(p):
            raise ScorerInfraError(f"missing PDDL file {p}")
    return domain_pddl, problem


def action_set(domain_pddl, problem):
    """tarski's parsed actions, exactly as upstream's get_problem(). Cached per domain file — the
    action schemas come from the domain, the instance only supplies objects."""
    if domain_pddl not in _ACTIONS:
        reader = PDDLReader(raise_on_error=True)
        reader.parse_domain(domain_pddl)
        _ACTIONS[domain_pddl] = reader.parse_instance(problem).actions
    return _ACTIONS[domain_pddl]


# ---------------------------------------------------------------- the lenient step


_THINK = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.S | re.I)
_ENUM = re.compile(r"^(?:step\s*)?\d{1,3}\s*[.):]\s+", re.I)
_BULLET = re.compile(r"^(?:[-*•]|#+)\s*")
_VERDICT = re.compile(r"plan is (in)?valid", re.I)


def _normalise(text):
    """Packaging off, words untouched: <think> blocks, markdown emphasis and code ticks, list
    bullets and step numbers. Never changes a word the parsers read."""
    text = _THINK.sub("", text).replace("**", "").replace("__", "").replace("`", "")
    lines = []
    for line in text.splitlines():
        s = _BULLET.sub("", line.strip())
        lines.append(_ENUM.sub("", s))
    return "\n".join(lines)


def _upto(text, marker):
    i = text.upper().find(marker)
    return text if i == -1 else text[:i]


def extract(raw, kind, mode="lenient"):
    """The one extraction, identical for every model. `upstream` returns the raw text, which is what
    upstream's evaluator fed its parsers; `lenient`:

    1. cut at the first `[STATEMENT]` — the answer ends where a new problem would begin (upstream
       sent stop="[STATEMENT]" to completion models for exactly this);
    2. plans: the LAST `[PLAN]…[PLAN END]` block, or everything before the last `[PLAN END]` when the
       model never re-opened `[PLAN]` (the prompt itself ends on an open `[PLAN]`); no `[PLAN END]`
       at all → the whole answer, as upstream;
    3. states: what follows the last `[RESULTING STATE]`, up to the next `[` marker;
    4. markdown, bullets and step numbers stripped (`_normalise`); the verdict phrase lower-cased.
    """
    if mode == "upstream":
        return raw
    text = _upto(raw or "", "[STATEMENT]")
    up = text.upper()
    if kind == "plan" and "[PLAN END]" in up:
        end = up.rfind("[PLAN END]")
        start = up.rfind("[PLAN]", 0, end)
        text = text[start + len("[PLAN]"):end] if start != -1 else text[:end]
    elif kind == "state":
        start = up.rfind("[RESULTING STATE]")
        if start != -1:
            text = text[start + len("[RESULTING STATE]"):]
        text = text.split("[", 1)[0]
    text = _normalise(text)
    if kind == "verification":
        text = _VERDICT.sub(lambda m: m.group(0).lower(), text)
    return text


# ---------------------------------------------------------------- task 3, ported


def parse_output(actions, output, data, plan_file):
    """Upstream ResponseEvaluator.parse_output (response_evaluation.py), line for line."""
    output_dict = {}
    goal_cond = precond_act = precond_act_flag = precond_pred = False
    for line in output.split("\n"):
        if "[STATEMENT]" in line:
            break
        if line.strip() == "":
            continue
        if goal_cond:
            output_dict["unmet_goal"] = ttp.text_to_state(line.strip(), data)
            goal_cond = False
            continue
        if precond_act:
            _, action = ttp.text_to_plan(line.strip(), actions, plan_file, data)
            output_dict["unmet_precondition"]["action"] = action
            precond_act = False
            precond_act_flag = True
            continue
        if precond_act_flag and precond_pred:
            output_dict["unmet_precondition"]["predicate"] = ttp.text_to_state(line.strip(), data)
            precond_pred = False
            precond_act_flag = False
        if "plan is valid" in line:
            if "valid" not in output_dict:
                output_dict["valid"] = True
            break
        elif "plan is invalid" in line:
            output_dict["valid"] = False
        if "unmet goal" in line and "unmet precondition" in line:
            break
        if "unmet goal" in line:
            output_dict["unmet_goal"] = ""
            goal_cond = True
        elif "unmet precondition" in line:
            if "action" in line:
                output_dict["unmet_precondition"] = {}
                precond_act = True
            else:
                precond_pred = True
        elif "Unmet precondition:" in line:
            if precond_act_flag:
                output_dict["unmet_precondition"]["predicate"] = ttp.text_to_state(line.strip(), data)
                precond_act_flag = False
    return output_dict


def verification_verdicts(parsed, gold):
    """Upstream evaluate_verification's three levels: binary, with type, with explanation."""
    binary = w_type = w_expl = False
    if parsed.get("valid") == gold["valid"]:
        binary = True
        if not parsed["valid"]:
            if sorted(parsed) == sorted(gold):
                w_type = True
                if "unmet_goal" in parsed:
                    if any(p in gold["unmet_goal"] for p in parsed["unmet_goal"]):
                        w_expl = True
                if "unmet_precondition" in parsed:
                    try:
                        if parsed["unmet_precondition"]["action"] == gold["unmet_precondition"]["action"]:
                            if any(p in gold["unmet_precondition"]["predicate"]
                                   for p in parsed["unmet_precondition"]["predicate"]):
                                w_expl = True
                    except KeyError:
                        pass
        else:
            w_type = w_expl = True
    return binary, w_type, w_expl


# ---------------------------------------------------------------- one row


def kind_of(task):
    if task.startswith("task_3"):
        return "verification"
    if task.startswith("task_7"):
        return "state"
    return "plan"


def _gt_plan_length(gt):
    return sum(1 for line in str(gt).splitlines() if line.strip().startswith("("))


def score_row(domain, task, instance_id, raw, mode="lenient"):
    """-> dict(correct, parse_fail, parse_error, extracted, + task extras). `correct` is the task's
    headline: plan valid (task 2: valid AND optimal length), state equal, verification binary."""
    out = {"correct": 0, "parse_fail": 0, "parse_error": None, "extracted": None}
    if not str(raw or "").strip():
        return out
    data = config(domain)
    inst = prompt_instance(domain, task, instance_id)
    kind = kind_of(task)
    text = extract(raw, kind, mode)
    domain_pddl, problem = paths(domain, task, instance_id)
    with tempfile.TemporaryDirectory(prefix="pbscore_") as tmp:
        plan_file = os.path.join(tmp, "llm_plan")
        try:
            if kind == "state":
                state = ttp.text_to_state(text, data)
                out["extracted"] = state
                out["correct"] = int(sorted(inst["ground_truth_plan"]) == sorted(state))
                return out
            actions = action_set(domain_pddl, problem)
            if kind == "verification":
                parsed = parse_output(actions, text, data, plan_file)
                gold = parse_output(actions, inst["ground_truth_plan"], data, plan_file)
                out["extracted"] = parsed
                if "valid" not in parsed:
                    out["parse_fail"] = 1
                    out.update(correct_w_type=0, correct_w_expl=0)
                    return out
                binary, w_type, w_expl = verification_verdicts(parsed, gold)
                out.update(correct=int(binary), correct_w_type=int(w_type), correct_w_expl=int(w_expl))
                return out
            llm_plan, _ = ttp.text_to_plan(text, actions, plan_file, data)
        except ScorerInfraError:
            raise
        except Exception as error:          # upstream: "Plan extraction failed" -> scored wrong
            out["parse_fail"] = 1
            out["parse_error"] = f"{type(error).__name__}: {error}"[:200]
            return out
        out["extracted"] = llm_plan
        # An empty plan still goes to VAL, as upstream's did: in replanning the unexpected event
        # sometimes achieves the goal, and "no further actions are needed" is then the right
        # answer (47 of upstream's 500 GPT-4 task-6 rows). VAL decides; empty_plan only records it.
        out["empty_plan"] = int(not llm_plan.strip())
        try:
            if "new_instance" in inst:      # tasks 4 and 6: validate against the modified problem
                new_domain, new_problem = os.path.join(tmp, "pr-new-domain.pddl"), os.path.join(tmp, "pr-new-problem.pddl")
                ModelWriter(inst["new_instance"]).write_files(new_domain, new_problem)
                valid = validate(new_domain, new_problem, plan_file)
            else:
                valid = validate(domain_pddl, problem, plan_file)
        except PlanUnvalidatable as error:  # malformed plan: invalid, and counted in `val_crash`
            valid = False
            out["val_crash"] = 1
            out["val_note"] = str(error)[:160]
        out["valid"] = int(valid)
        out["correct"] = int(valid)
        if "optimality" in task:
            out["llm_plan_length"] = sum(1 for line in llm_plan.split("\n") if line)
            out["optimal_length"] = _gt_plan_length(inst["ground_truth_plan"])
            out["correct"] = int(valid and out["llm_plan_length"] == out["optimal_length"])
    return out


# ---------------------------------------------------------------- a model's results


def _write_csv(path, records):
    if not records:
        return
    fields = list(dict.fromkeys(k for rec in records for k in rec))   # task-3 rows add columns

    def write(fh):
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    core._atomic_write(path, write)


def score_results(model_dir, model, sample, smoke=False, mode="lenient"):
    out_dir = core.results_dir(model_dir, sample, smoke)
    rows, problems = core.collect(out_dir, model)
    if not rows:
        raise SystemExit(f"no rows for {model} under {out_dir}")
    val_binary()                                   # fail before scoring, not per row
    scored, errors = [], collections.Counter()
    for r in rows:
        inst = prompt_instance(r["domain"], r["task"], r["instance_id"])
        if core._sha1(inst["query"]) != r.get("query_sha1"):
            problems.append(f"{r['uid']}: prompt changed since the row was generated")
        try:
            s = score_row(r["domain"], r["task"], r["instance_id"], r.get("raw_response", ""), mode)
        except RuntimeError as error:              # VAL said nothing usable for this one plan
            s = {"correct": 0, "parse_fail": 0, "parse_error": None, "extracted": None,
                 "scorer_error": str(error)[:200]}
            errors[r["slice"]] += 1
        scored.append({**{k: v for k, v in r.items() if k != "config"}, **s,
                       "scorer": SCORER_VERSION, "extraction": mode})
    slug = core.model_slug(model)
    suffix = "" if mode == "lenient" else f"_{mode}"

    def write_rows(fh):
        for x in scored:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    core._atomic_write(os.path.join(out_dir, f"{slug}_scored{suffix}.jsonl"), write_rows)

    def summary(group_key):
        groups = collections.defaultdict(list)
        for x in scored:
            groups[x[group_key]].append(x)
        recs = []
        for key in sorted(groups):
            xs = groups[key]
            answered = [x for x in xs if not x.get("empty")]
            rec = {group_key: key, "rows": len(xs), "empty": len(xs) - len(answered),
                   "correct": sum(x["correct"] for x in xs),
                   "acc": round(sum(x["correct"] for x in xs) / len(xs), 4),
                   "acc_answered": round(sum(x["correct"] for x in answered) / len(answered), 4) if answered else "",
                   "parse_fail": sum(x["parse_fail"] for x in xs),
                   "scorer_errors": sum(1 for x in xs if x.get("scorer_error")),
                   "val_crash": sum(1 for x in xs if x.get("val_crash")),
                   "top_parse_error": (collections.Counter(x["parse_error"].split(":")[0] for x in xs
                                                           if x.get("parse_error")).most_common(1) or [("", 0)])[0][0]}
            if key.startswith("task_3") or "/task_3" in key:
                rec["acc_w_type"] = round(sum(x.get("correct_w_type", 0) for x in xs) / len(xs), 4)
                rec["acc_w_expl"] = round(sum(x.get("correct_w_expl", 0) for x in xs) / len(xs), 4)
            recs.append(rec)
        return recs

    by_slice = summary("slice")
    kept = scored
    scored = [x for x in scored if x["slice"] not in BROKEN_UPSTREAM_SLICES]
    by_task = summary("task")
    scored = kept
    if errors:
        problems.append(f"VAL gave no verdict for {sum(errors.values())} plan(s) "
                        f"({dict(errors)}) — scored 0 with scorer_error set, NOT a real score")
    crashed = [x for x in scored if x.get("val_crash")]
    reached = [x for x in scored if "valid" in x]
    if reached and len(crashed) > MAX_VAL_CRASH_SHARE * len(reached):
        problems.append(f"VAL crashed or failed type-checking on {len(crashed)} of {len(reached)} plans "
                        f"(> {MAX_VAL_CRASH_SHARE:.0%}) — suspect the binary, not the plans")
    if crashed:
        print(f"  note: {len(crashed)} plan(s) made VAL crash or fail type-checking "
              f"({dict(collections.Counter(x['slice'] for x in crashed))}) — scored invalid, `val_crash` set", flush=True)
    _write_csv(os.path.join(out_dir, f"{slug}_scores_by_slice{suffix}.csv"), by_slice)
    _write_csv(os.path.join(out_dir, f"{slug}_scores_by_task{suffix}.csv"), by_task)
    print(f"scored {len(scored)} rows of {model} [{sample}{' smoke' if smoke else ''}, {mode}] "
          f"-> {out_dir}", flush=True)
    for rec in by_task:
        print(f"  {rec['task']:45s} n={rec['rows']:4d} acc={rec['acc']:.3f} empty={rec['empty']} "
              f"parse_fail={rec['parse_fail']} scorer_err={rec['scorer_errors']}"
              + (f" err={rec['top_parse_error']}" if rec["top_parse_error"] else ""), flush=True)
    for p in problems:
        print(f"  PROBLEM: {p}", flush=True)
    return by_slice, problems


# ---------------------------------------------------------------- self-test against upstream


def selftest():
    """Re-score upstream's stored GPT-4 responses and compare with upstream's stored verdicts. Those
    files were written by an older upstream code version, so agreement in `upstream` mode measures
    the port; the `lenient` column then shows what the lenient step changes, and in which direction."""
    files = sorted(glob.glob(os.path.join(UPSTREAM, "results", "blocksworld*", "gpt-4_chat", "*.json")))
    if not files:
        raise SystemExit("no upstream result files found")
    val_binary()
    print(f"{'file':55s} {'n':>4s} {'upstream-mode agree':>20s} {'lenient agree':>14s} "
          f"{'up acc':>7s} {'ours lenient':>12s}")
    for f in files:
        domain = f.split(os.sep)[-3]
        task = os.path.basename(f)[:-5]
        if not os.path.exists(os.path.join(core.PROMPTS_DIR, domain, f"{task}.json")):
            continue
        with open(f) as fh:
            stored = json.load(fh)["instances"]
        key = "llm_correct_binary" if task.startswith("task_3") else "llm_correct"
        # Only rows about the problem the shipped prompt file describes: upstream regenerated some
        # prompt files after these results were written (task 3: 381 of 500 ground truths differ),
        # and a stored verdict about another problem says nothing about this scorer. Tasks 1, 5 and
        # 8 read nothing from the prompt file but the instance id; tasks 4 and 6 read new_instance,
        # which upstream deleted from its stored rows, so they need the query itself to match.
        reads_gt = task.startswith(("task_2", "task_3", "task_7"))
        reads_new_instance = task.startswith(("task_4", "task_6"))
        rows = []
        for r in stored:
            if key not in r or not str(r.get("llm_raw_response", "")).strip():
                continue
            try:
                inst = prompt_instance(domain, task, r["instance_id"])
            except KeyError:
                continue
            same_query = inst["query"] == r.get("query")
            same_gt = str(inst["ground_truth_plan"]) == str(r.get("ground_truth_plan"))
            if (same_query or (reads_gt and same_gt)
                    or not (reads_gt or reads_new_instance)):
                rows.append(r)
        if not rows:
            print(f"{domain + '/' + task:55s}    0  (no stored row matches a shipped prompt; "
                  f"{len(stored)} stored)")
            continue
        agree = {m: 0 for m in EXTRACTION_MODES}
        ours_correct = 0
        for r in rows:
            for m in EXTRACTION_MODES:
                try:
                    c = score_row(domain, task, r["instance_id"], r["llm_raw_response"], m)["correct"]
                except RuntimeError:
                    c = -1
                agree[m] += int(c == int(bool(r[key])))
                if m == "lenient":
                    ours_correct += max(c, 0)
        n = len(rows)
        up_acc = sum(bool(r[key]) for r in rows) / n if n else 0
        print(f"{domain + '/' + task:55s} {n:4d} {agree['upstream']:>13d} ({agree['upstream'] / n:.3f}) "
              f"{agree['lenient']:>6d} ({agree['lenient'] / n:.3f}) {up_acc:7.3f} {ours_correct / n:12.3f}"
              f"   [{len(stored) - n} stored rows skipped]")


def main():
    ap = argparse.ArgumentParser(description="Score stored PlanBench rows offline (VAL + upstream parsers)")
    ap.add_argument("model_dir", nargs="?", help="a PLAN_<Model> folder")
    ap.add_argument("--model", help="model id, as the runner was given it")
    ap.add_argument("--sample", default="pilot10", choices=sorted(core.SAMPLES))
    ap.add_argument("--smoke", action="store_true", help="score smoke/ instead of the sample's results")
    ap.add_argument("--extraction", default="lenient", choices=EXTRACTION_MODES)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.model_dir and args.model):
        ap.error("model_dir and --model are required unless --selftest")
    _, problems = score_results(os.path.abspath(args.model_dir), args.model, args.sample,
                                args.smoke, args.extraction)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
