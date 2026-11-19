"""Tests for bbh's `--sample` path, and the proof that without it nothing changed.

    python Tasks_benchmarks/bbh/test_bbh_sample10.py      # a Python with openai + httpx for C and D
    BBH_TEST_PARTS=BD python ...                          # only some parts

The last line says which parts ran. **"ALL PASSED" means every part ran**; "PASSED ... SKIPPED"
means some did not, and is not the full proof.

**Part A — the legacy path is byte-identical to BASELINE. Local only:** it reads BASELINE's files
from this repo's git history, which Quest's copy does not have, so outside a git checkout it is
skipped with the reason printed. Two sandboxes are built from the same data: one with the core and
the five selected runners as they were at BASELINE (the commit before the sample work), one with the
working tree's. Every runner is driven WITHOUT `--sample` through a stub `openai`/`together` client
that answers deterministically from the prompt — transient errors, empty replies and rows that fail
every attempt mixed in — serial and threaded, sharded, `--limit`, resume, and two refusals. The two
sandboxes must agree on every file written, every request (order too, when serial), every client
built (serial runs only: the legacy lazy client is built by a race under --workers), stdout and
stderr (serial), and the exit code.

**Part B — the sample path's guarantees**, in-process with fake calls: the sample's ids and nothing
else; nothing under `results/` read or written; every per-row field on every row; `n_attempts` and
`latency_s` true under retries; `billed_*` summing every call, CutRetry stages once each, with the
provider cost fields (the DocVQA reviewer's 16,434-token case); a failed or unfinished row held
through a resubmit — even one killed before reaching it — and its replacement carrying the whole
history (`prior_n_attempts`, `prior_failed_runs`); `attempts_without_usage` counting calls; a status
code matched as a word; separate smoke and real halt markers; the thinking-off halt at the first
offending row; `parse_ok` never False on a row scored 1; the refusals; `--limit`; shards.

**Part C — the watchdog against a blocked socket**: a local server that goes silent, or trickles a
byte every 0.5 s (which defeats the SDK's per-read timeout), over HTTP and TLS, through the real
openai SDK; the SIGALRM watchdog must end every attempt.

**Part D — the five real sample runners, end to end**: each run as `__main__` in a sandbox copy,
through the real SDK against a local server playing scripted replies (cut at the cap, silent,
reasoning that should not be there); the server's request count checks the call accounting.
"""

import contextlib
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types

BBH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(BBH))
BASELINE = "1cc5bbd5"     # the last commit before the sample work (WP1, 2026-11-15)
RUNNERS = {
    "BBH_GPT_5.6_Luna": "gpt56luna_bbh_eval.py",
    "BBH_Gemini_Flash3.5lite_OpenRouter": "gemini35lite_bbh_eval.py",
    "BBH_Qwen": "qwen_bbh_eval.py",
    "BBH_Gemma": "gemma_bbh_eval.py",
    "BBH_Deepseek": "deepseek_bbh_eval.py",
}

# ======================================================================= Part A: the legacy path

STUB = r'''
import hashlib, json, os, runpy, sys, threading, time, types
LOG = os.environ["STUB_LOG"]
DUPS = json.load(open(os.environ["STUB_DUPS"]))
time.sleep = lambda s: None              # the legacy retry backs off 2-16 s on an empty reply
_lock, _tries = threading.Lock(), {}

def record(kind, payload):
    with _lock, open(LOG, "a") as fh:
        fh.write(json.dumps([kind, payload], sort_keys=True, default=repr) + "\n")

def reply(messages):
    prompt = messages[-1]["content"]
    h = int(hashlib.sha1(prompt.encode()).hexdigest(), 16)
    if prompt == "Reply with: ok":
        return "ok", h
    unique = not any(d in prompt for d in DUPS)      # attempt-dependent only where order cannot matter
    with _lock:
        n = _tries[prompt] = _tries.get(prompt, 0) + 1
    if h % 59 == 0 and not os.environ.get("STUB_NO_HARD_FAIL"):
        raise RuntimeError("stub: this item fails every attempt")
    if h % 53 == 0:
        return "", h                                  # empty at HTTP 200, every attempt
    if unique and h % 37 == 0 and n == 1:
        raise RuntimeError("stub: transient")
    if unique and h % 41 == 0 and n <= 2:
        return "", h
    q = prompt.split("Question:", 1)[-1].split("Please show", 1)[0].split()
    word = q[-1] if q else "x"
    return ["Reasoning.\nFinal Answer: (A)", "Final Answer: True", "Final Answer: No",
            "I think it is %s" % word, "Final Answer: **%s**" % word, "Final Answer: 8 apples",
            "Final Answer: valid"][h % 7], h

class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)

class Completions:
    def __init__(self, ctor):
        self.ctor = ctor
    def create(self, **kw):
        record("create", dict(kw, _client=self.ctor))
        text, h = reply(kw["messages"])
        msg = NS(content=text, reasoning_content=None)
        usage = NS(prompt_tokens=100 + h % 50, completion_tokens=10 + h % 90,
                   completion_tokens_details=NS(reasoning_tokens=h % 7))
        return NS(choices=[NS(message=msg, finish_reason="stop")], usage=usage, model="stub",
                  provider=["Google", "Google AI Studio"][h % 2])

class Client:
    def __init__(self, **kw):
        ctor = {k: v for k, v in kw.items() if k != "api_key"}
        ctor["api_key_given"] = kw.get("api_key") is not None
        record("client", ctor)
        self.chat = NS(completions=Completions(ctor))

for name, attr in (("openai", "OpenAI"), ("together", "Together")):
    mod = types.ModuleType(name)
    setattr(mod, attr, Client)
    sys.modules[name] = mod
runner = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(runner, run_name="__main__")
'''


def git_show(rev, path):
    return subprocess.run(["git", "-C", REPO, "show", f"{rev}:{path}"], check=True,
                          capture_output=True).stdout


def build_sandbox(root, version):
    """A bbh root holding the core and the five runners at `version` (BASELINE or the working tree),
    with data/, dotenv/ and the sample file linked from the real tree. No .env: keys stay unread."""
    os.makedirs(root)
    for name in ("data", "dotenv", "sample10.json"):
        os.symlink(os.path.join(BBH, name), os.path.join(root, name))
    files = ["bbh_eval_core.py"] + [f"{d}/{f}" for d, f in RUNNERS.items()]
    for rel in files:
        dest = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if version == "working":
            shutil.copy2(os.path.join(BBH, rel), dest)
        else:
            with open(dest, "wb") as fh:
                fh.write(git_show(version, f"Tasks_benchmarks/bbh/{rel}"))


def tree(root):
    """Every file a run wrote under the model folders: {relative path: sha1}."""
    out = {}
    for folder in RUNNERS:
        base = os.path.join(root, folder)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for f in filenames:
                p = os.path.join(dirpath, f)
                if p.endswith(".py"):
                    continue
                with open(p, "rb") as fh:
                    out[os.path.relpath(p, root)] = hashlib.sha1(fh.read()).hexdigest()
    return out


def run_stub(work, root, folder, args, extra_env=None):
    log = os.path.join(work, f"log_{os.path.basename(root)}.jsonl")
    if os.path.exists(log):
        os.remove(log)
    stub = os.path.join(work, "stub.py")
    env = dict(os.environ, STUB_LOG=log, STUB_DUPS=os.path.join(work, "dups.json"),
               PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", **(extra_env or {}))
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "DEEPINFRA_API_KEY", "DEEPSEEK_API_KEY",
                "TOGETHER_API_KEY"):
        env.pop(key, None)
    p = subprocess.run([sys.executable, "-B", stub,
                        os.path.join(root, folder, RUNNERS[folder]), *args],
                       capture_output=True, text=True, env=env, cwd=work)
    calls = []                    # no log: the run made no request (e.g. the resume guard refused)
    if os.path.exists(log):
        with open(log) as fh:
            calls = [line for line in fh]
    return p.returncode, p.stdout + "\n--stderr--\n" + p.stderr, calls


# (folder, args, serial?[, env]) — run in this order in BOTH sandboxes, so resume steps see the same
# state. The threaded Gemini case runs with no row that raises on every attempt: on such a row the
# legacy runner writes `backend=` from a thread-local holding whatever that worker thread answered
# LAST, so the baseline disagrees with itself from run to run (8 baseline runs gave two values on one
# row, 2026-11-15). That is a legacy bug, not a difference; the serial Gemini case covers the row.
LEGACY_CASES = [
    ("BBH_GPT_5.6_Luna", ["--task", "all"], True),
    ("BBH_GPT_5.6_Luna", ["--task", "all", "--workers", "5", "--prompt", "v2", "--total-shards", "2",
                          "--shard", "1"], False),
    ("BBH_Gemini_Flash3.5lite_OpenRouter", ["--task", "all"], True),
    ("BBH_Gemini_Flash3.5lite_OpenRouter", ["--task", "causal_judgement,navigate", "--workers", "5",
                                            "--total-shards", "3", "--shard", "0"], False,
     {"STUB_NO_HARD_FAIL": "1"}),
    ("BBH_Qwen", ["--task", "all", "--sleep", "0"], True),
    ("BBH_Qwen", ["--provider", "deepinfra", "--task", "dyck_languages,web_of_lies", "--sleep", "0",
                  "--total-shards", "2", "--shard", "1"], True),
    ("BBH_Gemma", ["--provider", "deepinfra", "--task", "all", "--sleep", "0", "--workers", "5"], False),
    ("BBH_Gemma", ["--task", "word_sorting", "--sleep", "0"], True),
    ("BBH_Deepseek", ["--task", "all"], True),
    ("BBH_Deepseek", ["--task", "all", "--limit", "3"], True),
    # resume: the second run reads the first's checkpoint; the third changes the prompt and must be
    # refused by the guard in both
    ("BBH_Deepseek", ["--task", "boolean_expressions,penguins_in_a_table"], True),
    ("BBH_Deepseek", ["--task", "boolean_expressions,penguins_in_a_table", "--prompt", "v2"], True),
]


def baseline_unavailable():
    """Why Part A cannot run here, or None. It needs this repo's git history (BASELINE's files):
    Quest's copy has none, and a git that cannot run (an x86 Python on an arm64 Mac) is the same."""
    try:
        p = subprocess.run(["git", "-C", REPO, "rev-parse", "--verify", "--quiet", BASELINE + "^{commit}"],
                           capture_output=True, text=True)
    except OSError as error:
        return f"git cannot be run: {error}"
    if p.returncode != 0:
        return (f"commit {BASELINE} is not reachable from {REPO} — not a git checkout with this "
                f"repo's history ({(p.stderr or p.stdout).strip()[:160] or 'git rev-parse failed'})")
    return None


def test_legacy_path_identical():
    """Returns False, having printed why, when Part A cannot run here (local-only: it needs git)."""
    why = baseline_unavailable()
    if why:
        print(f"  SKIPPED — Part A is local-only: {why}")
        return False
    work = tempfile.mkdtemp(prefix="bbh_legacy_")
    try:
        with open(os.path.join(work, "stub.py"), "w") as fh:
            fh.write(STUB)
        dups = set()
        for f in os.listdir(os.path.join(BBH, "data")):
            seen = set()
            for ex in json.load(open(os.path.join(BBH, "data", f)))["examples"]:
                (dups.add if ex["input"] in seen else seen.add)(ex["input"])
        json.dump(sorted(dups), open(os.path.join(work, "dups.json"), "w"))
        old, new = os.path.join(work, "old"), os.path.join(work, "new")
        build_sandbox(old, BASELINE)
        build_sandbox(new, "working")
        for folder, args, serial, *env in LEGACY_CASES:
            env = env[0] if env else None
            rc_o, out_o, calls_o = run_stub(work, old, folder, args, env)
            rc_n, out_n, calls_n = run_stub(work, new, folder, args, env)
            label = f"{folder} {' '.join(args)}"
            assert rc_o == rc_n, f"{label}: exit {rc_o} at baseline, {rc_n} now"
            if not serial:
                # Threaded: compare the requests as a multiset, and leave out client constructions.
                # The legacy Qwen/Gemma build their client lazily with no lock, so under --workers
                # the BASELINE itself builds 1 to 5 of them from run to run (reviewer, 2026-11-15:
                # [4, 5, 3, 1, 5, 1, 3, 1] over eight identical runs) — a race, not a difference.
                calls_o = sorted(c for c in calls_o if not c.startswith('["client"'))
                calls_n = sorted(c for c in calls_n if not c.startswith('["client"'))
            assert calls_o == calls_n, f"{label}: the requests differ"
            if serial:
                assert out_o.replace(old, "<root>") == out_n.replace(new, "<root>"), \
                    f"{label}: stdout/stderr differ"
            t_o, t_n = tree(old), tree(new)
            assert t_o == t_n, (f"{label}: written files differ: "
                                f"{sorted(set(t_o.items()) ^ set(t_n.items()))[:4]}")
            print(f"  identical: {label:<100} exit={rc_n} stub_log={len(calls_n):>5} "
                  f"files={len(t_n)}", flush=True)   # stub_log: requests (+ clients built, serial)
        assert not any("results_sample10" in p or "smoke_sample10" in p for p in tree(new))
        return True
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ======================================================================= Part B: the sample path

sys.path.insert(0, BBH)
import bbh_eval_core as core  # noqa: E402

MODEL = "fake/model-1.0"
SLUG = core.model_slug(MODEL)
SAMPLE = core.load_sample("sample10")
DATA = {t: core.load_task(t) for t in core.TASKS}
GOLD = {ex["input"]: ex["target"] for t in core.TASKS for ex in DATA[t]}
ALL_ROWS = []          # every row any Part B test wrote: checked against the parse_ok invariant


def question_of(messages):
    user = next(m["content"] for m in messages if m["role"] == "user")
    return user.split("Question: ", 1)[1].split("\n\n    Please show", 1)[0]


def h(text):
    return int(hashlib.sha1(text.encode()).hexdigest(), 16)


class FakeClient:
    """negotiate_pinned's probe target: accepts everything."""

    def __init__(self):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: None))


def reset():
    core._recent.clear()
    core._consecutive[0] = 0
    for k in core.STATS:
        core.STATS[k] = 0
    core.EMPTY_RETRY_WAIT = 0.0
    core.retry_delay = lambda error, attempt, default=5.0: 0.0


def run(model_dir, once, extra_config=None, **kw):
    reset()
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return core.run_sample(model_dir, MODEL, client=FakeClient(), once=once,
                                   wanted={"max_tokens": 64, "seed": 42}, sample="sample10",
                                   extra_config=extra_config or {"route": "fake"}, **kw)
        finally:
            for root in ("results_sample10", "smoke_sample10"):
                ALL_ROWS.extend(r for rs in rows_of(os.path.join(model_dir, root)).values() for r in rs)


def exit_code(fn):
    """The process exit code `fn` would produce: its return value, or a SystemExit's code."""
    try:
        return fn()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


def rows_of(root, tasks=None):
    out = {}
    for task in tasks or core.TASKS:
        d = os.path.join(root, task)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith(".jsonl"):
                out.setdefault(task, []).extend(json.loads(x) for x in open(os.path.join(d, f)))
    return out


def snapshot(path):
    out = {}
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            p = os.path.join(dirpath, f)
            st = os.stat(p)
            out[p] = (st.st_mtime_ns, open(p, "rb").read())
    return out


def overall_last(root):
    import csv
    rows = list(csv.DictReader(open(os.path.join(root, f"{SLUG}_bbh_overall.csv"))))
    return rows[-1]


def good_once(messages, model, params):
    q = question_of(messages)
    k = h(q)
    text = f"Some reasoning.\nFinal Answer: {GOLD[q]}" if k % 4 else "no marker here"
    return text, {"finish_reason": "stop", "prompt_tokens": len(messages[-1]["content"]),
                  "completion_tokens": k % 500 + 1,
                  "reasoning_tokens": None if k % 3 == 0 else k % 50, "served_model": "fake-served"}


def test_sample_serial_and_results_untouched():
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    try:
        # a results/ tree with rows under ANOTHER config: read, it would be refused; written, it changes
        planted = os.path.join(md, "results", "navigate", f"{SLUG}.jsonl")
        os.makedirs(os.path.dirname(planted))
        with open(planted, "w") as fh:
            fh.write(json.dumps({"idx": 0, "model_response": "x", "config": "other=1"}) + "\n")
        before = snapshot(os.path.join(md, "results"))
        assert run(md, good_once) == 0
        assert snapshot(os.path.join(md, "results")) == before, "results/ was touched"
        root = os.path.join(md, "results_sample10")
        rows = rows_of(root)
        assert sum(map(len, rows.values())) == 484
        for task, rs in rows.items():
            assert [r["idx"] for r in rs] == SAMPLE[task], task
            for r in rs:
                ex = DATA[task][r["idx"]]
                assert r["question"] == ex["input"] and r["gold_answer"] == ex["target"]
                for key in core.ROW_KEYS + core.PRIOR_KEYS + ("parse_ok", "failed", "config", "ts",
                                                              "sample"):
                    assert key in r, (task, r["idx"], key)
                assert "cost" not in r, "the per-call cost is last_call_cost now"
                assert (r["prior_n_attempts"], r["prior_failed_runs"], r["resumed_after_failure"]) \
                    == (0, 0, False)
                assert (r["halted"], r["think_tag_calls"], r["reasoning_chars"]) == (None, 0, 0)
                # one attempt: what was billed is what the row says
                assert (r["billed_prompt_tokens"], r["billed_completion_tokens"],
                        r["billed_reasoning_tokens"], r["attempts_without_usage"]) == \
                    (r["prompt_tokens"], r["completion_tokens"], r["reasoning_tokens"], 0)
                assert (r["billed_cost"], r["billed_estimated_cost"]) == (None, None)
                assert r["n_attempts"] == 1 and isinstance(r["latency_s"], float)
                assert r["score"] == core.score_response(r["model_response"], ex["target"], ex["input"])
                assert r["parse_ok"] == bool(r["final_answer"].strip()) and r["failed"] == 0
                k = h(ex["input"])
                assert r["reasoning_tokens"] == (None if k % 3 == 0 else k % 50)
                assert r["config"]["params"] == {"max_tokens": 64, "seed": 42}
                assert r["config"]["prompt"] == "v1" and r["config"]["route"] == "fake"
        last = overall_last(root)
        assert last["complete"] == "1" and last["n"] == "484" and last["failed"] == "0", last
        assert os.path.exists(os.path.join(root, f"{SLUG}_negotiated_params.json"))
        assert not os.path.exists(os.path.join(md, "smoke_sample10"))
        calls = []
        assert run(md, lambda m, mo, p: calls.append(1) or good_once(m, mo, p)) == 0 and not calls
        held = snapshot(root)
        try:
            run(md, good_once, prompt_version="v2")
        except SystemExit as e:
            assert "refusing to resume" in str(e)
        else:
            raise AssertionError("a prompt change was resumed across")
        assert snapshot(root) == held, "a refused resume wrote into the sample directory"
        print("  ok: 484 rows = the sample's ids; every field on every row; results/ untouched; "
              "resume calls nothing; a prompt change is refused with nothing written")
    finally:
        shutil.rmtree(md, ignore_errors=True)


class FakeClock:
    """Stands in for the core's `time` module: `time()` reads a counter that only the test moves, and
    `sleep()` advances it instead of waiting. Latency is then exact, whatever the machine's load —
    a wall-clock tolerance failed once in five runs while a smoke shared the CPU."""

    def __init__(self):
        self.now = 1_000_000.0
        self.strftime = time.strftime

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


def test_retries_keep_meta_true_and_bill_every_call():
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    seen = {}
    tasks = ["navigate", "web_of_lies", "word_sorting"]
    clock = FakeClock()

    def plan(q):
        k = h(q)
        return k % 3, 0.01 * (1 + k % 4)          # (failed attempts first, the answer's duration)

    def once(messages, model, params):
        q = question_of(messages)
        entry = seen.setdefault(id(messages), [messages, 0])   # kept alive: its id is never reused
        entry[1] += 1
        n = entry[1]
        fails, dur = plan(q)
        if n <= fails:
            clock.now += 7.0           # a failed attempt: any latency_s that includes one is >= 7
            if n % 2:
                raise RuntimeError("fake transient")
            return "", {"finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 1000,
                        "reasoning_tokens": 900}
        clock.now += dur
        return f"Final Answer: {GOLD[q]}", {"finish_reason": "stop", "prompt_tokens": 10,
                                            "completion_tokens": h(q) % 1000,
                                            "reasoning_tokens": None,
                                            "served_model": "m-" + hashlib.sha1(q.encode()).hexdigest()[:10]}
    real_time = core.time
    try:
        core.time = clock
        assert run(md, once, tasks=tasks) == 0
        retried = 0
        for task, rs in rows_of(os.path.join(md, "results_sample10"), tasks).items():
            assert [r["idx"] for r in rs] == SAMPLE[task]
            for r in rs:
                q = r["question"]
                fails, dur = plan(q)
                empties, raised = fails // 2, fails - fails // 2     # attempts 1, 3 raise; 2 is empty
                assert r["n_attempts"] == fails + 1, (task, r["idx"], r["n_attempts"], fails)
                assert r["latency_s"] == round(dur, 2), (r["latency_s"], dur, fails)
                assert r["completion_tokens"] == h(q) % 1000            # the answering attempt's
                assert r["served_model"] == "m-" + hashlib.sha1(q.encode()).hexdigest()[:10]
                assert r["billed_prompt_tokens"] == 10 * (empties + 1)
                assert r["billed_completion_tokens"] == 1000 * empties + h(q) % 1000
                assert r["billed_reasoning_tokens"] == (900 * empties if empties else None)
                assert r["attempts_without_usage"] == raised
                assert r["score"] == 1 and r["failed"] == 0
                retried += fails > 0
        print(f"  ok: {retried} rows retried — n_attempts exact, latency_s exactly the answering "
              f"attempt's on a fake clock (failed attempts excluded), tokens the answering attempt's, "
              f"billed_* every call's")
    finally:
        core.time = real_time
        shutil.rmtree(md, ignore_errors=True)


def test_billed_usage_and_cost_the_reviewers_case():
    """Attempt 1 raises (no usage); attempt 2 is HTTP 200, empty, 16,384 reasoning tokens billed at a
    cost; attempt 3 is empty with no usage object at all; attempt 4 answers in 50. The row says 50 —
    and billed 16,434, at both priced calls' cost, with 2 calls that reported no usage."""
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    task = "navigate"
    target = DATA[task][SAMPLE[task][0]]["input"]
    tries = {}

    def once(messages, model, params):
        q = question_of(messages)
        if q != target:
            return good_once(messages, model, params)
        tries[q] = tries.get(q, 0) + 1
        if tries[q] == 1:
            raise RuntimeError("Error code: 500 - upstream reset, request id req_7f401ab")
        if tries[q] == 2:
            return "", {"finish_reason": "length", "prompt_tokens": 120, "completion_tokens": 16384,
                        "reasoning_tokens": 16384, "last_call_cost": 0.5}
        if tries[q] == 3:
            return "", {"finish_reason": "stop"}             # HTTP 200, empty, no usage object
        return f"Final Answer: {GOLD[q]}", {"finish_reason": "stop", "prompt_tokens": 120,
                                            "completion_tokens": 50, "reasoning_tokens": 0,
                                            "last_call_cost": 0.01}
    try:
        assert run(md, once, tasks=[task], limit=1) == 0
        r = rows_of(os.path.join(md, "smoke_sample10"), [task])[task][0]
        got = {k: r[k] for k in ("n_attempts", "completion_tokens", "reasoning_tokens",
                                 "billed_prompt_tokens", "billed_completion_tokens",
                                 "billed_reasoning_tokens", "attempts_without_usage", "last_call_cost",
                                 "billed_cost", "billed_estimated_cost")}
        assert got == {"n_attempts": 4, "completion_tokens": 50, "reasoning_tokens": 0,
                       "billed_prompt_tokens": 240, "billed_completion_tokens": 16434,
                       "billed_reasoning_tokens": 16384, "attempts_without_usage": 2,
                       "last_call_cost": 0.01, "billed_cost": 0.51, "billed_estimated_cost": None}, got
        # the 500's request id holds "401": a substring test called it an auth failure and halted
        assert not core.is_fatal_request(RuntimeError("Error code: 500 - request id req_7f401ab"))
        assert core.is_fatal_request(RuntimeError("Error code: 401 - Incorrect API key provided"))
        assert core.is_fatal_request(RuntimeError("Error code: 401 - unauthorized"))
        assert not core.has_status_code("req_95003ab", core._TRANSIENT_CODES)
        assert core.has_status_code("Error code: 503 - overloaded", core._TRANSIENT_CODES)
        # what the SDK hands back: OpenRouter's usage.cost and DeepInfra's usage.estimated_cost
        resp = types.SimpleNamespace(
            choices=[types.SimpleNamespace(finish_reason="stop")], model="m",
            usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=2, cost=0.003,
                                        estimated_cost=0.0004, completion_tokens_details=None))
        meta = core.openai_meta(resp)
        assert (meta["last_call_cost"], meta["last_call_estimated_cost"]) == (0.003, 0.0004)
        assert "cost" not in meta and "estimated_cost" not in meta
        print("  ok: the reviewer's case — completion_tokens 50, billed_completion 16434, billed_cost "
              "0.51 over both priced calls, 2 calls without usage (one raised, one had none); "
              "openai_meta reads usage.cost / usage.estimated_cost as last_call_*; '401' in a request "
              "id is not fatal")
    finally:
        shutil.rmtree(md, ignore_errors=True)


def _failing_once(bad_qs, answer_bad):
    """The reviewer's resume probe: a bad row is HTTP 200, empty, at the cap on every attempt —
    16,384 reasoning tokens a call — unless `answer_bad`; every other row answers in 40."""
    calls = []

    def once(messages, model, params):
        q = question_of(messages)
        calls.append(q)
        if q in bad_qs and not answer_bad:
            return "", {"finish_reason": "length", "prompt_tokens": 120, "completion_tokens": 16384,
                        "reasoning_tokens": 16384}
        return f"Final Answer: {GOLD[q]}", {"finish_reason": "stop", "prompt_tokens": 120,
                                            "completion_tokens": 40, "reasoning_tokens": 30}
    return once, calls


def test_resume_keeps_a_failed_rows_history():
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    task = "navigate"
    bad = SAMPLE[task][2]
    bad_q = DATA[task][bad]["input"]
    path = os.path.join(md, "results_sample10")

    def the_row():
        return next(r for r in rows_of(path, [task])[task] if r["idx"] == bad)
    try:
        once, _ = _failing_once({bad_q}, answer_bad=False)
        assert run(md, once, tasks=[task]) == 1
        r = the_row()
        assert (r["failed"], r["n_attempts"], r["billed_completion_tokens"]) == (1, 5, 81920)
        once, _ = _failing_once({bad_q}, answer_bad=False)          # failed again on resubmit
        assert run(md, once, tasks=[task]) == 1
        r = the_row()
        assert (r["failed"], r["n_attempts"], r["prior_n_attempts"], r["prior_failed_runs"],
                r["billed_completion_tokens"]) == (1, 5, 5, 1, 163840), r
        once, calls = _failing_once({bad_q}, answer_bad=True)
        assert run(md, once, tasks=[task]) == 0
        assert calls == [bad_q], "resume must call exactly the failed row"
        r = the_row()
        got = {k: r[k] for k in ("failed", "n_attempts", "prior_n_attempts", "prior_failed_runs",
                                 "resumed_after_failure", "completion_tokens",
                                 "billed_prompt_tokens", "billed_completion_tokens",
                                 "billed_reasoning_tokens", "attempts_without_usage")}
        assert got == {"failed": 0, "n_attempts": 1, "prior_n_attempts": 10, "prior_failed_runs": 2,
                       "resumed_after_failure": True, "completion_tokens": 40,
                       "billed_prompt_tokens": 11 * 120, "billed_completion_tokens": 10 * 16384 + 40,
                       "billed_reasoning_tokens": 10 * 16384 + 30, "attempts_without_usage": 0}, got
        assert len(rows_of(path, [task])[task]) == len(SAMPLE[task])
        print("  ok: replaced after two failed runs — n_attempts 1 (this run), prior_n_attempts 10, "
              "prior_failed_runs 2, billed_completion 163,880 = every call of all three runs")
    finally:
        shutil.rmtree(md, ignore_errors=True)


def test_killed_mid_resubmit_keeps_held_rows():
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    task = "navigate"
    a, b = SAMPLE[task][2], SAMPLE[task][5]
    qa, qb = DATA[task][a]["input"], DATA[task][b]["input"]
    path = os.path.join(md, "results_sample10")
    orig = core.sample_row

    def killer(task_, idx, *args, **kw):
        if idx == b:
            raise SystemExit(143)                 # what the SIGTERM handler raises, mid-loop
        return orig(task_, idx, *args, **kw)
    try:
        once, _ = _failing_once({qa, qb}, answer_bad=False)
        assert run(md, once, tasks=[task]) == 1
        once, _ = _failing_once({qa, qb}, answer_bad=True)
        core.sample_row = killer
        try:
            assert exit_code(lambda: run(md, once, tasks=[task])) == 143
        finally:
            core.sample_row = orig
        rows = {r["idx"]: r for r in rows_of(path, [task])[task]}
        assert sorted(rows) == SAMPLE[task], "a held row was dropped by the killed resubmit"
        assert rows[a]["resumed_after_failure"] and rows[a]["billed_completion_tokens"] == 81920 + 40
        assert (rows[b]["failed"], rows[b]["n_attempts"], rows[b]["billed_completion_tokens"]) == \
            (1, 5, 81920), "the row the killed run never reached must keep its run-1 history"
        once, calls = _failing_once({qa, qb}, answer_bad=True)
        assert run(md, once, tasks=[task]) == 0 and calls == [qb]
        rb = next(r for r in rows_of(path, [task])[task] if r["idx"] == b)
        assert (rb["prior_n_attempts"], rb["prior_failed_runs"], rb["billed_completion_tokens"]) == \
            (5, 1, 81960)
        print("  ok: a resubmit killed before a failed row keeps it, with its billing; the next "
              "resubmit replaces it and folds that history in")
    finally:
        shutil.rmtree(md, ignore_errors=True)


def _file_billed(path, task):
    rows = rows_of(path, [task])[task]
    assert len(rows) == len({r["idx"] for r in rows}), "an idx appears twice in the file"
    return {k: sum(r.get(k) or 0 for r in rows) for k in ("billed_prompt_tokens",
                                                          "billed_completion_tokens")}, rows


def test_inflight_row_kept_when_killed():
    """The reviewer's case: killed (SIGTERM) during the retry stage, after the first stage's 8,192
    tokens were billed. The row must be in the file, failed, with those tokens — and its earlier
    failed run folded in. The file's billed total = every token the fake billed, across both runs."""
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    cut = core.CutRetry("CEILING", "RETRY CEILING", "FORCE NOW", 99)
    task = "navigate"
    x = SAMPLE[task][3]
    xq = DATA[task][x]["input"]
    billed_out = {"prompt": 0, "completion": 0}
    phase = {"run": 1}

    def bill(meta):
        billed_out["prompt"] += meta.get("prompt_tokens") or 0
        billed_out["completion"] += meta.get("completion_tokens") or 0
        return meta

    def call(sent, stage, q):
        if q == xq and phase["run"] == 1:             # run 1: empty at HTTP 200, every attempt
            return "", "", bill({"finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 1000})
        if q == xq and phase["run"] == 2:
            if stage == "first":
                return "part", "part", bill({"finish_reason": "length", "prompt_tokens": 100,
                                             "completion_tokens": 8192, "last_call_cost": 0.002})
            os.kill(os.getpid(), signal.SIGTERM)       # the wall clock, during the retry stage
            time.sleep(1)
            raise AssertionError("SIGTERM was not delivered")
        text = f"Final Answer: {GOLD[q]}"
        return text, text, bill({"finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 7})

    def once(messages, model, params):
        q = question_of(messages)
        return cut(messages, lambda sent, stage: call(sent, stage, q), model)
    cfg = dict(cut.config(), route="fake")
    path = os.path.join(md, "results_sample10")
    held = signal.getsignal(signal.SIGTERM)
    try:
        assert run(md, once, tasks=[task], extra_config=cfg) == 1
        phase["run"] = 2
        assert exit_code(lambda: run(md, once, tasks=[task], extra_config=cfg)) == 143
        totals, rows = _file_billed(path, task)
        assert sorted(r["idx"] for r in rows) == SAMPLE[task], "the killed row is missing"
        r = next(r for r in rows if r["idx"] == x)
        got = {k: r[k] for k in ("failed", "n_attempts", "latency_s", "billed_completion_tokens",
                                 "billed_prompt_tokens", "attempts_without_usage", "billed_cost",
                                 "prior_n_attempts", "prior_failed_runs")}
        assert got == {"failed": 1, "n_attempts": 1, "latency_s": None,
                       "billed_completion_tokens": 5 * 1000 + 8192, "billed_prompt_tokens": 600,
                       "attempts_without_usage": 1, "billed_cost": 0.002,
                       "prior_n_attempts": 5, "prior_failed_runs": 1}, got
        assert r["error"].startswith("interrupted: SystemExit"), r["error"]
        assert r["halted"] == "SystemExit: 143", r["halted"]
        assert totals == {"billed_prompt_tokens": billed_out["prompt"],
                          "billed_completion_tokens": billed_out["completion"]}, (totals, billed_out)
        print(f"  ok: SIGTERM in the retry stage — the row is written failed with the first stage's "
              f"8,192 tokens and run 1's 5,000 folded in, 1 call without usage; the file bills "
              f"{totals['billed_completion_tokens']:,} completion tokens = what the fake billed")
    finally:
        signal.signal(signal.SIGTERM, held)
        shutil.rmtree(md, ignore_errors=True)


def test_failure_halt_row_is_written():
    """The row whose failure trips FAILURE_HALT is written, with every attempt it paid for."""
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    task = "navigate"
    bad = set(SAMPLE[task][2:5])
    bad_q = {DATA[task][i]["input"] for i in bad}
    billed_out = {"completion": 0}

    def once(messages, model, params):
        q = question_of(messages)
        if q in bad_q:
            billed_out["completion"] += 500
            return "", {"finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 500}
        billed_out["completion"] += 7
        return f"Final Answer: {GOLD[q]}", {"finish_reason": "stop", "prompt_tokens": 10,
                                            "completion_tokens": 7}
    limit = core.CONSECUTIVE_FAILURE_LIMIT
    path = os.path.join(md, "results_sample10")
    try:
        core.CONSECUTIVE_FAILURE_LIMIT = 3
        try:
            run(md, once, tasks=[task])
        except SystemExit as e:
            assert "failure" in str(e).lower(), e
        else:
            raise AssertionError("three failed rows in a row did not halt")
        assert os.path.exists(os.path.join(md, "sample10_FAILURE_HALT.txt"))
        totals, rows = _file_billed(path, task)
        assert [r["idx"] for r in rows] == SAMPLE[task][:5], [r["idx"] for r in rows]
        last = rows[-1]
        assert (last["idx"], last["failed"], last["n_attempts"], last["billed_completion_tokens"]) == \
            (SAMPLE[task][4], 1, 5, 2500), last
        assert totals["billed_completion_tokens"] == billed_out["completion"], (totals, billed_out)
        print(f"  ok: FAILURE_HALT — the row that tripped it is written failed with its 5 attempts "
              f"(2,500 tokens); the file bills {totals['billed_completion_tokens']:,} = what the fake billed")
    finally:
        core.CONSECUTIVE_FAILURE_LIMIT = limit
        shutil.rmtree(md, ignore_errors=True)


def test_a_call_is_counted_once():
    """HTTP 200 with usage but no `choices`: openai_meta books the bill, then choices[0] raises.
    The call is billed, and not counted again as a call without usage."""
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    task = "navigate"
    target = DATA[task][SAMPLE[task][0]]["input"]
    tries = {}

    def response(content, completion, choices=True):
        msg = types.SimpleNamespace(content=content, reasoning_content=None)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg, finish_reason="stop")] if choices else [],
            model="m", usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=completion,
                                                   completion_tokens_details=None))

    def once(messages, model, params):
        q = question_of(messages)
        tries[q] = tries.get(q, 0) + 1
        r = response(f"Final Answer: {GOLD[q]}", 40, choices=not (q == target and tries[q] == 1))
        meta = core.openai_meta(r)                    # books first, as every runner now does
        return r.choices[0].message.content, meta     # IndexError on the empty one
    try:
        assert run(md, once, tasks=[task], limit=1) == 0
        r = rows_of(os.path.join(md, "smoke_sample10"), [task])[task][0]
        got = {k: r[k] for k in ("n_attempts", "billed_completion_tokens", "attempts_without_usage")}
        assert got == {"n_attempts": 2, "billed_completion_tokens": 80, "attempts_without_usage": 0}, got
        tagged = core.openai_meta(response("<think>weighing it</think>Final Answer: Yes", 5))
        assert (tagged["think_tag"], tagged["reasoning_chars"]) == (1, 0), tagged
        assert core.openai_meta(response("Final Answer: Yes", 5))["think_tag"] == 0
        assert core.has_think_tag("ok</think>Final Answer: No") and core.has_think_tag("<think></think>Yes")
        assert not core.has_think_tag("I think so. Final Answer: Yes")
        print("  ok: a reply with usage and no choices is billed once and not counted as a call "
              "without usage; a <think> tag in the content is flagged per call (think_tag_calls), "
              "not counted as reasoning_chars")
    finally:
        shutil.rmtree(md, ignore_errors=True)


def test_halt_in_the_inner_handler_counts_the_call_once():
    """An ordinary error is counted by the inner handler; the MALFORMED-request halt raised from that
    same handler reaches the outer one — which must not count the call again."""
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    task = "navigate"
    target = DATA[task][SAMPLE[task][2]]["input"]

    def once(messages, model, params):
        if question_of(messages) == target:
            raise RuntimeError("Error code: 400 - Unsupported parameter: 'max_tokens'")
        return good_once(messages, model, params)
    try:
        try:
            run(md, once, tasks=[task])
        except SystemExit as e:
            assert "malformed" in str(e).lower(), e
        else:
            raise AssertionError("a malformed request did not halt")
        assert os.path.exists(os.path.join(md, "sample10_FAILURE_HALT.txt"))
        r = rows_of(os.path.join(md, "results_sample10"), [task])[task][-1]
        assert (r["idx"], r["failed"], r["n_attempts"], r["attempts_without_usage"]) == \
            (SAMPLE[task][2], 1, 1, 1), r
        assert r["halted"].startswith("SystemExit"), r["halted"]
        print("  ok: a halt raised from the inner handler after an ordinary error counts that call "
              "once (attempts_without_usage 1, not 2); the halted row is written")
    finally:
        shutil.rmtree(md, ignore_errors=True)


def test_answered_row_recovered_when_killed_while_written():
    """The call returned an answer; the kill lands while the row is being built. The row must reach
    the file ANSWERED, its earlier failed run folded in, and the file must bill what was billed."""
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    task = "navigate"
    x = SAMPLE[task][3]
    xq = DATA[task][x]["input"]
    billed_out = {"completion": 0}
    phase = {"run": 1}

    def once(messages, model, params):
        q = question_of(messages)
        if q == xq and phase["run"] == 1:
            billed_out["completion"] += 1000
            return "", {"finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 1000}
        billed_out["completion"] += 7
        return f"Final Answer: {GOLD[q]}", {"finish_reason": "stop", "prompt_tokens": 100,
                                            "completion_tokens": 7}
    orig = core.build_row
    fired = []

    def kill_while_building(task_, idx, example, text, meta, *rest):
        if idx == x and text and not fired:
            fired.append(1)
            raise SystemExit(143)                 # SIGTERM, landing while the row is built
        return orig(task_, idx, example, text, meta, *rest)
    path = os.path.join(md, "results_sample10")
    try:
        assert run(md, once, tasks=[task]) == 1
        phase["run"] = 2
        core.build_row = kill_while_building
        try:
            assert exit_code(lambda: run(md, once, tasks=[task])) == 143 and fired
        finally:
            core.build_row = orig
        totals, rows = _file_billed(path, task)
        r = next(r for r in rows if r["idx"] == x)
        got = {k: r[k] for k in ("failed", "score", "n_attempts", "billed_completion_tokens",
                                 "prior_n_attempts", "prior_failed_runs", "resumed_after_failure")}
        assert got == {"failed": 0, "score": 1, "n_attempts": 1, "billed_completion_tokens": 5007,
                       "prior_n_attempts": 5, "prior_failed_runs": 1, "resumed_after_failure": True}, got
        assert r["model_response"] == f"Final Answer: {GOLD[xq]}" and r["latency_s"] is not None
        assert r["halted"] is None, "its call was not interrupted: it had returned an answer"
        assert totals["billed_completion_tokens"] == billed_out["completion"], (totals, billed_out)
        print("  ok: killed while an answered row was being written — the row reaches the file "
              "answered, its failed run folded in (prior 5 attempts, 5,007 tokens billed); the file "
              "bills what the fake billed")
    finally:
        shutil.rmtree(md, ignore_errors=True)


def _line_of(func, text, occurrence=1):
    """The source line number of `text` in `func` — the n-th line containing it."""
    import inspect
    lines, first = inspect.getsourcelines(func)
    hits = [first + k for k, line in enumerate(lines) if text in line]
    assert len(hits) >= occurrence, (func.__name__, text, hits)
    return hits[occurrence - 1]


def _stop_at(func_name, line, nth):
    """A sys.settrace hook raising SystemExit(143) the nth time `line` of `func_name` is about to run
    — how a SIGTERM handler lands at an eval-breaker check before that line (MMLU's B24/B25)."""
    hits = [0]

    def local(frame, event, arg):
        if event == "line" and frame.f_lineno == line:
            hits[0] += 1
            if hits[0] == nth:
                raise SystemExit(143)
        return local

    def tracer(frame, event, arg):
        return local if frame.f_code.co_name == func_name else None
    return tracer


def _stop_after(func_name, arm_line, nth, stop_line):
    """Raise SystemExit(143) at `stop_line`, the first time it runs after the nth run of `arm_line`.
    For a statement spanning several lines, whose first line the tracer reports twice per pass (on
    entry, and again when the call returns to store its result): arming on a one-line statement
    just before it counts passes, not line events."""
    state = {"hits": 0, "armed": False}

    def local(frame, event, arg):
        if event == "line":
            if frame.f_lineno == arm_line:
                state["hits"] += 1
                state["armed"] = state["hits"] == nth
            elif frame.f_lineno == stop_line and state["armed"]:
                state["armed"] = False
                raise SystemExit(143)
        return local

    def tracer(frame, event, arg):
        return local if frame.f_code.co_name == func_name else None
    return tracer


def _stopped_run(md, tracer, billed, task="navigate"):
    """One task, every row answering with a distinct bill (50 + idx) so a bill under the wrong id
    shows; a stop delivered by `tracer`. Returns the RAW lines of the task file, parsed."""
    def once(messages, model, params):
        q = question_of(messages)
        idx = next(i for i in SAMPLE[task] if DATA[task][i]["input"] == q)
        billed[0] += 50 + idx
        return f"Final Answer: {GOLD[q]}", {"finish_reason": "stop", "prompt_tokens": 1,
                                            "completion_tokens": 50 + idx}
    sys.settrace(tracer)
    try:
        exit_code(lambda: run(md, once, tasks=[task]))
    finally:
        sys.settrace(None)
    raw = open(os.path.join(md, "results_sample10", task, f"{SLUG}.jsonl")).read().splitlines()
    return [json.loads(line) for line in raw]


def test_stop_before_a_rows_call_writes_nothing_for_it():
    """S1: a stop landing just before row #3's call starts must write nothing for #3 — not a row
    carrying #2's answer or bill under #3's id."""
    md = tempfile.mkdtemp(prefix="bbh_s1_")
    task = "navigate"
    try:
        billed = [0]
        arm = _line_of(core.run_sample_tasks, "current = i")
        line = _line_of(core.run_sample_tasks, "row = sample_row(")
        assert line == arm + 1, (arm, line)
        rows = _stopped_run(md, _stop_after("run_sample_tasks", arm, 3, line), billed)
        assert [r["idx"] for r in rows] == SAMPLE[task][:2], [r["idx"] for r in rows]
        assert not any(r.get("halted") for r in rows)
        assert sum(r["billed_completion_tokens"] for r in rows) == billed[0], billed
        # call_with_retries itself: after a completed call (its result stashed), a stop at the next
        # call's entry finds no earlier result — even for a caller that does not reset
        reset()
        msgs = [{"role": "user", "content": core.PROMPTS["v1"].format(question=DATA[task][0]["input"])}]
        core.call_with_retries(good_once, msgs, "m", md)
        assert core._INFLIGHT.result is not None, "the completed call stashed nothing"
        entry = _line_of(core.call_with_retries, "require_main_thread()")
        sys.settrace(_stop_at("call_with_retries", entry, 1))
        try:
            core.call_with_retries(good_once, msgs, "m", md)
            raise AssertionError("the stop did not land")
        except SystemExit:
            sys.settrace(None)
            assert core.take_inflight() is None, "the next row would carry the previous row's result"
        finally:
            sys.settrace(None)
        print(f"  ok: S1 — a stop just before row #3's call: rows {[r['idx'] for r in rows]} on disk, "
              f"none for #3, file billed {billed[0]} = what was billed; a stop at call_with_retries' "
              f"entry finds no earlier row's result")
    finally:
        sys.settrace(None)
        shutil.rmtree(md, ignore_errors=True)


def test_stop_around_the_append_writes_the_row_once():
    """S2: a stop landing at row #3's append, or just after it, writes #3 exactly once, answered —
    counted on the raw lines of the file."""
    md = tempfile.mkdtemp(prefix="bbh_s2_")
    task = "navigate"
    try:
        append_line = _line_of(core.run_sample_tasks, "records.append(row)", 1)
        after_line = _line_of(core.run_sample_tasks, "held.pop(i, None)")
        assert after_line == append_line + 1, (append_line, after_line)
        notes = []
        for label, line in (("at the append", append_line), ("just after it", after_line)):
            sub = os.path.join(md, label.replace(" ", "_"))
            billed = [0]
            rows = _stopped_run(sub, _stop_at("run_sample_tasks", line, 3), billed)
            idxs = [r["idx"] for r in rows]
            assert idxs == SAMPLE[task][:3], (label, idxs)
            assert not rows[2]["failed"] and rows[2]["halted"] is None, rows[2]
            assert sum(r["billed_completion_tokens"] for r in rows) == billed[0], (label, billed)
            notes.append(f"{label}: {len(idxs)} raw lines, {len(set(idxs))} idx, billed {billed[0]}")
        print("  ok: S2 — a stop around row #3's append writes it once, answered — " + "; ".join(notes))
    finally:
        sys.settrace(None)
        shutil.rmtree(md, ignore_errors=True)


def test_failed_row_written_then_retried():
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    task, bad = "navigate", SAMPLE["navigate"][3]
    bad_q = DATA[task][bad]["input"]
    calls = []

    def broken(messages, model, params):
        if question_of(messages) == bad_q:
            calls.append(1)
            raise RuntimeError("fake outage for one item")
        return good_once(messages, model, params)
    try:
        assert run(md, broken, tasks=[task]) == 1, "a failed row must exit 1"
        rs = rows_of(os.path.join(md, "results_sample10"), [task])[task]
        assert [r["idx"] for r in rs] == SAMPLE[task], "the failed row was dropped"
        r = next(r for r in rs if r["idx"] == bad)
        assert (r["failed"], r["model_response"], r["n_attempts"], r["latency_s"], r["parse_ok"],
                r["score"], r["attempts_without_usage"], r["billed_completion_tokens"]) == \
            (1, "", 5, None, False, 0, 5, 0), r
        assert len(calls) == 5
        per_task = list(__import__("csv").DictReader(open(os.path.join(
            md, "results_sample10", task, f"{SLUG}_overall.csv"))))[0]
        assert per_task["failed"] == "1" and per_task["n"] == str(len(SAMPLE[task]))
        calls.clear()
        assert run(md, lambda m, mo, p: calls.append(1) or good_once(m, mo, p), tasks=[task]) == 0
        assert len(calls) == 1, "resume must call exactly the failed row"
        r = next(r for r in rows_of(os.path.join(md, "results_sample10"), [task])[task] if r["idx"] == bad)
        assert r["failed"] == 0 and r["model_response"] and r["attempts_without_usage"] == 5
        print("  ok: a row failing all 5 attempts is written failed (5 calls without usage), counted "
              "in _overall.csv, exits 1; resume calls exactly that row and keeps the 5")
    finally:
        shutil.rmtree(md, ignore_errors=True)


def test_billing_halt_and_marker_prefixes():
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    task = "word_sorting"
    stop_at = SAMPLE[task][12]
    stop_q = DATA[task][stop_at]["input"]

    def once(messages, model, params):
        if question_of(messages) == stop_q:     # DeepInfra's out-of-balance 402, verbatim wording
            raise RuntimeError("Error code: 402 - You need positive balance to do inference. "
                               "Please add balance manually or setup top-up")
        return good_once(messages, model, params)
    try:
        try:
            run(md, once, tasks=[task])
        except SystemExit as e:
            assert "billing" in str(e), e
        else:
            raise AssertionError("a billing refusal did not halt the run")
        real = os.path.join(md, "sample10_BILLING_HALT.txt")
        assert os.path.exists(real)
        rows = rows_of(os.path.join(md, "results_sample10"), [task])[task]
        assert [r["idx"] for r in rows] == SAMPLE[task][:13], [r["idx"] for r in rows]
        halted = rows[-1]                     # the row in flight when the halt fired: written failed
        assert (halted["idx"], halted["failed"], halted["n_attempts"], halted["attempts_without_usage"],
                halted["latency_s"]) == (stop_at, 1, 1, 1, None), halted
        assert halted["error"].startswith("interrupted: SystemExit"), halted["error"]
        assert halted["halted"].startswith("SystemExit: aborting"), halted["halted"]
        try:
            run(md, once, tasks=[task], limit=13)
        except SystemExit:
            pass
        assert os.path.exists(real), "a --limit run cleared the real run's halt marker"
        assert os.path.exists(os.path.join(md, "smoke_sample10_BILLING_HALT.txt"))
        assert open(real).read().count("model=") == 1
        print("  ok: a billing refusal halts at once with sample10_BILLING_HALT.txt, every finished row "
              "kept and the row in flight written failed ('interrupted'); a --limit run writes "
              "smoke_sample10_BILLING_HALT.txt and leaves the real one")
    finally:
        shutil.rmtree(md, ignore_errors=True)


def test_cut_retry_billing_calls_and_reasoning():
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    cut = core.CutRetry("CEILING", "RETRY CEILING", "FORCE NOW", 99)
    sent_log, stage_calls = [], {}

    def call(sent, stage, q):
        sent_log.append((stage, [m["role"] for m in sent], sent[0]["content"]))
        stage_calls[(q, stage)] = stage_calls.get((q, stage), 0) + 1
        stage_calls[q] = stage_calls.get(q, 0) + 1
        k = h(q) % 5
        # 0 answers first; 1 cut, retry answers; 2 forced; 3 cut, retry empty once; 4 cut, retry raises once
        if stage == "first" and k:
            return "partial " + q[:5], "partial " + q[:5], {"finish_reason": "length",
                                                            "completion_tokens": 100, "reasoning_chars": 5}
        if stage == "retry" and k == 2:
            return "partial2", "handback-" + q[:8], {"finish_reason": "length", "completion_tokens": 200,
                                                     "reasoning_chars": 7}
        if stage == "retry" and k == 3 and stage_calls[(q, stage)] == 1:
            return "", "", {"finish_reason": "stop", "completion_tokens": 30, "reasoning_chars": 7}
        if stage == "retry" and k == 4 and stage_calls[(q, stage)] == 1:
            raise RuntimeError("fake stage error")
        if stage == "forced":
            assert sent[-2] == {"role": "assistant", "content": "handback-" + q[:8]}
            assert sent[-1] == {"role": "user", "content": "FORCE NOW"}
        text = f"Final Answer: {GOLD[q]}"
        return text, text, {"finish_reason": "stop", "completion_tokens": 7,
                            "reasoning_chars": {"first": 5, "retry": 7, "forced": 0}[stage]}

    def once(messages, model, params):
        q = question_of(messages)
        return cut(messages, lambda sent, stage: call(sent, stage, q), model)
    tasks = ["web_of_lies", "navigate", "word_sorting"]
    try:
        assert run(md, once, tasks=tasks, extra_config=dict(cut.config(), route="fake")) == 0
        for stage, roles, first in sent_log:
            assert roles[:2] == ["system", "user"] and first in ("CEILING", "RETRY CEILING"), (stage, roles)
        seen = set()
        for task, rs in rows_of(os.path.join(md, "results_sample10"), tasks).items():
            for r in rs:
                q = r["question"]
                k = h(q) % 5
                seen.add(k)
                assert r["score"] == 1, r
                assert bool(r.get("cut_retry")) == (k > 0) and bool(r.get("forced_answer")) == (k == 2)
                assert r["completion_tokens"] == {0: 7, 1: 107, 2: 307, 3: 107, 4: 107}[k]
                assert r["billed_completion_tokens"] == {0: 7, 1: 107, 2: 307, 3: 137, 4: 107}[k], r
                assert r["attempts_without_usage"] == (1 if k == 4 else 0), r
                assert r["reasoning_chars"] == {0: 5, 1: 12, 2: 12, 3: 19, 4: 12}[k], r
                assert r["n_attempts"] == (2 if k in (3, 4) else 1)
                # true HTTP calls on an answered row = n_attempts + cut_retry + forced_answer
                assert stage_calls[q] == r["n_attempts"] + bool(r.get("cut_retry")) + \
                    bool(r.get("forced_answer")), (k, stage_calls[q], r["n_attempts"])
                if k:
                    assert r["first_response"] == "partial " + q[:5]
        assert seen == {0, 1, 2, 3, 4}, seen
        print("  ok: CutRetry — usage summed per attempt, a kept stage billed once, a stage that "
              "raised counted as a call without usage, reasoning_chars summed over every stage, "
              "calls = n_attempts + cut_retry + forced_answer")
    finally:
        shutil.rmtree(md, ignore_errors=True)


def test_unfinished_cut_row_exits_1_and_is_called_again():
    """DeepSeek's second call cut at the cap with content but no reasoning_content to hand back:
    CutRetry returns the content with finish `length`. Not an answer: exit 1, and resume calls it."""
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    cut = core.CutRetry("CEILING", "RETRY CEILING", "FORCE NOW", 99)
    task = "navigate"
    bad = SAMPLE[task][4]
    bad_q = DATA[task][bad]["input"]
    fixed = {"on": False}

    def call(sent, stage, q):
        if q == bad_q and not fixed["on"]:
            if stage == "first":
                return "part one", "thinking...", {"finish_reason": "length", "completion_tokens": 100,
                                                   "reasoning_chars": 11}
            return f"Final Answer: {GOLD[q]}", "", {"finish_reason": "length", "completion_tokens": 200,
                                                    "reasoning_chars": 0}
        text = f"Final Answer: {GOLD[q]}"
        return text, "", {"finish_reason": "stop", "completion_tokens": 7, "reasoning_chars": 0}

    def once(messages, model, params):
        q = question_of(messages)
        return cut(messages, lambda sent, stage: call(sent, stage, q), model)
    cfg = dict(cut.config(), route="fake")
    path = os.path.join(md, "results_sample10")
    try:
        assert run(md, once, tasks=[task], extra_config=cfg) == 1, "an unfinished row must exit 1"
        r = next(r for r in rows_of(path, [task])[task] if r["idx"] == bad)
        assert core.is_unfinished(r) and r["finish_reason"] == "length" and not r.get("forced_answer")
        assert "unfinished" in overall_last(path)["problems"] and overall_last(path)["complete"] == "0"
        fixed["on"] = True
        assert run(md, once, tasks=[task], extra_config=cfg) == 0
        r = next(r for r in rows_of(path, [task])[task] if r["idx"] == bad)
        assert not core.is_unfinished(r) and r["resumed_after_failure"] and r["prior_failed_runs"] == 1
        assert r["billed_completion_tokens"] == 300 + 7
        print("  ok: a cut row with no forced answer is unfinished — exit 1, roll-up incomplete, and "
              "resume calls it again with its 300 billed tokens folded in")
    finally:
        shutil.rmtree(md, ignore_errors=True)


def test_thinking_halts_at_the_first_row():
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    task = "navigate"
    path = os.path.join(md, "results_sample10")
    try:
        for signal_key, value in (("reasoning_chars", 812), ("reasoning_tokens", 5),
                                  ("inline_think", None)):
            shutil.rmtree(path, ignore_errors=True)
            target = DATA[task][SAMPLE[task][3]]["input"]
            calls = []

            def once(messages, model, params):
                q = question_of(messages)
                calls.append(q)
                text, meta = good_once(messages, model, params)
                meta = dict(meta, reasoning_tokens=0, reasoning_chars=0)
                if q == target and signal_key == "inline_think":
                    text = "<think>let me ponder</think>" + text
                    meta["think_tag"] = int(core.has_think_tag(text))
                elif q == target:
                    meta[signal_key] = value
                return text, meta
            code = exit_code(lambda: run(md, once, tasks=[task], thinking_off=True))
            assert code == 3, code
            assert len(calls) == 4, f"{len(calls)} calls: the run went on past the thinking row"
            marker = os.path.join(md, "sample10_THINKING_HALT.txt")
            assert os.path.exists(marker)
            rows = rows_of(path, [task])[task]
            assert [r["idx"] for r in rows] == SAMPLE[task][:4] and core.is_thinking(rows[-1])
            calls.clear()
            assert exit_code(lambda: run(md, once, tasks=[task], thinking_off=True)) == 3 and not calls, \
                "a run started while THINKING_HALT.txt exists"
            os.remove(marker)                     # deleted without reading the rows
            assert exit_code(lambda: run(md, once, tasks=[task], thinking_off=True)) == 3 and not calls, \
                "a resume over a stored thinking row called rows"
            assert os.path.exists(marker) and "found on resume" in open(marker).read(), \
                "the stored-row refusal did not rewrite THINKING_HALT.txt"
            os.remove(marker)
        shutil.rmtree(path, ignore_errors=True)
        code = exit_code(lambda: run(md, once, tasks=[task], thinking_off=True, limit=5))
        assert code == 3 and os.path.exists(os.path.join(md, "smoke_sample10_THINKING_HALT.txt"))
        print("  ok: thinking_off — reasoning_chars > 0 or reasoning_tokens > 0 halts at that row "
              "(4 calls, not 25) with sample10_THINKING_HALT.txt and exit 3; a resubmit on the same "
              "config makes 0 calls and exits 3, with the marker present or deleted (then rewritten); "
              "smoke uses smoke_sample10_")
    finally:
        shutil.rmtree(md, ignore_errors=True)


def test_parse_ok_agrees_with_the_score():
    checked = 0
    for task in core.TASKS:
        for idx in SAMPLE[task][:6]:
            ex = DATA[task][idx]
            gold, q = ex["target"], ex["input"]
            for resp in (f"Final Answer: {gold}", gold, f"({gold})", f"Reasoning.\nFinal Answer: **{gold}**",
                         "Final Answer: ", "Final Answer:", "", "   ", f"I think {gold} is right",
                         f"Final Answer: {gold} because", "no answer", f"\\({gold}\\)"):
                score, ok = core.score_response(resp, gold, q), core.parse_ok(resp)
                assert not (score == 1 and not ok), (task, idx, resp)
                checked += 1
    assert core.score_response("(B)", "(B)", "Options:\n(A) x\n(B) y") == 1 and core.parse_ok("(B)")
    assert not core.parse_ok("Final Answer:   ") and not core.parse_ok("")
    bad = [r for r in ALL_ROWS if r["score"] == 1 and not r["parse_ok"]]
    assert not bad, bad[:1]
    print(f"  ok: parse_ok is never False on a row scored 1 — {checked} constructed responses and "
          f"{len(ALL_ROWS)} rows the other tests wrote; a bare `(B)` scores 1 with parse_ok")


def test_refusals():
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    try:
        for kw, word in (({"workers": 5}, "--workers"), ({"save_every": 0}, "--save-every")):
            try:
                run(md, good_once, **kw)
            except SystemExit as e:
                assert word in str(e), e
            else:
                raise AssertionError(f"{kw} was not refused")
        assert not os.path.exists(os.path.join(md, "results_sample10")), "a refused run wrote files"
        box = {}

        def off_main():
            try:
                core.call_with_retries(good_once, [{"role": "user", "content": "x"}], "m", md)
            except RuntimeError as e:
                box["e"] = e
        t = threading.Thread(target=off_main)
        t.start()
        t.join()
        assert "main thread" in str(box.get("e")), box
        p = subprocess.run([sys.executable, "-c", (
            "import sys; sys.path.insert(0, %r); import bbh_eval_core as c; "
            "sys.argv = ['x', '--sample', 'sample10', '--save-every', '0']; "
            "c.sample_main(description='t', default_model='m', model_dir=%r, client=None, "
            "once=None, wanted={})") % (BBH, md)], capture_output=True, text=True)
        assert p.returncode == 2 and "must be 1 or more" in p.stderr, p.stderr
        print("  ok: --workers 5 and --save-every 0 refused before any call or file; "
              "call_with_retries off the main thread raises")
    finally:
        shutil.rmtree(md, ignore_errors=True)


def test_smoke_and_shards():
    md = tempfile.mkdtemp(prefix="bbh_sample_")
    try:
        assert run(md, good_once, limit=3) == 0
        assert not os.path.exists(os.path.join(md, "results_sample10"))
        rows = rows_of(os.path.join(md, "smoke_sample10"))
        assert sum(map(len, rows.values())) == 60
        assert all([r["idx"] for r in rs] == SAMPLE[t][:3] for t, rs in rows.items())
        assert overall_last(os.path.join(md, "smoke_sample10"))["complete"] == "0"
        for s in (0, 1):
            assert run(md, good_once, shard=s, total_shards=2) == 0
        rows = rows_of(os.path.join(md, "results_sample10"))
        for t, rs in rows.items():
            assert sorted(r["idx"] for r in rs) == SAMPLE[t], t
        assert overall_last(os.path.join(md, "results_sample10"))["complete"] == "1"
        print("  ok: --limit 3 writes 60 rows to smoke_sample10/ only; two shards merge into one "
              "complete roll-up")
    finally:
        shutil.rmtree(md, ignore_errors=True)


# ======================================================================= Part C: the watchdog
#
# A blocked socket, not a busy loop: a local server that accepts the request and then either says
# nothing or trickles a byte every 0.5 s — the second defeats the SDK's read timeout (httpx's timeout
# is per read) — over plain HTTP and over TLS. The real openai SDK, `max_retries=0`, client timeout
# 400 s; the core's SIGALRM watchdog at 3 s must end each attempt. Adapted from the DocVQA package
# reviewer's watchdog_probe.py (2026-11-15). Needs `openai`, `httpx` and the `openssl` CLI for TLS.

def _have_sdk():
    try:
        import httpx  # noqa: F401
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def _read_request(conn):
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = conn.recv(65536)
        if not d:
            return
        buf += d
    head, body = buf.split(b"\r\n\r\n", 1)
    n = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            n = int(line.split(b":")[1])
    while len(body) < n:
        d = conn.recv(65536)
        if not d:
            return
        body += d


def _serve(mode, tls_ctx):
    import socket
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    def loop():
        while True:
            c, _ = srv.accept()
            if tls_ctx:
                c = tls_ctx.wrap_socket(c, server_side=True)

            def handle(c=c):
                try:
                    _read_request(c)
                    if mode == "silent":
                        time.sleep(3600)
                    else:
                        c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                                  b"Transfer-Encoding: chunked\r\n\r\n")
                        while True:
                            c.sendall(b"1\r\n\n\r\n")
                            time.sleep(0.5)
                except Exception:
                    pass
            threading.Thread(target=handle, daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()
    return srv.getsockname()[1]


def test_watchdog_ends_a_blocked_socket():
    if not _have_sdk():
        print("  SKIPPED — Part C needs openai and httpx; run this file with a Python that has them")
        return False
    import httpx
    import ssl
    from openai import OpenAI

    def guard():               # a watchdog that never fires must fail this test, not hang it
        time.sleep(120)
        print("FAIL: Part C gave up after 120 s — the watchdog did not end a blocked call", flush=True)
        os._exit(4)
    threading.Thread(target=guard, daemon=True).start()
    work = tempfile.mkdtemp(prefix="bbh_wd_")
    tls_ctx = None
    gen = subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                          "-subj", "/CN=127.0.0.1", "-keyout", os.path.join(work, "key.pem"),
                          "-out", os.path.join(work, "cert.pem")], capture_output=True)
    if gen.returncode == 0:
        tls_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_ctx.load_cert_chain(os.path.join(work, "cert.pem"), os.path.join(work, "key.pem"))
    cases = [("silent", None), ("trickle", None)] + ([("trickle", tls_ctx)] if tls_ctx else [])
    held = core.HARD_CALL_TIMEOUT
    try:
        core.HARD_CALL_TIMEOUT = 3
        for mode, ctx in cases:
            reset()
            port = _serve(mode, ctx)
            scheme = "https" if ctx else "http"
            client = OpenAI(base_url=f"{scheme}://127.0.0.1:{port}/v1", api_key="k", timeout=400,
                            max_retries=0,
                            http_client=httpx.Client(verify=False, timeout=400) if ctx else None)

            def once(m):
                r = client.chat.completions.create(model="m", messages=m, max_tokens=5)
                return r.choices[0].message.content, core.openai_meta(r)
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                text, meta = core.call_with_retries(once, [{"role": "user", "content": "x" * 1000}],
                                                    "probe", work, tries=2)
            took = time.time() - t0
            assert text == "" and meta["n_attempts"] == 2 and core.STATS["timeouts"] == 2, (meta, core.STATS)
            assert meta["attempts_without_usage"] == 2 and took < 12, took
            print(f"  ok: watchdog ended a {mode} {'TLS' if ctx else 'plain'} socket: 2 attempts, "
                  f"2 timeouts, {took:.1f} s against a 400 s client timeout")
        if not tls_ctx:
            print("  note: TLS case skipped — the openssl CLI could not make a certificate")
        return True
    finally:
        core.HARD_CALL_TIMEOUT = held
        shutil.rmtree(work, ignore_errors=True)


# ======================================================================= Part D: the real runners
#
# Each sample runner, unmodified, run as __main__ in a sandbox copy of the tree, through the real
# openai SDK pointed at a local server that plays a scripted sequence of replies — cut at the cap,
# silent (the watchdog ends it), reasoning that should not be there — with a 3 s watchdog. Adapted
# from the BBH reviewer's driver.py and wrapper.py (2026-11-15). The server counts the requests,
# which checks `n_attempts + cut_retry + forced_answer` against what was actually sent.

WRAPPER = r"""
import os, runpy, sys
runner = sys.argv[1]; sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(runner))))
import openai
_orig = openai.OpenAI.__init__
def init(self, *a, **kw):
    kw["base_url"] = os.environ["PROBE_URL"]; kw["api_key"] = "k"
    _orig(self, *a, **kw)
openai.OpenAI.__init__ = init
import bbh_eval_core as core
_rs = core.run_sample
def rs(*a, **kw):
    kw["call_timeout"] = int(os.environ.get("BBH_WRAPPER_TIMEOUT", "3"))
    return _rs(*a, **kw)
core.run_sample = rs
core.EMPTY_RETRY_WAIT = 0.2
runpy.run_path(runner, run_name="__main__")
"""

SAMPLE_RUNNERS = {
    "gpt": ("BBH_GPT_5.6_Luna/gpt56luna_bbh_eval.py", "gpt-5.6-luna"),
    "gemini": ("BBH_Gemini_Flash3.5lite_OpenRouter/gemini35lite_bbh_eval.py", "google/gemini-3.5-flash-lite"),
    "qwen": ("BBH_Qwen_DeepInfra/qwen_di_bbh_sample.py", "Qwen/Qwen3.5-9B"),
    "gemma": ("BBH_Gemma_DeepInfra/gemma_di_bbh_sample.py", "google/gemma-4-31B-it"),
    "deepseek": ("BBH_Deepseek/deepseek_bbh_sample.py", "deepseek-flash"),
}


def _body(content, finish, pt, ct, rt=None, reasoning=None):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
             "cost": 0.0005, "estimated_cost": 0.0003}
    if rt is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": rt}
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "served-m",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}


def _ok():
    return _body("Some reasoning.\nFinal Answer: Yes", "stop", 110, 50, 0)


SCENARIOS = {          # the non-probe replies, in order; the last one repeats
    "ok": [_ok],
    # DeepSeek's shape: a cut reply carries hidden reasoning
    "cut_silent_ok": [lambda: _body("partial reasoning", "length", 100, 8000, 7000, reasoning="thinking..."),
                      "silent", _ok],
    # Qwen's shape with reasoning off: it deliberates in the visible text, no reasoning field at all
    "cut_visible_silent_ok": [lambda: _body("partial visible deliberation", "length", 100, 8000),
                              "silent", _ok],
    "cut_cut_silent_forced": [lambda: _body("partial reasoning", "length", 100, 8000, 7000,
                                            reasoning="thinking..."),
                              lambda: _body("partial two", "length", 120, 8100, 7100, reasoning="thinking two"),
                              "silent", lambda: _body("Final Answer: Yes", "stop", 300, 9, 0)],
    "cut_cut_noreasoning": [lambda: _body("partial reasoning", "length", 100, 8000, 7000,
                                          reasoning="thinking..."),
                            lambda: _body("Final Answer: Yes", "length", 120, 8100, 0)],
    "think": [lambda: _body("Final Answer: Yes", "stop", 110, 50, 0, reasoning="I am thinking")],
    "think_inline": [lambda: _body("<think>pondering</think>Final Answer: Yes", "stop", 110, 50, 0)],
    # the reviewer's example: Qwen's first stage cut at the cap, then the retry hangs and the job is
    # killed — the cut stage's 8,192 tokens were billed and must reach the file
    "cut_then_hang": [lambda: _body("partial visible deliberation", "length", 100, 8192), "silent"],
}


def test_runners_end_to_end():
    if not _have_sdk():
        print("  SKIPPED — Part D needs openai and httpx; run this file with a Python that has them")
        return False
    # BaseHTTPRequestHandler at HTTP/1.1 reads EVERY request on a kept-alive connection, so a retry
    # the SDK sends on a reused connection is answered (a mock that stopped after the first request
    # left DocVQA's retry hanging unread). A "silent" reply blocks only its own connection, which the
    # SDK drops once the watchdog aborts the read.
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    state = {"scen": "ok", "n": 0}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            probe = req["messages"][-1]["content"] == "Reply with: ok"
            with lock:
                if not probe:
                    state["n"] += 1
                k, acts = state["n"], SCENARIOS[state["scen"]]
            act = (lambda: _body("ok", "stop", 5, 1, 0)) if probe else acts[min(k, len(acts)) - 1]
            if act == "silent":
                time.sleep(3600)
                return
            data = json.dumps(act()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    work = tempfile.mkdtemp(prefix="bbh_e2e_")
    root = os.path.join(work, "bbh")
    os.makedirs(root)
    for name in ("data", "dotenv", "sample10.json"):
        os.symlink(os.path.join(BBH, name), os.path.join(root, name))
    shutil.copy2(os.path.join(BBH, "bbh_eval_core.py"), root)
    for rel, _ in SAMPLE_RUNNERS.values():
        os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
        shutil.copy2(os.path.join(BBH, rel), os.path.join(root, rel))
    with open(os.path.join(work, "wrapper.py"), "w") as fh:
        fh.write(WRAPPER)

    def drive(who, scen, limit=1, fresh=True):
        rel, model = SAMPLE_RUNNERS[who]
        folder = os.path.join(root, os.path.dirname(rel))
        out = "smoke_sample10" if limit else "results_sample10"
        if fresh:
            shutil.rmtree(os.path.join(folder, out), ignore_errors=True)
            for f in os.listdir(folder):
                if f.endswith("_HALT.txt"):
                    os.remove(os.path.join(folder, f))
        with lock:
            state.update(scen=scen, n=0)
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        p = subprocess.run([sys.executable, "-B", os.path.join(work, "wrapper.py"), os.path.join(root, rel),
                            "--model", model, "--sample", "sample10", "--task", "navigate",
                            *(["--limit", str(limit)] if limit else []),
                            "--prompt", "v1", "--sleep", "0"],
                           env=dict(env, PROBE_URL=url, PYTHONUNBUFFERED="1"),
                           capture_output=True, text=True, timeout=300)
        path = os.path.join(folder, out, "navigate", core.model_slug(model) + ".jsonl")
        rows = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
        with lock:
            sent = state["n"]
        return p, (rows[0] if rows else {}), sent, folder

    def calls_of(r):
        return r["n_attempts"] + bool(r.get("cut_retry")) + bool(r.get("forced_answer"))
    try:
        for who in SAMPLE_RUNNERS:
            p, r, sent, _ = drive(who, "ok")
            assert p.returncode == 0, (who, p.stdout[-600:], p.stderr[-600:])
            assert all(k in r for k in core.ROW_KEYS + core.PRIOR_KEYS), (who, sorted(r))
            assert (r["billed_cost"], r["billed_estimated_cost"]) == (0.0005, 0.0003), (who, r)
            assert (r["last_call_cost"], r["last_call_estimated_cost"]) == (0.0005, 0.0003)
            assert r["attempts_without_usage"] == 0 and r["parse_ok"] and sent == calls_of(r) == 1
            print(f"  ok: {who:<8} ok              exit 0, every field, billed_cost and "
                  f"billed_estimated_cost from usage, 1 request")
        for who, scen in (("qwen", "cut_visible_silent_ok"), ("deepseek", "cut_silent_ok")):
            p, r, sent, _ = drive(who, scen)
            assert p.returncode == 0, (who, p.stdout[-600:], p.stderr[-600:])
            assert (r["n_attempts"], r["cut_retry"], r["attempts_without_usage"]) == (2, True, 1), r
            assert r["billed_completion_tokens"] == 8050 and sent == calls_of(r) == 3, (sent, r)
            print(f"  ok: {who:<8} {scen:<22} first call cut, retry hung (watchdog), retried: "
                  f"3 requests = n_attempts 2 + cut_retry, 1 without usage, billed 8,050")
        p, r, sent, folder = drive("qwen", "cut_silent_ok")
        # judged per row: the row's own calls finish (cut stage, hung retry, answering retry), then halt
        assert p.returncode == 3 and sent == 3 and r["reasoning_chars"] == len("thinking..."), \
            (p.returncode, sent, r.get("reasoning_chars"))
        assert os.path.exists(os.path.join(folder, "smoke_sample10_THINKING_HALT.txt"))
        print("  ok: qwen     cut_silent_ok          a CUT stage came back with reasoning: the row "
              "halts the run (its 3 calls, no further row), exit 3")
        p, r, sent, _ = drive("deepseek", "cut_cut_silent_forced")
        assert p.returncode == 0, (p.stdout[-600:], p.stderr[-600:])
        assert (r["n_attempts"], r["cut_retry"], r["forced_answer"], r["attempts_without_usage"]) == \
            (2, True, True, 1), r
        assert r["billed_completion_tokens"] == 8000 + 8100 + 9 and sent == calls_of(r) == 4, (sent, r)
        assert r["reasoning_chars"] == len("thinking...") + len("thinking two"), r
        print("  ok: deepseek cut_cut_silent_forced  4 requests = n_attempts 2 + cut_retry + "
              "forced_answer; reasoning_chars 23 = both cut stages")
        p, r, sent, folder = drive("deepseek", "cut_cut_noreasoning")
        assert p.returncode == 1, (p.returncode, p.stdout[-600:])
        assert core.is_unfinished(r) and sent == 2, r
        print("  ok: deepseek cut_cut_noreasoning    retry cut with content and no reasoning: "
              "unfinished, exit 1")
        for who in ("gemma", "qwen"):
            p, r, sent, folder = drive(who, "think_inline")
            assert p.returncode == 3, (who, p.returncode, p.stdout[-400:])
            assert (r["think_tag_calls"], r["reasoning_chars"]) == (1, 0), r
            print(f"  ok: {who:<8} think_inline           <think> written inside the content: "
                  f"think_tag_calls 1, halted, exit 3")
        # killed by SIGTERM while the retry stage hangs, after the cut first stage was billed
        rel, model = SAMPLE_RUNNERS["qwen"]
        folder = os.path.join(root, os.path.dirname(rel))
        shutil.rmtree(os.path.join(folder, "smoke_sample10"), ignore_errors=True)
        for f in os.listdir(folder):              # the think case before left its marker here
            if f.endswith("_HALT.txt"):
                os.remove(os.path.join(folder, f))
        with lock:
            state.update(scen="cut_then_hang", n=0)
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        proc = subprocess.Popen([sys.executable, "-B", os.path.join(work, "wrapper.py"),
                                 os.path.join(root, rel), "--model", model, "--sample", "sample10",
                                 "--task", "navigate", "--limit", "1", "--prompt", "v1", "--sleep", "0"],
                                env=dict(env, PROBE_URL=url, PYTHONUNBUFFERED="1",
                                         BBH_WRAPPER_TIMEOUT="60"),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.time() + 60
        while time.time() < deadline:
            with lock:
                if state["n"] >= 2:
                    break
            time.sleep(0.05)
        time.sleep(0.3)
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=60)
        path = os.path.join(folder, "smoke_sample10", "navigate", core.model_slug(model) + ".jsonl")
        assert os.path.exists(path), ("no row file after the kill", proc.returncode, out[-900:], err[-900:])
        r = json.loads(open(path).readline())
        assert proc.returncode == 143, (proc.returncode, out[-400:], err[-400:])
        assert (r["failed"], r["billed_completion_tokens"], r["billed_prompt_tokens"],
                r["attempts_without_usage"], r["n_attempts"]) == (1, 8192, 100, 1, 1), r
        assert r["error"].startswith("interrupted: SystemExit") and r["halted"] == "SystemExit: 143", r
        print("  ok: qwen     SIGTERM in the retry   the real runner, killed while its retry hung: the "
              "row is in the file, failed, billing the cut stage's 8,192 tokens, exit 143")
        for who in ("gemma", "qwen"):
            p, r, sent, folder = drive(who, "think")
            assert p.returncode == 3, (who, p.returncode, p.stdout[-600:], p.stderr[-400:])
            assert os.path.exists(os.path.join(folder, "smoke_sample10_THINKING_HALT.txt"))
            assert r["reasoning_chars"] == len("I am thinking"), r
            print(f"  ok: {who:<8} think                  reasoning_content came back: halted, "
                  f"smoke_sample10_THINKING_HALT.txt, exit 3")
            # a REAL run (no --limit) halts at its first row; its resubmits call nothing
            p, r, sent, folder = drive(who, "think", limit=0)
            marker = os.path.join(folder, "sample10_THINKING_HALT.txt")
            assert p.returncode == 3 and sent == 1 and os.path.exists(marker), (p.returncode, sent)
            p, _, sent, _ = drive(who, "ok", limit=0, fresh=False)
            assert p.returncode == 3 and sent == 0, ("resubmit with the marker", p.returncode, sent)
            os.remove(marker)
            p, _, sent, _ = drive(who, "ok", limit=0, fresh=False)
            assert p.returncode == 3 and sent == 0, ("resubmit without the marker", p.returncode, sent)
            assert os.path.exists(marker) and "found on resume" in open(marker).read()
            print(f"  ok: {who:<8} resubmit               after a real run halted on thinking: 0 rows "
                  f"called and exit 3, with the marker and with it deleted (then rewritten)")
        return True
    finally:
        srv.shutdown()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    started = time.time()
    parts = os.environ.get("BBH_TEST_PARTS", "ABCD").upper()
    ran, skipped = [], []
    if "B" in parts:
        print("Part B — the sample path")
        test_sample_serial_and_results_untouched()
        test_retries_keep_meta_true_and_bill_every_call()
        test_billed_usage_and_cost_the_reviewers_case()
        test_resume_keeps_a_failed_rows_history()
        test_killed_mid_resubmit_keeps_held_rows()
        test_inflight_row_kept_when_killed()
        test_failure_halt_row_is_written()
        test_a_call_is_counted_once()
        test_halt_in_the_inner_handler_counts_the_call_once()
        test_answered_row_recovered_when_killed_while_written()
        test_stop_before_a_rows_call_writes_nothing_for_it()
        test_stop_around_the_append_writes_the_row_once()
        test_failed_row_written_then_retried()
        test_billing_halt_and_marker_prefixes()
        test_cut_retry_billing_calls_and_reasoning()
        test_unfinished_cut_row_exits_1_and_is_called_again()
        test_thinking_halts_at_the_first_row()
        test_refusals()
        test_smoke_and_shards()
        test_parse_ok_agrees_with_the_score()           # last: it reads every row the others wrote
        ran.append("B")
    for letter, title, fn in (("C", "the watchdog against a blocked socket", test_watchdog_ends_a_blocked_socket),
                              ("D", "the real runners through the SDK", test_runners_end_to_end),
                              ("A", f"without --sample, against {BASELINE} (local only)",
                               test_legacy_path_identical)):
        if letter in parts:
            print(f"Part {letter} — {title}")
            (ran if fn() else skipped).append(letter)
    took = time.time() - started
    if skipped:
        print(f"PASSED: Part(s) {', '.join(ran)} in {took:.0f}s — Part(s) {', '.join(skipped)} "
              f"SKIPPED (see above): this is NOT the full proof")
    else:
        print(f"ALL PASSED ({', '.join(ran)}) in {took:.0f}s")
