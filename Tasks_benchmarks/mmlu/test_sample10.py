"""Offline tests for the 10% sample rerun in `mmlu_eval_core.py` — no provider is contacted.

    /path/to/python Tasks_benchmarks/mmlu/test_sample10.py   # needs python-dotenv, openai, httpx

The subprocess runs use a fake `openai` module; the watchdog test uses the real SDK against local
mock servers only (one silent, one that trickles bytes to defeat the SDK's read timeout).

Part A proves the guarantee the plan puts first (`sample10-plan.md` § Rules): **without `--sample`,
the changed core and runners behave exactly as before** — same filenames, same prompts, same rows.
It compares against BASELINE, the commit before the change, two ways: every top-level definition of
the old core is byte-identical in the new one and none is rebound; and the old and new GPT and Gemini
runners, the merge script included, are run side by side against a fake `openai` and must write
byte-identical files and send identical requests.

Part B attacks the sample path's own guarantees: a sample run cannot read or write `results/`; `idx`
indexes the whole subject; `n_attempts` and `latency_s` are true under retries; `billed_*` count every
attempt's usage, once; a failed row is written as failed, never dropped; resume refuses a changed
config; CutRetry puts the ceiling first; the merge checks ids against the sample file; every runner
sends PROMPT_V2; the SIGALRM watchdog ends a call blocked on a socket, re-arms across CutRetry stages
and refuses to run off the main thread; status codes match as words; a smoke never clears a real
run's halt marker; Qwen and Gemma exit 3 if thinking comes back on; a resubmit keeps a failed row's
history; the merge refuses damaged shards; each runner sends exactly its settled config. B12 and
B14-B17 are the WP8 reviewer's probes (scratchpad/rv/), copied in.
"""

import ast
import concurrent.futures
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REL = os.path.relpath(HERE, REPO)
BASELINE = "1cc5bbd5"     # the commit before the sample change
LEGACY_RUNNERS = {"MMLU_GPT_5.6_Luna": ("gpt56luna_mmlu_eval.py", "gpt-5.6-luna"),
                  "MMLU_Gemini_Flash3.5lite_OpenRouter": ("gemini35lite_mmlu_eval.py",
                                                          "google/gemini-3.5-flash-lite")}
SAMPLE_RUNNERS = {"MMLU_GPT_5.6_Luna": ("gpt56luna_mmlu_eval.py", "gpt-5.6-luna"),
                  "MMLU_Gemini_Flash3.5lite_OpenRouter": ("gemini35lite_mmlu_eval.py",
                                                          "google/gemini-3.5-flash-lite"),
                  "MMLU_Qwen_DeepInfra": ("qwen_di_mmlu_sample.py", "Qwen/Qwen3.5-9B"),
                  "MMLU_Gemma_DeepInfra": ("gemma_di_mmlu_sample.py", "google/gemma-4-31B-it"),
                  "MMLU_Deepseek": ("deepseek_mmlu_sample.py", "deepseek-flash")}
PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"PASS  {name}", flush=True)


def git_show(path):
    return subprocess.run(["git", "-C", REPO, "show", f"{BASELINE}:{path}"], check=True,
                          capture_output=True, text=True).stdout


# ---------------------------------------------------------------- a fake openai, deterministic

FAKE_OPENAI = r'''
import hashlib, json, os, threading
_LOCK = threading.Lock()

class _NS:
    def __init__(self, **kw): self.__dict__.update(kw)

def _answer(prompt):
    h = hashlib.md5(prompt.encode()).digest()
    letter = "ABCD"[h[0] % 4]
    shape = h[1] % 4
    if shape == 0: return "Some reasoning.\nFinal Answer: " + letter
    if shape == 1: return "Thinking it through.\n\nFinal Answer: (" + letter + ")"
    if shape == 2: return "I believe it is " + letter + " but I am not sure"
    return "Reasoning.\nFinal Answer: **" + letter + "**"

class _Completions:
    def __init__(self, owner): self.owner = owner
    def create(self, model, messages, **kw):
        log = os.environ.get("FAKE_OPENAI_LOG")
        if log:
            with _LOCK, open(log, "a") as fh:
                fh.write(json.dumps({"model": model, "messages": messages, "kw": kw,
                                     "client": self.owner.kw}, sort_keys=True) + "\n")
        prompt = messages[-1]["content"]
        text = "ok" if prompt == "Reply with: ok" else _answer(prompt)
        if os.environ.get("FAKE_THINK_TAG") and text != "ok":
            text = "<think>let me see</think>\n" + text
        usage = _NS(prompt_tokens=len(prompt) // 4, completion_tokens=len(text) // 4 + 7,
                    completion_tokens_details=_NS(
                        reasoning_tokens=int(os.environ.get("FAKE_REASONING_TOKENS", "0"))))
        msg = _NS(content=text, reasoning_content=os.environ.get("FAKE_REASONING") or None)
        return _NS(choices=[_NS(message=msg, finish_reason="stop")], usage=usage, model=model,
                   provider="FakeBackend")

class OpenAI:
    def __init__(self, *a, **kw):
        self.kw = {k: v for k, v in kw.items() if k != "api_key"}
        self.chat = _NS(completions=_Completions(self))
'''


def build_tree(root, version):
    """A copy of the benchmark: core, merge, the two legacy runners, the sample runners, data."""
    mm = os.path.join(root, "mmlu")
    os.makedirs(mm)
    os.symlink(os.path.join(HERE, "data"), os.path.join(mm, "data"))
    files = ["mmlu_eval_core.py", "merge_mmlu_shards.py"] + [
        f"{d}/{f}" for d, (f, _) in LEGACY_RUNNERS.items()]
    if version == "new":
        files += ["sample10.json"] + [f"{d}/{f}" for d, (f, _) in SAMPLE_RUNNERS.items()]
    for f in files:
        os.makedirs(os.path.dirname(os.path.join(mm, f)), exist_ok=True)
        with open(os.path.join(mm, f), "w") as fh:
            fh.write(git_show(f"{REL}/{f}") if version == "old" else open(os.path.join(HERE, f)).read())
    return mm


def run(mm, folder, script, args, env):
    p = subprocess.run([sys.executable, script] + args, cwd=os.path.join(mm, folder), env=env,
                       capture_output=True, text=True, timeout=600)
    return p


def tree_md5(top, skip=()):
    out = {}
    for path in sorted(glob.glob(os.path.join(top, "**", "*"), recursive=True)):
        rel = os.path.relpath(path, top)
        if os.path.isfile(path) and not any(rel.startswith(s) for s in skip):
            out[rel] = hashlib.md5(open(path, "rb").read()).hexdigest()
    return out


def fake_env(tmp, log):
    fake_dir = os.path.join(tmp, "fake")
    os.makedirs(fake_dir, exist_ok=True)
    with open(os.path.join(fake_dir, "openai.py"), "w") as fh:
        fh.write(FAKE_OPENAI)
    env = dict(os.environ, PYTHONPATH=fake_dir, FAKE_OPENAI_LOG=log, PYTHONHASHSEED="0")
    for k in list(env):
        if k.endswith("_API_KEY"):
            env.pop(k)
    return env


# ---------------------------------------------------------------- Part A: no --sample, no change


def test_static_core():
    old_src = git_show(f"{REL}/mmlu_eval_core.py")
    new_src = open(os.path.join(HERE, "mmlu_eval_core.py")).read()

    def defs(src):
        out, imports = {}, set()
        for node in ast.parse(src).body:
            seg = ast.get_source_segment(src, node)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                out.setdefault(node.name, []).append(seg)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            out.setdefault(n.id, []).append(seg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.update(a.asname or a.name for a in node.names)
        return out, imports
    old, old_imp = defs(old_src)
    new, new_imp = defs(new_src)
    for name, segs in old.items():
        assert len(segs) == 1, name
        assert new.get(name) == segs, f"{name} changed or rebound: {len(new.get(name, []))} definition(s)"
    assert old_imp <= new_imp, old_imp - new_imp
    ok(f"A1 core: all {len(old)} top-level names of {BASELINE} byte-identical, none rebound "
       f"({len(new) - len(old)} new names added)")


def test_legacy_runs_identical():
    tmp = tempfile.mkdtemp(prefix="mmlu_equiv_")
    try:
        trees = {v: build_tree(os.path.join(tmp, v), v) for v in ("old", "new")}
        scenarios = [
            ("serial, all subjects", ["--limit", "3", "--subject", "all"]),
            ("v1 prompt", ["--limit", "2", "--subject", "Business_ethics,Formal_logic", "--prompt", "v1"]),
            ("4 workers", ["--limit", "5", "--subject", "Marketing,Moral_scenarios", "--workers", "4"]),
        ]
        for folder, (script, model) in LEGACY_RUNNERS.items():
            for label, args in scenarios:
                outs, logs = {}, {}
                for v, mm in trees.items():
                    shutil.rmtree(os.path.join(mm, folder, "results"), ignore_errors=True)
                    log = os.path.join(tmp, f"{v}.log")
                    if os.path.exists(log):
                        os.remove(log)
                    p = run(mm, folder, script, args, fake_env(tmp, log))
                    assert p.returncode == 0, (v, folder, label, p.stderr[-2000:])
                    outs[v] = {k: h for k, h in tree_md5(os.path.join(mm, folder, "results")).items()}
                    logs[v] = sorted(open(log).read().splitlines())
                assert outs["old"] and outs["old"] == outs["new"], (folder, label)
                assert logs["old"] == logs["new"], (folder, label, "requests differ")
                ok(f"A2 {folder} [{label}]: {len(outs['old'])} files byte-identical, "
                   f"{len(logs['old'])} requests identical")
            # sharded run + merge, the run_mmlu.sh path
            outs = {}
            for v, mm in trees.items():
                shutil.rmtree(os.path.join(mm, folder, "results"), ignore_errors=True)
                env = fake_env(tmp, os.path.join(tmp, f"{v}.log"))
                for shard in range(3):
                    p = run(mm, folder, script, ["--limit", "4", "--subject", "all", "--shard",
                                                 str(shard), "--total-shards", "3"], env)
                    assert p.returncode == 0, p.stderr[-2000:]
                p = subprocess.run([sys.executable, "../merge_mmlu_shards.py", "--model", model,
                                    "--model-dir", ".", "--total-shards", "3"],
                                   cwd=os.path.join(mm, folder), env=env, capture_output=True, text=True)
                # --limit leaves every subject short, so the merge must report it and exit 1 — in
                # both versions alike
                outs[v] = (tree_md5(os.path.join(mm, folder, "results")), p.returncode, p.stdout)
            assert outs["old"] == outs["new"], (folder, "sharded + merge")
            ok(f"A3 {folder} [3 shards + merge]: {len(outs['old'][0])} files byte-identical, "
               f"merge exit {outs['old'][1]} and stdout identical")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- Part B: the sample path


def load_core():
    sys.path.insert(0, HERE)
    import mmlu_eval_core as core
    return core


def sentinel_results(model_dir):
    """A stand-in `results/` the sample run must leave alone."""
    d = os.path.join(model_dir, "results", "Business_ethics")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "sentinel.jsonl"), "w") as fh:
        fh.write('{"idx": 0, "model_response": "do not touch"}\n')
    return tree_md5(os.path.join(model_dir, "results"))


def test_sample_runners_end_to_end():
    core = load_core()
    ids, _ = core.load_sample("sample10")
    tmp = tempfile.mkdtemp(prefix="mmlu_sample_")
    try:
        mm = build_tree(os.path.join(tmp, "new"), "new")
        for folder, (script, model) in SAMPLE_RUNNERS.items():
            model_dir = os.path.join(mm, folder)
            os.makedirs(model_dir, exist_ok=True)
            before = sentinel_results(model_dir)
            log = os.path.join(tmp, f"{folder}.log")
            env = fake_env(tmp, log)
            # a real run's markers; a smoke must not clear them, and shard 0 clears only its own
            real_markers = ["sample10_BILLING_HALT.txt", "sample10_shard1_FAILURE_HALT.txt"]
            for m in real_markers + ["sample10_shard0_FAILURE_HALT.txt"]:
                open(os.path.join(model_dir, m), "w").write("stale\n")
            p = run(mm, folder, script, ["--model", model, "--sample", "sample10", "--limit", "1",
                                         "--subject", "Management"], env)
            assert p.returncode == 0, p.stderr[-1500:]
            assert all(os.path.exists(os.path.join(model_dir, m)) for m in real_markers + [
                "sample10_shard0_FAILURE_HALT.txt"]), "a smoke cleared a real run's marker"
            p = run(mm, folder, script, ["--model", model, "--sample", "sample10", "--save-every", "0"], env)
            assert p.returncode != 0 and "--save-every must be at least 1" in p.stderr, p.stderr[-500:]
            os.remove(log)
            for shard in range(5):
                p = run(mm, folder, script, ["--model", model, "--sample", "sample10",
                                             "--shard", str(shard), "--total-shards", "5"], env)
                assert p.returncode == 0, (folder, p.stdout[-1500:], p.stderr[-1500:])
            p = subprocess.run([sys.executable, "../merge_mmlu_shards.py", "--model", model,
                                "--model-dir", ".", "--total-shards", "5", "--sample", "sample10"],
                               cwd=model_dir, env=env, capture_output=True, text=True)
            assert p.returncode == 0, p.stdout + p.stderr
            assert tree_md5(os.path.join(model_dir, "results")) == before, "results/ was touched"
            # each shard cleared its own marker at startup; the unsharded run's prefix is not theirs
            assert not any(os.path.exists(os.path.join(model_dir, f"sample10_shard{k}_FAILURE_HALT.txt"))
                           for k in (0, 1))
            assert os.path.exists(os.path.join(model_dir, "sample10_BILLING_HALT.txt"))
            slug = core.model_slug(model)
            total = 0
            for subject in core.SUBJECTS:
                data = core.load_subject(subject)
                rows = [json.loads(x) for x in open(os.path.join(
                    model_dir, "results_sample10", subject, f"{slug}.jsonl"))]
                assert [r["idx"] for r in rows] == ids[subject], subject
                for r in rows:
                    assert r["question"] == data[r["idx"]]["question"]
                    assert r["answer_index"] == int(data[r["idx"]]["answer"])
                    for f in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s",
                              "finish_reason", "n_attempts", "parse_ok", "served_model"):
                        assert f in r, f
                    assert isinstance(r["prompt_tokens"], int) and isinstance(r["latency_s"], float)
                    assert r["n_attempts"] == 1 and r["config"]["params"]["seed"] == 42
                    for f in core.SAMPLE_FIELDS:
                        assert f in r or f in ("backend", "last_call_cost",
                                               "last_call_estimated_cost", "cut_retry",
                                               "forced_answer"), f
                    if "cut_retry" not in r:
                        assert (r["billed_prompt_tokens"], r["billed_completion_tokens"],
                                r["billed_reasoning_tokens"]) == (r["prompt_tokens"],
                                                                  r["completion_tokens"],
                                                                  r["reasoning_tokens"])
                    # 0, not negative: every HTTP call of every runner went through core.create
                    assert r["attempts_without_usage"] == 0
                    assert r["billed_cost"] is None and r["billed_estimated_cost"] is None
                    assert (r["prior_n_attempts"], r["prior_failed_runs"],
                            r["resumed_after_failure"]) == (0, 0, False)
                    assert r["reasoning_chars"] == 0
                    assert r["config"]["prompt"] == "v2" and r["config"]["sample"] == "sample10"
                total += len(rows)
            assert total == 401
            reqs = [json.loads(x) for x in open(log)]
            calls = [q for q in reqs if q["messages"][-1]["content"] != "Reply with: ok"]
            assert len(calls) == 401, len(calls)
            prompts = {core.build_prompt(data_ex, "v2") for s in core.SUBJECTS
                       for i, data_ex in enumerate(core.load_subject(s)) if i in set(ids[s])}
            assert {q["messages"][-1]["content"] for q in calls} == prompts
            assert all(q["client"].get("max_retries") == 0 for q in reqs)
            roles = {tuple(m["role"] for m in q["messages"]) for q in calls}
            ok(f"B1 {folder}: 5 shards + merge = 401 rows, idx = sample ids, every field present "
               f"(billed_* = this call's usage), results/ untouched, PROMPT_V2 user turn, "
               f"max_retries=0, roles {sorted(roles)}; a smoke left all 3 real-run halt markers, each "
               f"shard cleared only its own; --save-every 0 refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_inprocess(core, model_dir, once, config=None, subjects=("Business_ethics",),
                  limit=0, shard=0, total=1):
    ids, sha = core.load_sample("sample10")
    config = config or {"model": "fake", "params": {"seed": 42}, "prompt": "v2",
                        "sample": "sample10", "sample_sha1": sha}
    return core.run_sample_subjects(model_dir, "fake/model-1.0", once, config, "sample10", ids,
                                    list(subjects), limit=limit, shard=shard, total_shards=total,
                                    save_every=3)


def test_serial_only():
    """Sample calls run in the main thread, where SIGALRM works: --workers > 1 is refused, and a call
    from any other thread raises instead of running unguarded."""
    core = load_core()
    import argparse
    args = argparse.Namespace(prompt="v2", workers=4, save_every=20, subject="all",
                              sample="sample10", limit=0, shard=0, total_shards=1, sleep=0.0,
                              model="m")
    try:
        core.run_sample(args, model_dir=tempfile.mkdtemp(), client=None, once=None, wanted={})
        raise AssertionError("--workers 4 was accepted")
    except SystemExit as e:
        assert "--workers must be 1" in str(e)
    caught = []

    def worker():
        try:
            core.call_with_retries(lambda m: ("Final Answer: A", {}), [], "m", tempfile.mkdtemp())
        except RuntimeError as e:
            caught.append(str(e))
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert caught and "main thread" in caught[0], caught
    ok("B2 --workers 4 refused for a sample run; call_with_retries from a non-main thread raises "
       "(never runs unguarded)")


_real_time = time


class _FastTime:
    """core.time with sleep shortened: back-off still happens, only quicker."""
    time = staticmethod(_real_time.time)
    strftime = staticmethod(_real_time.strftime)

    @staticmethod
    def sleep(seconds):
        _real_time.sleep(0.2 if seconds > 0.2 else seconds)


def test_retries_attempts_latency_and_failed_rows():
    core = load_core()
    real_time = core.time
    core.time = _FastTime
    tmp = tempfile.mkdtemp(prefix="mmlu_retry_")
    try:
        ids, _ = core.load_sample("sample10")
        plan = {i: k % 4 for k, i in enumerate(ids["Business_ethics"])}   # failures before success
        doomed = ids["Business_ethics"][3]
        plan[doomed] = 99
        seen = {}
        lock = threading.Lock()

        def once(messages):
            q = messages[-1]["content"]
            idx = next(i for i in ids["Business_ethics"]
                       if core.load_subject("Business_ethics")[i]["question"] in q)
            with lock:
                seen[idx] = seen.get(idx, 0) + 1
                n = seen[idx]
            time.sleep(0.01)
            if n <= plan[idx]:
                if n % 2:
                    raise RuntimeError("503 upstream connect error")
                return "", {"finish_reason": "stop", "completion_tokens": 0}
            return "Final Answer: B", {"finish_reason": "stop", "prompt_tokens": 10,
                                       "completion_tokens": 3}
        for workers in (1,):
            seen.clear()
            core.STATS.update({k: 0 for k in core.STATS})
            core._recent.clear()
            core._consecutive[0] = 0
            shutil.rmtree(tmp, ignore_errors=True)
            _, rows = run_inprocess(core, tmp, once)
            by = {r["idx"]: r for r in rows}
            assert sorted(by) == ids["Business_ethics"]          # nothing dropped
            for i, r in by.items():
                if i == doomed:
                    assert r["failed"] and r["model_response"] == "" and r["n_attempts"] == 5
                    assert r["latency_s"] is None and not r["parse_ok"] and r["score"] == 0
                else:
                    assert r["n_attempts"] == plan[i] + 1, (i, r["n_attempts"], plan[i])
                    assert r["latency_s"] < 0.15, (i, r["latency_s"])   # back-off excluded
                    assert not r["failed"]
            # resume: only the failed row is called again
            seen.clear()
            plan[doomed] = 0
            _, rows = run_inprocess(core, tmp, once)
            assert list(seen) == [doomed] and seen[doomed] == 1, seen
            assert not any(r["failed"] for r in rows) and len(rows) == len(ids["Business_ethics"])
            plan[doomed] = 99
            ok(f"B3 serial: n_attempts = failures + 1 on every row, latency_s excludes "
               f"back-off (<0.15 s against 0.2 s sleeps), the all-fail row written failed with "
               f"n_attempts=5, and resume re-called only that row")
    finally:
        core.time = real_time
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_refuses_changed_config_and_foreign_rows():
    core = load_core()
    tmp = tempfile.mkdtemp(prefix="mmlu_cfg_")
    try:
        once = lambda m: ("Final Answer: A", {"finish_reason": "stop"})
        _, sha = core.load_sample("sample10")
        run_inprocess(core, tmp, once)
        try:
            run_inprocess(core, tmp, once, config={"model": "fake", "params": {"seed": 7},
                                                   "prompt": "v2", "sample": "sample10",
                                                   "sample_sha1": sha})
            raise AssertionError("resume across a config change was not refused")
        except SystemExit as e:
            assert "refusing to resume" in str(e)
        try:
            run_inprocess(core, tmp, once, shard=0, total=1, subjects=("Business_ethics",))
            # same shard boundaries: fine. Now pretend the unsharded file is shard 0 of 2.
            src = os.path.join(tmp, "results_sample10", "Business_ethics", "fake-model-1_0.jsonl")
            shutil.copy(src, src.replace(".jsonl", "_shard0of2.jsonl"))
            run_inprocess(core, tmp, once, shard=0, total=2)
            raise AssertionError("foreign rows were not refused")
        except SystemExit as e:
            assert "not in this shard" in str(e)
        ok("B4 resume refuses a changed config and rows from other shard boundaries")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_smoke_dir_and_halt():
    core = load_core()
    tmp = tempfile.mkdtemp(prefix="mmlu_smoke_")
    try:
        before = sentinel_results(tmp)
        out, rows = run_inprocess(core, tmp, lambda m: ("Final Answer: A", {}), limit=3,
                                  subjects=("Formal_logic", "Moral_scenarios"))
        assert out.endswith("smoke_sample10") and len(rows) == 6
        assert not os.path.exists(os.path.join(tmp, "results_sample10"))
        assert tree_md5(os.path.join(tmp, "results")) == before
        core.MARKER_PREFIX = "sample10_"

        def broke(m):
            raise RuntimeError("Error code: 402 - You need positive balance to do inference.")
        try:
            run_inprocess(core, tmp, broke)
            raise AssertionError("billing did not halt")
        except SystemExit:
            pass
        assert os.path.exists(os.path.join(tmp, "sample10_BILLING_HALT.txt"))
        assert tree_md5(os.path.join(tmp, "results")) == before
        core.MARKER_PREFIX = ""
        ok("B5 --limit writes smoke_sample10/ only; a DeepInfra 402 halts with "
           "sample10_BILLING_HALT.txt; results/ untouched throughout")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cutretry():
    core = load_core()
    sent = []

    def call(messages, stage):
        sent.append((stage, [m["role"] for m in messages], messages[0]["content"]))
        if stage in ("first", "retry"):
            return "partial…", "partial…", {"finish_reason": "length", "completion_tokens": 100}
        return "Final Answer: C", "", {"finish_reason": "stop", "completion_tokens": 5,
                                       "reasoning_chars": 0}
    cut = core.CutRetry("CEILING", "RETRY", "FORCE", 64)
    msgs = core.build_messages(core.load_subject("Formal_logic")[0])
    text, meta = cut(msgs, call, "m")
    assert text == "Final Answer: C" and meta["forced_answer"] and meta["cut_retry"]
    assert meta["completion_tokens"] == 205
    assert sent[0] == ("first", ["system", "user"], "CEILING")
    assert sent[1] == ("retry", ["system", "user"], "RETRY")
    assert sent[2][0] == "forced" and sent[2][1] == ["system", "user", "assistant", "user"]
    assert msgs == core.build_messages(core.load_subject("Formal_logic")[0])   # not mutated
    # per-thread state: two rows in flight in two threads do not share a first call
    cut2 = core.CutRetry("C", "R", "F", 8)
    firsts = []
    gate = threading.Barrier(2)

    def call2(messages, stage):
        if stage == "first":
            gate.wait()
            firsts.append(messages[-1]["content"])
            return "cut", "cut", {"finish_reason": "length"}
        return "Final Answer: " + messages[-1]["content"][-1], "", {"finish_reason": "stop"}
    rows = [[{"role": "user", "content": "row-X"}], [{"role": "user", "content": "row-Y"}]]
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        got = list(pool.map(lambda m: cut2(m, call2)[1]["first_response"], rows))
    assert got == ["cut", "cut"] and sorted(firsts) == ["row-X", "row-Y"]
    ok("B6 CutRetry: ceiling is the FIRST message (system, user), retry then forced, usage summed, "
       "messages not mutated, per-thread state")


def test_runner_wordings():
    """Qwen's three instructions are PlanBench's word for word; Deepseek's too but for one phrase."""
    import importlib.util

    def consts(path):
        tree = ast.parse(open(path).read())
        out = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                try:
                    out[node.targets[0].id] = ast.literal_eval(node.value)
                except Exception:
                    pass
        return out
    plan = os.path.join(REPO, "Transition_processes_benchmarks", "LLMs-Planning_bench")
    pq = consts(os.path.join(plan, "PLAN_Qwen_DeepInfra", "qwen_di_plan_eval.py"))
    mq = consts(os.path.join(HERE, "MMLU_Qwen_DeepInfra", "qwen_di_mmlu_sample.py"))
    for k in ("PROMPT_CEILING", "RETRY_CEILING", "FORCE_ANSWER", "ANSWER_MAX", "WANTED", "FIXED"):
        assert pq[k] == mq[k], k
    pd = consts(os.path.join(plan, "PLAN_Deepseek", "deepseek_plan_eval.py"))
    md = consts(os.path.join(HERE, "MMLU_Deepseek", "deepseek_mmlu_sample.py"))
    for k in ("PROMPT_CEILING", "FORCE_ANSWER", "ANSWER_MAX", "WANTED", "NO_THINKING"):
        assert pd[k] == md[k], k
    assert pd["RETRY_CEILING"].replace("as soon as you have a plan. Your",
                                       "as soon as you have an answer. Your") == md["RETRY_CEILING"]
    pg = consts(os.path.join(plan, "PLAN_Gemma_DeepInfra", "gemma_di_plan_eval.py"))
    mg = consts(os.path.join(HERE, "MMLU_Gemma_DeepInfra", "gemma_di_mmlu_sample.py"))
    assert pg["WANTED"] == mg["WANTED"]
    ok("B7 Qwen's ceiling/retry/force/caps = PlanBench's byte for byte; Deepseek's likewise except "
       "'a plan' -> 'an answer'; Gemma's WANTED = PlanBench's")


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def fake_response(text, prompt, completion, reasoning=None, finish="stop", usage=True, cost=None,
                  estimated_cost=None, reasoning_text=None):
    details = _NS(reasoning_tokens=reasoning) if reasoning is not None else None
    u = None
    if usage:
        u = _NS(prompt_tokens=prompt, completion_tokens=completion, completion_tokens_details=details)
        if cost is not None:
            u.cost = cost
        if estimated_cost is not None:
            u.estimated_cost = estimated_cost
    message = _NS(content=text, reasoning_content=reasoning_text)
    return _NS(choices=[_NS(message=message, finish_reason=finish)], usage=u, model="m")


class ScriptedClient:
    """A client whose create() returns, or raises, the next scripted item."""
    def __init__(self, items):
        self.items = iter(items)
        self.chat = _NS(completions=_NS(create=self._create))

    def _create(self, **kwargs):
        item = next(self.items)
        if isinstance(item, BaseException):
            raise item
        return item


def plain_once(core, client):
    def once(messages):
        r = core.create(client, model="m", messages=messages)
        meta = core.openai_meta(r)
        return r.choices[0].message.content, meta
    return once


def test_billed_usage():
    """billed_* and billed_*cost sum every HTTP call of the row; attempts_without_usage counts the
    calls that returned no usage, CutRetry stages included; the answer's own usage stays in *_tokens."""
    core = load_core()
    real_time = core.time
    core.time = _FastTime
    md = tempfile.mkdtemp(prefix="mmlu_bill_")
    try:
        ex = core.load_subject("Business_ethics")[6]
        msgs = core.build_messages(ex)

        # the coordinator's case: an empty HTTP 200 that billed 16,384 reasoning tokens, then 50
        client = ScriptedClient([fake_response("", 120, 16384, reasoning=16384, cost=0.04),
                                 fake_response("Final Answer: B", 120, 50, reasoning=10, cost=0.0002)])
        text, meta = core.call_with_retries(plain_once(core, client), msgs, "m", md)
        row = core.sample_row(6, ex, text, meta, {}, "sample10")
        assert (row["completion_tokens"], row["n_attempts"]) == (50, 2)
        assert (row["billed_completion_tokens"], row["billed_prompt_tokens"],
                row["billed_reasoning_tokens"], row["attempts_without_usage"]) == (16434, 240, 16394, 0)
        assert abs(row["billed_cost"] - 0.0402) < 1e-12 and row["last_call_cost"] == 0.0002
        assert row["billed_estimated_cost"] is None and "last_call_estimated_cost" not in row

        # an exception and a timeout leave no usage; reasoning never reported -> None
        client = ScriptedClient([RuntimeError("503 Service Unavailable"), core.CallTimeout("hung"),
                                 fake_response("Final Answer: B", 100, 40, estimated_cost=0.001)])
        text, meta = core.call_with_retries(plain_once(core, client), msgs, "m", md)
        assert (meta["n_attempts"], meta["attempts_without_usage"], meta["billed_completion_tokens"],
                meta["billed_reasoning_tokens"], meta["billed_estimated_cost"],
                meta["billed_cost"]) == (3, 2, 40, None, 0.001, None)

        # usage booked, then the runner raises (HTTP 200 with usage but no choices): one call,
        # billed, counted once — not also as a call without usage
        client = ScriptedClient([_NS(choices=[], usage=_NS(prompt_tokens=100, completion_tokens=900,
                                                           completion_tokens_details=None), model="m"),
                                 fake_response("Final Answer: B", 100, 40)])
        text, meta = core.call_with_retries(plain_once(core, client), msgs, "m", md)
        assert (meta["n_attempts"], meta["attempts_without_usage"],
                meta["billed_completion_tokens"]) == (2, 0, 940), meta

        # a response with no usage object at all is a call without usage
        client = ScriptedClient([fake_response("", 0, 0, usage=False),
                                 fake_response("Final Answer: B", 90, 30)])
        text, meta = core.call_with_retries(plain_once(core, client), msgs, "m", md)
        assert (meta["attempts_without_usage"], meta["billed_completion_tokens"]) == (1, 30)

        # every attempt failed: the row is written failed and still carries what was billed
        client = ScriptedClient([fake_response("", 100, 8000, reasoning=8000)] * 5)
        text, meta = core.call_with_retries(plain_once(core, client), msgs, "m", md)
        row = core.sample_row(6, ex, text, meta, {}, "sample10")
        assert row["failed"] and row["billed_completion_tokens"] == 40000 and row["latency_s"] is None

        # CutRetry: a stage kept across attempts billed once; a stage that raised is a call without
        # usage; reasoning text summed over every stage; costs summed; last_call_* the last call's
        cut = core.CutRetry("CEIL", "RETRY", "FORCE", 64)
        client = ScriptedClient([
            fake_response("cut", 100, 8192, finish="length", reasoning_text="abc",
                          estimated_cost=0.003),                          # attempt 1: first, cut
            RuntimeError("502 Bad Gateway"),                              # attempt 1: retry raised
            fake_response("", 110, 30, reasoning_text="de", estimated_cost=0.0001),  # 2: retry, empty
            fake_response("Final Answer: B", 110, 40, estimated_cost=0.0002)])       # 3: retry answers

        def call(sent, stage):
            r = core.create(client, model="m", messages=sent)
            meta = core.openai_meta(r)
            text = r.choices[0].message.content
            return text, text, meta
        text, meta = core.call_with_retries(lambda m: cut(m, call, "m"), msgs, "m", md)
        assert text == "Final Answer: B" and meta["n_attempts"] == 3 and meta["cut_retry"]
        assert meta["completion_tokens"] == 8192 + 40                    # the answering attempt's
        assert meta["billed_completion_tokens"] == 8192 + 30 + 40        # every call, first once
        assert meta["billed_prompt_tokens"] == 100 + 110 + 110
        assert meta["attempts_without_usage"] == 1                       # the retry that raised
        assert meta["reasoning_chars"] == 5                              # 3 + 2, stages included
        assert abs(meta["billed_estimated_cost"] - 0.0033) < 1e-12
        assert meta["last_call_estimated_cost"] == 0.0002
        ok("B8 billed usage: empty 16,384-token attempt + 50-token answer -> completion_tokens 50, "
           "billed_completion_tokens 16434, billed_cost the sum (last_call_cost the last call's); "
           "exceptions, timeouts and usage-less responses counted per HTTP call in "
           "attempts_without_usage, CutRetry stages included; billed_reasoning and billed_*cost None "
           "when unreported; a failed row keeps its bill; a kept CutRetry stage billed once; "
           "reasoning_chars summed over every stage")
    finally:
        core.time = real_time
        core.STATS.update({k: 0 for k in core.STATS})
        core._recent.clear()
        core._consecutive[0] = 0
        shutil.rmtree(md, ignore_errors=True)


# The keep-alive mock servers of scratchpad/wd/watchdog_probe.py (the DocVQA reviewer's), copied so
# this test does not depend on a scratch directory: "silent" accepts and never answers; "trickle"
# sends a chunk every 0.5 s forever, which resets the SDK's read timeout each time.
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


def _serve(mode, cert_dir=None):
    import socket
    import ssl
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    ctx = None
    if cert_dir:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(os.path.join(cert_dir, "cert.pem"), os.path.join(cert_dir, "key.pem"))

    def loop():
        while True:
            c, _ = srv.accept()
            if ctx:
                c = ctx.wrap_socket(c, server_side=True)

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


def test_watchdog_blocked_socket():
    """The SIGALRM watchdog ends a real openai-SDK call blocked in socket I/O — including one the
    SDK's own read timeout never ends — and the row comes back failed with its attempts counted."""
    import httpx
    from openai import OpenAI
    core = load_core()
    real_time, real_limit = core.time, core.HARD_CALL_TIMEOUT
    core.time = _FastTime
    core.HARD_CALL_TIMEOUT = 3
    md = tempfile.mkdtemp(prefix="mmlu_wd_")
    msgs = [{"role": "user", "content": "x" * 1000}]
    try:
        # first, the trap: the SDK alone, read timeout 2 s, against the trickling server
        port = _serve("trickle")
        bare = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="k", timeout=2, max_retries=0)
        t = threading.Thread(target=lambda: bare.chat.completions.create(
            model="m", messages=msgs, max_tokens=5), daemon=True)
        t0 = time.time()
        t.start()
        t.join(8)
        assert t.is_alive(), "the SDK timed out on its own — the mock no longer defeats it"
        sdk_note = f"SDK alone (timeout=2) still blocked after {time.time() - t0:.0f} s"

        cert_dir = os.environ.get("WD_CERT_DIR")
        modes = [("silent", None), ("trickle", None)] + ([("trickle", cert_dir)] if cert_dir else [])
        results = []
        for mode, certs in modes:
            port = _serve(mode, certs)
            scheme = "https" if certs else "http"
            client = OpenAI(base_url=f"{scheme}://127.0.0.1:{port}/v1", api_key="k", timeout=400,
                            max_retries=0,
                            http_client=httpx.Client(verify=False, timeout=400) if certs else None)

            def once(m, client=client):
                r = core.create(client, model="m", messages=m, max_tokens=5)
                return r.choices[0].message.content, core.openai_meta(r)
            before = core.STATS["timeouts"]
            t0 = time.time()
            text, meta = core.call_with_retries(once, msgs, "probe", md, tries=2)
            elapsed = time.time() - t0
            assert text == "" and meta["n_attempts"] == 2 and meta["latency_s"] is None
            assert meta["attempts_without_usage"] == 2 and meta["billed_completion_tokens"] == 0
            assert core.STATS["timeouts"] - before == 2
            assert 5.5 < elapsed < 9, elapsed               # two 3 s watchdogs + one short back-off
            results.append(f"{mode}{'+tls' if certs else ''} {elapsed:.1f}s")
            core._consecutive[0] = 0
            core._recent.clear()
        ok(f"B9 watchdog vs a blocked socket: {sdk_note}; call_with_retries(tries=2, 3 s watchdog) "
           f"ended every call — {', '.join(results)} — row failed, 2 timeouts, 2 attempts without "
           f"usage")
    finally:
        core.time, core.HARD_CALL_TIMEOUT = real_time, real_limit
        shutil.rmtree(md, ignore_errors=True)


def test_status_codes_are_words():
    core = load_core()
    hex_500 = RuntimeError("Error code: 500 - {'error': 'internal', 'request_id': 'req_9c401de2b'}")
    assert not core.is_fatal_request(hex_500), "a 401 inside a hex request id read as auth"
    assert core.is_fatal_request(RuntimeError("Error code: 401 - {'error': 'invalid key'}"))
    assert not core.mentions("upstream id 0x7f5029a", ["502"])
    assert core.mentions("Error code: 502 - Bad Gateway", ["502"])
    assert core.mentions("You need positive balance to do inference", core.BILLING_SIGNATURES)
    ok("B10 status codes match as words: '401' inside a hex request id of a 500 is not an auth "
       "failure; a real 401 still is; word signatures still match as substrings")


def test_thinking_off_is_checked():
    """Qwen and Gemma run with thinking off: reasoning text, or (Gemma) a reasoning token count, on a
    row means it came back on — the shard says so and exits 3. Clean replies exit 0."""
    tmp = tempfile.mkdtemp(prefix="mmlu_think_")
    try:
        mm = build_tree(os.path.join(tmp, "new"), "new")
        base = fake_env(tmp, os.path.join(tmp, "t.log"))
        cases = [("MMLU_Qwen_DeepInfra", {"FAKE_REASONING": "Let me think first."}, 3),
                 ("MMLU_Gemma_DeepInfra", {"FAKE_REASONING": "Let me think first."}, 3),
                 ("MMLU_Gemma_DeepInfra", {"FAKE_REASONING_TOKENS": "7"}, 3),
                 ("MMLU_Qwen_DeepInfra", {"FAKE_THINK_TAG": "1"}, 3),
                 ("MMLU_Gemma_DeepInfra", {"FAKE_THINK_TAG": "1"}, 3),
                 ("MMLU_GPT_5.6_Luna", {"FAKE_THINK_TAG": "1"}, 0),
                 ("MMLU_Deepseek", {"FAKE_THINK_TAG": "1"}, 0),
                 ("MMLU_Qwen_DeepInfra", {}, 0), ("MMLU_Gemma_DeepInfra", {}, 0)]
        seen = []
        for folder, extra, want in cases:
            script = SAMPLE_RUNNERS[folder][0]
            log = os.path.join(tmp, "t.log")
            if os.path.exists(log):
                os.remove(log)
            marker = os.path.join(mm, folder, "smoke_sample10_THINKING_HALT.txt")
            p = run(mm, folder, script, ["--model", SAMPLE_RUNNERS[folder][1], "--sample",
                                         "sample10", "--limit", "3", "--subject", "Management"],
                    dict(base, **extra))
            assert p.returncode == want, (folder, extra, p.returncode, p.stdout[-600:])
            assert ("THINKING CAME BACK ON" in p.stdout) == (want == 3), (folder, extra)
            assert os.path.exists(marker) == (want == 3), (folder, extra)
            rows = [json.loads(l) for l in open(glob.glob(os.path.join(
                mm, folder, "smoke_sample10", "Management", "*.jsonl"))[0])]
            calls = len(open(log).read().splitlines()) - 1            # minus the negotiation probe
            # halted at the FIRST offending row: one call made, that row saved as evidence
            assert (len(rows), calls) == ((1, 1) if want == 3 else (3, 3)), (folder, len(rows), calls)
            row = rows[0]
            if "FAKE_REASONING" in extra:
                assert row["reasoning_chars"] == len(extra["FAKE_REASONING"])
            if "FAKE_THINK_TAG" in extra:
                assert row["think_tag_calls"] == 1
            assert bool(row["config"].get("forbid_thinking")) == (
                folder in ("MMLU_Qwen_DeepInfra", "MMLU_Gemma_DeepInfra"))
            seen.append(f"{folder[5:]}{'+' + list(extra)[0] if extra else ''}->{p.returncode}")
        ok("B11 thinking off checked on Qwen and Gemma, halting at the first offending row (1 call, "
           "the row saved, smoke_sample10_THINKING_HALT.txt, exit 3): " + ", ".join(seen))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class _IdxClient:
    """A client that answers each sampled row by a per-row script: the next item of plan[idx]."""
    def __init__(self, core, subject, plan):
        self.data, self.plan = core.load_subject(subject), plan
        self.chat = _NS(completions=_NS(create=self._create))

    def _create(self, model, messages, **kwargs):
        q = messages[-1]["content"]
        idx = next(i for i in self.plan if self.data[i]["question"] in q)
        item = self.plan[idx](idx)
        if isinstance(item, BaseException):
            raise item
        return item


def test_resume_history():
    """The reviewer's resume_probe, as a test. A resubmit replaces a stored failed or cut row only
    with a new row, which carries the earlier runs' bill and attempts; a resubmit killed mid-row
    leaves the stored row exactly as it was."""
    core = load_core()
    real_time = core.time
    core.time = _FastTime
    tmp = tempfile.mkdtemp(prefix="mmlu_hist_")
    try:
        ids, sha = core.load_sample("sample10")
        subj = "Business_ethics"
        doomed = ids[subj][3]
        fail = lambda i: fake_response("", 120, 16384, estimated_cost=0.01)   # billed, empty
        answer = lambda i: fake_response("Final Answer: B", 120, 50, estimated_cost=0.0001)
        plan = {i: answer for i in ids[subj]}

        def once_for(client):
            return plain_once(core, client)
        cfg = {"model": "fake", "params": {"seed": 42}, "prompt": "v2", "sample": "sample10",
               "sample_sha1": sha}
        path = os.path.join(tmp, "results_sample10", subj, "fake-model-1_0.jsonl")

        def run_once(plan_now):
            return core.run_sample_subjects(tmp, "fake/model-1.0",
                                            once_for(_IdxClient(core, subj, plan_now)), cfg,
                                            "sample10", ids, [subj], save_every=3)

        def rows():
            return {json.loads(l)["idx"]: json.loads(l) for l in open(path)}

        def reset():
            core._recent.clear()
            core._consecutive[0] = 0

        # run 1: the doomed row fails all 5 attempts (5 x 16,384 billed)
        run_once({**plan, doomed: fail})
        r = rows()[doomed]
        assert r["failed"] and r["n_attempts"] == 5 and r["billed_completion_tokens"] == 81920
        billed_1 = sum(x["billed_completion_tokens"] for x in rows().values())
        reset()
        # run 2, killed while calling the failed row: the stored row stays exactly as it was
        stored = rows()[doomed]

        def killed(i):
            raise SystemExit(143)
        try:
            run_once({**plan, doomed: killed})
            raise AssertionError("the killed run did not exit")
        except SystemExit:
            pass
        after_kill = rows()
        k = after_kill[doomed]
        # still there, still failed, nothing of its bill lost; the killed run joins its history
        assert len(after_kill) == len(ids[subj]) and k["failed"] and k["halted"].startswith("SystemExit")
        assert (k["billed_completion_tokens"], k["prior_n_attempts"], k["prior_failed_runs"],
                k["n_attempts"], k["attempts_without_usage"]) == (81920, 5, 1, 1, 1), k
        reset()
        # run 3: fails again — the new failed row replaces it and carries the history
        run_once({**plan, doomed: fail})
        r = rows()[doomed]
        assert r["failed"] and (r["n_attempts"], r["prior_n_attempts"], r["prior_failed_runs"],
                                r["resumed_after_failure"]) == (5, 6, 2, True)
        assert r["billed_completion_tokens"] == 2 * 81920
        reset()
        # run 4: answers — n_attempts is this run's, the bill is every run's
        run_once({**plan, doomed: answer})
        r = rows()[doomed]
        assert not r["failed"] and (r["n_attempts"], r["prior_n_attempts"], r["prior_failed_runs"],
                                    r["resumed_after_failure"]) == (1, 11, 3, True)
        assert r["billed_completion_tokens"] == 2 * 81920 + 50
        assert abs(r["billed_estimated_cost"] - (10 * 0.01 + 0.0001)) < 1e-9
        total = sum(x["billed_completion_tokens"] for x in rows().values())
        assert total == billed_1 + 81920 + 50, (total, billed_1)    # every token billed, once
        others = [x for i, x in rows().items() if i != doomed]
        assert all(x["prior_n_attempts"] == 0 and not x["resumed_after_failure"] for x in others)
        reset()

        # a CutRetry config: a stored row still cut at the cap is replaced the same way
        shutil.rmtree(tmp)
        cfg = dict(cfg, forced_answer={"prompt": "F", "max_tokens": 8})
        cut = lambda i: fake_response("half an answer", 120, 8192, finish="length")
        run_once({**plan, doomed: cut})
        r = rows()[doomed]
        assert core.is_cut(r) and not r["failed"]
        run_once({**plan, doomed: answer})
        r = rows()[doomed]
        assert (r["model_response"], r["prior_n_attempts"], r["prior_failed_runs"],
                r["resumed_after_failure"], r["billed_completion_tokens"]) == (
            "Final Answer: B", 1, 1, True, 8192 + 50)
        # a resubmit killed BEFORE it reaches a stored failed row leaves that row in the file
        shutil.rmtree(tmp)
        cfg.pop("forced_answer", None)
        first, later = ids[subj][2], ids[subj][7]
        run_once({**plan, first: fail, later: fail})
        reset()
        stored_later = rows()[later]
        try:
            run_once({**plan, first: killed, later: answer})
        except SystemExit:
            pass
        after = rows()
        assert after[later] == stored_later and after[first]["halted"] and len(after) == len(ids[subj])
        reset()
        ok("B12 resume history: a row failed 5x (81,920 billed), then a resubmit killed on it writes "
           "it as a `halted` failed row with the bill intact and the killed run in its history; "
           "failing again folds to prior_n_attempts 6 / prior_failed_runs 2; answering then gives "
           "n_attempts 1, prior_n_attempts 11, prior_failed_runs 3, "
           "billed_completion_tokens 163,890 and the file total equals every token billed; a cut "
           "row under CutRetry is replaced with its history the same way; a resubmit killed before "
           "reaching a stored failed row leaves that row untouched")
    finally:
        core.time = real_time
        core._recent.clear()
        core._consecutive[0] = 0
        shutil.rmtree(tmp, ignore_errors=True)


def test_watchdog_rearms_across_cutretry():
    """The reviewer's wd_cutretry_probe, as a test: the real Qwen runner's `once`, the real SDK, a
    local server whose first reply is cut at the cap and whose every later request trickles. The
    watchdog re-arms for the retry stage and ends it; the kept first stage is billed once."""
    import importlib.util
    import socket
    from openai import OpenAI
    count = [0]

    def serve():
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)

        def handle(c):
            _read_request(c)
            count[0] += 1
            if count[0] == 1:
                time.sleep(2)            # a slow first stage: the retry must still get its own 3 s
                js = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                 "choices": [{"index": 0, "finish_reason": "length", "message": {
                                     "role": "assistant", "content": "partial reasoning ..."}}],
                                 "usage": {"prompt_tokens": 100, "completion_tokens": 8192,
                                           "total_tokens": 8292}}).encode()
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d"
                          b"\r\n\r\n" % len(js) + js)
                return
            c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                      b"Transfer-Encoding: chunked\r\n\r\n")
            try:
                while True:
                    c.sendall(b"1\r\n\n\r\n")
                    time.sleep(0.5)
            except Exception:
                pass

        def loop():
            while True:
                c, _ = srv.accept()
                threading.Thread(target=handle, args=(c,), daemon=True).start()
        threading.Thread(target=loop, daemon=True).start()
        return srv.getsockname()[1]

    tmp = tempfile.mkdtemp(prefix="mmlu_wdcut_")
    core = load_core()
    real_time, real_limit = core.time, core.HARD_CALL_TIMEOUT
    try:
        mm = build_tree(os.path.join(tmp, "new"), "new")
        spec = importlib.util.spec_from_file_location(
            "qwen_runner_wd", os.path.join(mm, "MMLU_Qwen_DeepInfra", "qwen_di_mmlu_sample.py"))
        qwen = importlib.util.module_from_spec(spec)
        # the runner must import the TREE's core, whose ENV_PATH has no .env — not the cached repo
        # core, whose load_dotenv would read the real keys into this process
        cached = sys.modules.pop("mmlu_eval_core", None)
        had_key = "DEEPINFRA_API_KEY" in os.environ
        os.environ.setdefault("DEEPINFRA_API_KEY", "dummy")     # the import builds a client
        try:
            spec.loader.exec_module(qwen)
        finally:
            if cached is not None:
                sys.modules["mmlu_eval_core"] = cached
            if not had_key:
                os.environ.pop("DEEPINFRA_API_KEY", None)
        qcore = qwen.core
        assert qcore is not core and qcore.ENV_PATH.startswith(tmp)
        port = serve()
        qwen.client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="k", timeout=1500,
                             max_retries=0)
        qcore.HARD_CALL_TIMEOUT = 3
        qcore.time = _FastTime
        msgs = qcore.build_messages(qcore.load_subject("Econometrics")[0])
        t0 = time.time()
        text, meta = qcore.call_with_retries(
            lambda m: qwen.once(m, "Qwen/Qwen3.5-9B", dict(qwen.WANTED)), msgs, "qwen", tmp, tries=2)
        elapsed = time.time() - t0
        # attempt 1 = 2 s first stage + a full 3 s for the retry (re-armed), attempt 2 = 3 s for the
        # retry (the first stage kept): ~8.2 s. Without the re-arm attempt 1 ends at 3 s: ~6.2 s.
        assert text == "" and meta["n_attempts"] == 2 and 7.5 < elapsed < 11, (text, elapsed)
        assert (meta["billed_completion_tokens"], meta["billed_prompt_tokens"],
                meta["attempts_without_usage"]) == (8192, 100, 2), meta
        ok(f"B14 watchdog across CutRetry stages (real Qwen runner, real SDK, local server): first "
           f"stage cut at 8192 after 2 s, retry stage trickling — the retry got its own 3 s (re-armed), "
           f"ended in {elapsed:.1f}s over 2 attempts; first stage billed once (8192), 2 hung calls "
           f"in attempts_without_usage")
    finally:
        core.time, core.HARD_CALL_TIMEOUT = real_time, real_limit
        shutil.rmtree(tmp, ignore_errors=True)


FAKE_CUT = r"""
# the reviewer's fakecut: first and retry stages end at length; the forced stage answers
import json, os
class _NS:
    def __init__(self, **kw): self.__dict__.update(kw)
class _C:
    def __init__(self, o): self.o = o
    def create(self, model, messages, **kw):
        log = os.environ.get("FAKE_LOG")
        if log:
            with open(log, "a") as fh:
                fh.write(json.dumps({"roles": [m["role"] for m in messages], "kw": kw,
                                     "first": messages[0]["content"][:40]}) + "\n")
        last = messages[-1]["content"]
        mode = os.environ.get("FAKE_MODE", "cut")
        reasoning = None
        if last == "Reply with: ok":
            text, fin, ct = "ok", "stop", 1
        elif (messages[-2]["role"] == "assistant") if len(messages) > 1 else False:
            text, fin, ct = "Final Answer: B", "stop", 6
        else:
            fin, ct = "length", kw.get("max_tokens", 0)
            text = "" if mode == "deepseek" else "long visible reasoning that never ends"
            reasoning = "hidden chain of thought ..." if mode == "deepseek" else None
        u = _NS(prompt_tokens=100, completion_tokens=ct,
                completion_tokens_details=_NS(reasoning_tokens=ct if reasoning else None))
        return _NS(choices=[_NS(message=_NS(content=text, reasoning_content=reasoning),
                                finish_reason=fin)], usage=u, model=model)
class OpenAI:
    def __init__(self, *a, **kw): self.chat = _NS(completions=_C(self))
"""


def test_cut_path_end_to_end():
    """The reviewer's cut_probe, as a test: Qwen and Deepseek runners, first and retry cut, forced
    answer — the row, its bill and the requests each stage sent."""
    tmp = tempfile.mkdtemp(prefix="mmlu_cut_")
    try:
        mm = build_tree(os.path.join(tmp, "new"), "new")
        fake = os.path.join(tmp, "fakecut")
        os.makedirs(fake)
        open(os.path.join(fake, "openai.py"), "w").write(FAKE_CUT)
        notes = []
        for folder, mode, cap in (("MMLU_Qwen_DeepInfra", "qwen", 8192),
                                  ("MMLU_Deepseek", "deepseek", 16384)):
            log = os.path.join(tmp, f"{mode}.log")
            env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
            env.update(PYTHONPATH=fake, FAKE_LOG=log, FAKE_MODE=mode)
            p = run(mm, folder, SAMPLE_RUNNERS[folder][0], ["--limit", "1", "--subject",
                                                            "Management"], env)
            assert p.returncode == 0, (folder, p.stdout[-800:])
            reqs = [json.loads(l) for l in open(log)][1:]          # drop the negotiation probe
            assert [q["roles"] for q in reqs] == [["system", "user"], ["system", "user"],
                                                  ["system", "user", "assistant", "user"]]
            assert [q["kw"]["max_tokens"] for q in reqs] == [cap, cap, 4096]
            if mode == "qwen":
                assert all(q["kw"]["extra_body"] == {"reasoning": {"enabled": False}} for q in reqs)
            else:
                assert "extra_body" not in reqs[0]["kw"] and "extra_body" not in reqs[1]["kw"]
                assert reqs[2]["kw"]["extra_body"] == {"thinking": {"type": "disabled"}}
            row = json.loads(open(glob.glob(os.path.join(
                mm, folder, "smoke_sample10", "Management", "*.jsonl"))[0]).readline())
            assert row["cut_retry"] and row["forced_answer"] and row["n_attempts"] == 1
            assert row["billed_completion_tokens"] == row["completion_tokens"] == 2 * cap + 6
            assert row["attempts_without_usage"] == 0 and row["forced_reasoning_chars"] == 0
            want_rc = 0 if mode == "qwen" else 2 * len("hidden chain of thought ...")
            assert row["reasoning_chars"] == want_rc, row["reasoning_chars"]
            notes.append(f"{mode}: 3 calls, billed {row['billed_completion_tokens']}, "
                         f"reasoning_chars {row['reasoning_chars']}")
        ok("B15 cut path end to end — first/retry cut, forced answer at max_tokens 4096 (Qwen with "
           "reasoning off on every stage, Deepseek with thinking disabled on the forced one only): "
           + "; ".join(notes))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_merge_refuses_damage():
    """The reviewer's merge_probe, as a test: the merge exits 1 on a missing shard, a partial
    shard, a duplicated row and a failed row, and 0 once restored."""
    tmp = tempfile.mkdtemp(prefix="mmlu_merge_")
    try:
        mm = build_tree(os.path.join(tmp, "new"), "new")
        folder = "MMLU_Gemma_DeepInfra"
        script, model = SAMPLE_RUNNERS[folder]
        env = fake_env(tmp, os.path.join(tmp, "m.log"))
        md = os.path.join(mm, folder)
        for shard in range(5):
            p = run(mm, folder, script, ["--shard", str(shard), "--total-shards", "5"], env)
            assert p.returncode == 0, p.stderr[-500:]

        def merge():
            return subprocess.run([sys.executable, "../merge_mmlu_shards.py", "--model", model,
                                   "--model-dir", ".", "--total-shards", "5", "--sample",
                                   "sample10"], cwd=md, env=env, capture_output=True,
                                  text=True).returncode
        assert merge() == 0
        slug = "google-gemma-4-31B-it"
        d = os.path.join(md, "results_sample10", "Econometrics")
        f, g = (os.path.join(d, f"{slug}_shard{k}of5.jsonl") for k in (2, 3))
        lines, glines = open(f).read().splitlines(), open(g).read().splitlines()
        results = {}
        shutil.move(f, f + ".bak")
        results["missing shard"] = merge()
        shutil.move(f + ".bak", f)
        open(f, "w").write("\n".join(lines[:-1]) + "\n")
        results["partial shard"] = merge()
        open(f, "w").write("\n".join(lines) + "\n")
        open(g, "w").write("\n".join(glines + [lines[0]]) + "\n")
        results["duplicate row"] = merge()
        open(g, "w").write("\n".join(glines) + "\n")
        r = json.loads(lines[0])
        r.update(model_response="", failed=True, score=0, parse_ok=False)
        open(f, "w").write("\n".join([json.dumps(r)] + lines[1:]) + "\n")
        results["failed row"] = merge()
        open(f, "w").write("\n".join(lines) + "\n")
        results["restored"] = merge()
        r = json.loads(lines[0])
        r["reasoning_chars"] = 12                       # a Gemma row that came back with reasoning
        open(f, "w").write("\n".join([json.dumps(r)] + lines[1:]) + "\n")
        p = subprocess.run([sys.executable, "../merge_mmlu_shards.py", "--model", model, "--model-dir",
                            ".", "--total-shards", "5", "--sample", "sample10"], cwd=md, env=env,
                           capture_output=True, text=True)
        results["Gemma row with reasoning"] = p.returncode
        assert "thinking came back on" in p.stdout
        open(f, "w").write("\n".join(lines) + "\n")
        # Deepseek reasons by design: its rows carry reasoning and the merge must not alarm
        dfolder = "MMLU_Deepseek"
        dscript, dmodel = SAMPLE_RUNNERS[dfolder]
        denv = dict(env, FAKE_REASONING="hidden chain of thought", FAKE_REASONING_TOKENS="40")
        for shard in range(5):
            p = run(mm, dfolder, dscript, ["--shard", str(shard), "--total-shards", "5"], denv)
            assert p.returncode == 0, p.stdout[-400:]
        p = subprocess.run([sys.executable, "../merge_mmlu_shards.py", "--model", dmodel,
                            "--model-dir", ".", "--total-shards", "5", "--sample", "sample10"],
                           cwd=os.path.join(mm, dfolder), env=denv, capture_output=True, text=True)
        drows = [json.loads(l) for l in open(glob.glob(os.path.join(
            mm, dfolder, "results_sample10", "Management", "*-flash.jsonl"))[0])]
        assert all(x["reasoning_chars"] > 0 and x["billed_reasoning_tokens"] == 40 for x in drows)
        results["Deepseek rows with reasoning"] = p.returncode
        assert results == {"missing shard": 1, "partial shard": 1, "duplicate row": 1,
                           "failed row": 1, "restored": 0, "Gemma row with reasoning": 1,
                           "Deepseek rows with reasoning": 0}, results
        ok(f"B16 merge exit codes: {results}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_request_kwargs():
    """The reviewer's kwargs_probe, as a test: what each runner actually sends — the settled config,
    and for Gemma no reasoning parameter at all."""
    tmp = tempfile.mkdtemp(prefix="mmlu_kw_")
    want = {"MMLU_GPT_5.6_Luna": {"max_completion_tokens": 16384, "reasoning_effort": "low", "seed": 42},
            "MMLU_Gemini_Flash3.5lite_OpenRouter": {"max_tokens": 8192, "seed": 42,
                                                    "extra_body": {"reasoning": {"effort": "minimal"}}},
            "MMLU_Qwen_DeepInfra": {"temperature": 0, "max_tokens": 8192, "seed": 42,
                                    "extra_body": {"reasoning": {"enabled": False}}},
            "MMLU_Gemma_DeepInfra": {"temperature": 0, "max_tokens": 4096, "seed": 42},
            "MMLU_Deepseek": {"temperature": 0, "max_tokens": 16384, "seed": 42}}
    try:
        mm = build_tree(os.path.join(tmp, "new"), "new")
        for folder, (script, model) in SAMPLE_RUNNERS.items():
            log = os.path.join(tmp, f"{folder}.log")
            p = run(mm, folder, script, ["--model", model, "--sample", "sample10", "--limit", "1",
                                         "--subject", "Management"], fake_env(tmp, log))
            assert p.returncode == 0, p.stderr[-500:]
            reqs = [json.loads(l) for l in open(log)]
            assert len(reqs) == 2 and all(q["kw"] == want[folder] for q in reqs), (folder, reqs)
            assert reqs[0]["client"]["max_retries"] == 0
        ok("B17 request kwargs per runner = the settled config exactly (probe and call); Gemma sends "
           "no reasoning parameter; every client max_retries=0")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


FAKE_EDGE = r"""
# Deepseek's CutRetry edge: the retry is cut at length with content but no reasoning_content
import json, os
class _NS:
    def __init__(self, **kw): self.__dict__.update(kw)
class _C:
    def create(self, model, messages, **kw):
        log = os.environ.get("FAKE_LOG")
        if log:
            with open(log, "a") as fh:
                fh.write(json.dumps({"roles": [m["role"] for m in messages]}) + "\n")
        first = messages[0]["content"]
        if messages[-1]["content"] == "Reply with: ok":
            text, rc, fin, ct = "ok", None, "stop", 1
        elif os.environ.get("FAKE_STAGE_MODE") == "ok":
            text, rc, fin, ct = "Final Answer: B", None, "stop", 20
        elif first.startswith("Think briefly"):
            text, rc, fin, ct = "", "hidden thought that ran out", "length", 16384
        else:
            text, rc, fin, ct = "Partial answer, B maybe", None, "length", 16384
        u = _NS(prompt_tokens=100, completion_tokens=ct,
                completion_tokens_details=_NS(reasoning_tokens=ct if rc else 0))
        return _NS(choices=[_NS(message=_NS(content=text, reasoning_content=rc), finish_reason=fin)],
                   usage=u, model=model)
class OpenAI:
    def __init__(self, *a, **kw): self.chat = _NS(completions=_C())
"""


def test_deepseek_retry_cut_with_content():
    """plan_eval_core.py:1024-1029, here: a Deepseek retry cut at length with content but no
    reasoning has nothing to hand back, so no forced answer — the row stays cut (`is_cut`), the shard
    exits 1, and the resubmit calls it again and folds its history."""
    tmp = tempfile.mkdtemp(prefix="mmlu_edge_")
    try:
        mm = build_tree(os.path.join(tmp, "new"), "new")
        fake = os.path.join(tmp, "fakeedge")
        os.makedirs(fake)
        open(os.path.join(fake, "openai.py"), "w").write(FAKE_EDGE)
        folder = "MMLU_Deepseek"
        log = os.path.join(tmp, "edge.log")
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        env.update(PYTHONPATH=fake, FAKE_LOG=log, FAKE_STAGE_MODE="edge")
        p = run(mm, folder, SAMPLE_RUNNERS[folder][0], ["--subject", "Management"], env)
        assert p.returncode == 1 and "cut row(s) in this shard" in p.stdout, p.stdout[-600:]
        assert not any(json.loads(l)["roles"][-2:-1] == ["assistant"] for l in open(log))  # no forced
        path = glob.glob(os.path.join(mm, folder, "results_sample10", "Management", "*.jsonl"))[0]
        rows = [json.loads(l) for l in open(path)]
        assert rows and all(r["finish_reason"] == "length" and r["cut_retry"]
                            and not r.get("forced_answer") and not r["failed"]
                            and r["model_response"] == "Partial answer, B maybe" for r in rows)
        n = len(rows)
        env["FAKE_STAGE_MODE"] = "ok"
        p = run(mm, folder, SAMPLE_RUNNERS[folder][0], ["--subject", "Management"], env)
        assert p.returncode == 0, p.stdout[-600:]
        rows = [json.loads(l) for l in open(path)]
        assert len(rows) == n and all(
            r["model_response"] == "Final Answer: B" and r["resumed_after_failure"]
            and r["prior_failed_runs"] == 1 and r["prior_n_attempts"] == 1
            and r["billed_completion_tokens"] == 2 * 16384 + 20 for r in rows), rows[0]
        ok(f"B19 Deepseek retry cut at length with content and no reasoning: no forced call, all {n} "
           f"rows kept as cut (finish length, cut_retry, no forced_answer), shard exit 1; the "
           f"resubmit re-called every one, exit 0, history folded (billed 2x16384+20)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_parse_ok_never_false_when_scored():
    """parse_ok = the scorer accepts at least one option, asked about each in turn. The gold option
    is one of those four calls, made with exactly the arguments score_response got, so score == 1
    implies parse_ok. Checked on every stored response the core-based folders hold, every smoke row,
    and edge cases including bare answers with no marker."""
    core = load_core()
    checked = scored = 0
    files = (glob.glob(os.path.join(HERE, "MMLU_GPT_5.6_Luna", "results", "*", "*.jsonl"))
             + glob.glob(os.path.join(HERE, "MMLU_Gemini_Flash3.5lite_Google", "results", "*", "*.jsonl"))
             + glob.glob(os.path.join(HERE, "MMLU_*", "smoke_sample10", "*", "*.jsonl")))
    files = [f for f in files if "_shard" not in os.path.basename(f)]
    for f in files:
        for line in open(f):
            r = json.loads(line)
            resp, choices, k = r.get("model_response") or "", r["choices"], int(r["answer_index"])
            ex = {"subject": r["subject"], "question": r["question"], "choices": choices,
                  "answer": str(k)}
            row = core.sample_row(r["idx"], ex, resp, {}, {}, "sample10")   # as the run writes it
            assert row["score"] == core.score_response(resp, choices[k], core.LABELS[k], k)
            assert not row["score"] or row["parse_ok"], (f, r["idx"])
            checked += 1
            scored += row["score"]
    choices = ["1 and 2", "1 and 0.5", "2 and -0.5", "1 and -3"]
    edge = ["B", "(B)", "b.", "B. 1 and 0.5", "1 and 0.5", "1", "Final Answer: B", "Final Answer: <B>",
            "A good guess", "Final Answer: The answer is B", "1 and 0.5.", "", "   "]
    ex = {"subject": "econometrics", "question": "q", "choices": choices, "answer": "1"}
    for resp in edge:
        row = core.sample_row(0, ex, resp, {}, {}, "sample10")
        assert not row["score"] or row["parse_ok"], resp
        checked += 1
    ok(f"B20 parse_ok is never False on a row scored 1: {checked} responses checked ({scored} scored "
       f"1) — every stored core-run response, every smoke row, and bare/no-marker edge cases")


def test_thinking_row_stops_resubmit():
    """Addendum D: a resubmit or resume that finds a stored row already showing thinking stops before
    calling any row — the negotiation probe aside — rewrites THINKING_HALT and exits 3, even when the
    thinking row is in a later subject than rows still to call."""
    tmp = tempfile.mkdtemp(prefix="mmlu_think2_")
    try:
        mm = build_tree(os.path.join(tmp, "new"), "new")
        notes = []
        for folder, bad in (("MMLU_Gemma_DeepInfra", {"FAKE_REASONING_TOKENS": "7"}),
                            ("MMLU_Qwen_DeepInfra", {"FAKE_REASONING": "thinking again"})):
            script = SAMPLE_RUNNERS[folder][0]
            log = os.path.join(tmp, f"{folder}.log")
            marker = os.path.join(mm, folder, "sample10_THINKING_HALT.txt")
            p = run(mm, folder, script, ["--subject", "Management"], dict(fake_env(tmp, log), **bad))
            assert p.returncode == 3 and os.path.exists(marker), (folder, p.returncode)
            os.remove(log)
            os.remove(marker)
            # the resubmit: a clean provider, every subject — Business_ethics..Logical_fallacies
            # come before Management and all still have rows to call
            p = run(mm, folder, script, ["--subject", "all"], fake_env(tmp, log))
            reqs = [json.loads(l) for l in open(log)]
            calls = [q for q in reqs if q["messages"][-1]["content"] != "Reply with: ok"]
            assert p.returncode == 3 and len(calls) == 0, (folder, p.returncode, len(calls))
            assert os.path.exists(marker) and "THINKING CAME BACK ON" in open(marker).read()
            assert not os.path.exists(os.path.join(mm, folder, "results_sample10", "Business_ethics"))
            notes.append(f"{folder[5:]}: exit 3, {len(reqs)} request(s) (probe only), 0 row calls")
        ok("B21 resubmit after a thinking row: " + "; ".join(notes) + "; THINKING_HALT rewritten")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_in_flight_row_is_billed():
    """The reviewer's inflight_probe (K1-K5), as a test: a run stopped mid-row — SIGTERM's
    SystemExit, a billing halt, FAILURE_HALT — still writes that row, failed and `halted` where it was
    stopped mid-call, with every call it was billed; the file's billed total equals what was billed."""
    core = load_core()
    real_time = core.time
    core.time = _FastTime
    tmp = tempfile.mkdtemp(prefix="mmlu_inflight_")
    billed = [0]
    ids, sha = core.load_sample("sample10")
    cfg = {"model": "fake", "params": {"seed": 42}, "prompt": "v2", "sample": "sample10",
           "sample_sha1": sha}

    def resp(text, c, finish="stop"):
        billed[0] += c
        return fake_response(text, 0, c, finish=finish)

    def run_rows(md, subj, script, config):
        data, n = core.load_subject(subj), {}

        class C:
            class chat:
                class completions:
                    @staticmethod
                    def create(model, messages, **kw):
                        q = messages[-1]["content"]
                        i = next(i for i in ids[subj] if data[i]["question"] in q)
                        n[i] = n.get(i, 0) + 1
                        r = script(i, n[i])
                        if isinstance(r, BaseException):
                            raise r
                        return r
        core._recent.clear()
        core._consecutive[0] = 0
        try:
            core.run_sample_subjects(md, "fake", plain_once(core, C), config, "sample10", ids,
                                     [subj], save_every=3)
            return 0
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else str(e.code)[:40]

    def file_rows(md, subj):
        return {json.loads(l)["idx"]: json.loads(l)
                for l in open(os.path.join(md, "results_sample10", subj, "fake.jsonl"))}

    def total(md, subj):
        return sum(r["billed_completion_tokens"] or 0 for r in file_rows(md, subj).values())

    try:
        subj = "Business_ethics"
        doomed = ids[subj][3]
        ok_ = lambda: resp("Final Answer: B", 50)
        empty = lambda: resp("", 16384)
        out = []

        # K1: the resubmit of a stored failed row is killed after one more billed attempt
        md = os.path.join(tmp, "k1")
        billed[0] = 0
        run_rows(md, subj, lambda i, c: empty() if i == doomed else ok_(), cfg)
        code = run_rows(md, subj, lambda i, c: (empty() if c == 1 else SystemExit(143))
                        if i == doomed else ok_(), cfg)
        r = file_rows(md, subj)[doomed]
        assert code == 143 and total(md, subj) == billed[0], (total(md, subj), billed[0])
        assert r["failed"] and r["halted"].startswith("SystemExit") and r["prior_failed_runs"] == 1
        assert r["n_attempts"] == 2 and r["attempts_without_usage"] == 1
        out.append(f"K1 file {total(md, subj)} = billed {billed[0]}")

        # K2: a first run killed mid-row after 2 billed attempts
        md = os.path.join(tmp, "k2")
        billed[0] = 0
        code = run_rows(md, subj, lambda i, c: (empty() if c <= 2 else SystemExit(143))
                        if i == doomed else ok_(), cfg)
        r = file_rows(md, subj)[doomed]
        assert code == 143 and total(md, subj) == billed[0] and r["halted"] and r["n_attempts"] == 3
        out.append(f"K2 {total(md, subj)} = {billed[0]}")

        # K3: a billing halt (402) on the 3rd attempt after 2 billed ones
        md = os.path.join(tmp, "k3")
        billed[0] = 0
        code = run_rows(md, subj, lambda i, c: (empty() if c <= 2 else
                                                RuntimeError("Error code: 402 - Insufficient Balance"))
                        if i == doomed else ok_(), cfg)
        r = file_rows(md, subj)[doomed]
        assert total(md, subj) == billed[0] and "billing" in str(code).lower() and r["halted"]
        assert os.path.exists(os.path.join(md, "BILLING_HALT.txt"))
        # the 402 call raised, its handler raised the halt: that call is counted exactly once
        assert (r["n_attempts"], r["attempts_without_usage"]) == (3, 1), r
        out.append(f"K3 {total(md, subj)} = {billed[0]}")

        # K6: an ordinary error, then a halt raised from inside the handler of the next error —
        # each call counted once; and a malformed-request halt on the first attempt — once
        for label, script, want in (
                ("503 then 402", lambda i, c: RuntimeError("Error code: 503 - upstream connect error")
                 if c == 1 else RuntimeError("Error code: 402 - Insufficient Balance"), (2, 2)),
                ("malformed first", lambda i, c: RuntimeError(
                    "Error code: 400 - Unsupported parameter: 'seed'"), (1, 1))):
            md = os.path.join(tmp, "k6" + label.replace(" ", "_"))
            code = run_rows(md, subj, lambda i, c, f=script: f(i, c) if i == doomed else ok_(), cfg)
            r = file_rows(md, subj)[doomed]
            assert r["halted"] and (r["n_attempts"], r["attempts_without_usage"]) == want, (label, r)
        out.append("K6 a halt raised inside the error handler counts that call once")

        # K4: FAILURE_HALT on the 10th consecutive failed row — that row is written too
        subj2 = "Elementary_math"
        md = os.path.join(tmp, "k4")
        billed[0] = 0
        code = run_rows(md, subj2, lambda i, c: empty(), cfg)
        rows4 = file_rows(md, subj2)
        assert "failure" in str(code).lower() and len(rows4) == 10 and total(md, subj2) == billed[0], (
            code, len(rows4), total(md, subj2), billed[0])
        assert all(x["failed"] and not x["halted"] for x in rows4.values())
        out.append(f"K4 {len(rows4)} rows, {total(md, subj2)} = {billed[0]}")

        # K5: a stored cut row whose re-call fails is replaced by the failed row — billing intact
        md = os.path.join(tmp, "k5")
        billed[0] = 0
        cfg_cut = dict(cfg, forced_answer={"prompt": "F", "max_tokens": 8})
        run_rows(md, subj, lambda i, c: resp("... Final Answer: B", 8192, "length")
                 if i == doomed else ok_(), cfg_cut)
        run_rows(md, subj, lambda i, c: empty() if i == doomed else ok_(), cfg_cut)
        r = file_rows(md, subj)[doomed]
        assert r["failed"] and r["prior_failed_runs"] == 1 and total(md, subj) == billed[0]
        assert r["billed_completion_tokens"] == 8192 + 5 * 16384
        out.append(f"K5 {total(md, subj)} = {billed[0]}")
        ok("B22 the row in flight is billed: " + "; ".join(out) + " — killed rows marked `halted`, "
           "the 10th FAILURE_HALT row written, a failed re-call of a cut row replaces it, loudly")
    finally:
        core.time = real_time
        core._recent.clear()
        core._consecutive[0] = 0
        shutil.rmtree(tmp, ignore_errors=True)


def test_held_row_windows():
    """No window loses or doubles a stored row. (a) A stop after the stored row was taken but before
    its replacement is appended (injected: fold_history raises) leaves the stored row in the file.
    (b) A stop between the append and the pop (injected: held.pop raises) writes the idx once — the
    new row."""
    core = load_core()
    real = (core.time, core.fold_history, core.load_sample_checkpoint)
    core.time = _FastTime
    tmp = tempfile.mkdtemp(prefix="mmlu_window_")
    try:
        ids, sha = core.load_sample("sample10")
        subj = "Business_ethics"
        doomed = ids[subj][3]
        cfg = {"model": "fake", "params": {"seed": 42}, "prompt": "v2", "sample": "sample10",
               "sample_sha1": sha}
        fail = lambda i: fake_response("", 120, 16384)
        answer = lambda i: fake_response("Final Answer: B", 120, 50)
        plan = {i: answer for i in ids[subj]}
        path = os.path.join(tmp, "results_sample10", subj, "fake-model-1_0.jsonl")

        def go(plan_now):
            core._recent.clear()
            core._consecutive[0] = 0
            try:
                core.run_sample_subjects(tmp, "fake/model-1.0",
                                         plain_once(core, _IdxClient(core, subj, plan_now)), cfg,
                                         "sample10", ids, [subj], save_every=3)
            except SystemExit:
                pass
            return [json.loads(l) for l in open(path)]
        go({**plan, doomed: fail})

        # (a) the stop lands while the replacement is built, and keeps landing: the stored row stays
        def fold_stop(new, old):
            raise SystemExit(143)
        core.fold_history = fold_stop
        rows = go(plan)
        mine = [r for r in rows if r["idx"] == doomed]
        assert len(mine) == 1 and mine[0]["failed"] and mine[0]["n_attempts"] == 5, mine
        core.fold_history = real[1]

        # (b) the stop lands between the append and the pop
        class StopPop(dict):
            def pop(self, *a):
                raise SystemExit(143)

        def checkpoint(path, config, cut_retry=False):
            done, held = real[2](path, config, cut_retry)
            return done, StopPop(held)
        core.load_sample_checkpoint = checkpoint
        rows = go(plan)
        mine = [r for r in rows if r["idx"] == doomed]
        assert len(mine) == 1 and not mine[0]["failed"] and mine[0]["resumed_after_failure"], mine
        assert len(rows) == len({r["idx"] for r in rows}) == len(ids[subj])
        core.load_sample_checkpoint = real[2]

        # (c) a stored failed row again; the kill lands once, after the call returned, while its
        # row is written: the row is written as answered, its history folded in
        shutil.rmtree(tmp)
        go({**plan, doomed: fail})
        once_more = [True]

        def fold_once(new, old):
            if once_more[0]:
                once_more[0] = False
                raise SystemExit(143)
            return real[1](new, old)
        core.fold_history = fold_once
        rows = go(plan)
        mine = [r for r in rows if r["idx"] == doomed]
        assert len(mine) == 1 and not mine[0]["failed"] and mine[0]["resumed_after_failure"], mine
        assert mine[0]["billed_completion_tokens"] == 5 * 16384 + 50 and not mine[0]["halted"]
        core.fold_history = real[1]

        # (d) the same with no stored row: a first run killed while an answered row is written
        shutil.rmtree(tmp)
        real_row, hit = core.sample_row, [True]

        def row_once(idx, *a, **k):
            if idx == doomed and hit[0]:
                hit[0] = False
                raise SystemExit(143)
            return real_row(idx, *a, **k)
        core.sample_row = row_once
        try:
            rows = go(plan)
        finally:
            core.sample_row = real_row
        mine = [r for r in rows if r["idx"] == doomed]
        assert len(mine) == 1 and not mine[0]["failed"] and mine[0]["billed_completion_tokens"] == 50
        ok("B23 row-write windows: a kill after the call returned, while its row is written, writes "
           "it as answered with its history folded (with and without a stored row); a stop that "
           "keeps failing there leaves the stored row, once; a stop between the append and the pop "
           "writes the idx once, as the new row")
    finally:
        core.time, core.fold_history, core.load_sample_checkpoint = real
        core._recent.clear()
        core._consecutive[0] = 0
        shutil.rmtree(tmp, ignore_errors=True)


def test_sbatch_scripts():
    """Each run_mmlu_sample10.sh runs in its own folder wherever it is submitted from, logs to
    absolute paths there in append mode, and runs the sample, sharded, unbuffered."""
    import re
    for folder, (script, model) in SAMPLE_RUNNERS.items():
        text = open(os.path.join(HERE, folder, "run_mmlu_sample10.sh")).read()
        base = f"/projects/p32983/Tasks_benchmarks/mmlu/{folder}"
        assert f"#SBATCH --output={base}/log_sample10_shard%a.txt" in text, folder
        assert f"#SBATCH --error={base}/log_sample10_shard%a.err" in text, folder
        assert "#SBATCH --open-mode=append" in text and "#SBATCH --array=0-4" in text
        assert "export PYTHONUNBUFFERED=1" in text
        cd = text.index(f"\ncd {base} || exit 1\n")
        call = text.index(f"/projects/p32983/pythonenvs/hai-teams/bin/python {script} \\")
        assert cd < call, folder
        tail = text[call:]
        for arg in (f"--model {model}", "--sample sample10", "--prompt v2",
                    '--shard "$SLURM_ARRAY_TASK_ID"', "--total-shards 5"):
            assert arg in tail, (folder, arg)
        assert "CLOCK_PENDING" not in text and not re.search(r"--workers", tail)
    ok("B18 sbatch: absolute log paths in each folder with --open-mode=append, `cd <folder> || exit 1` "
       "before the python line, --sample sample10, 5 shards, PYTHONUNBUFFERED")


if __name__ == "__main__":
    test_static_core()
    test_legacy_runs_identical()
    test_sample_runners_end_to_end()
    test_serial_only()
    test_retries_attempts_latency_and_failed_rows()
    test_resume_refuses_changed_config_and_foreign_rows()
    test_smoke_dir_and_halt()
    test_cutretry()
    test_runner_wordings()
    test_billed_usage()
    test_watchdog_blocked_socket()
    test_status_codes_are_words()
    test_thinking_off_is_checked()
    test_resume_history()
    test_watchdog_rearms_across_cutretry()
    test_cut_path_end_to_end()
    test_merge_refuses_damage()
    test_request_kwargs()
    test_deepseek_retry_cut_with_content()
    test_parse_ok_never_false_when_scored()
    test_thinking_row_stops_resubmit()
    test_in_flight_row_is_billed()
    test_held_row_windows()
    test_sbatch_scripts()
    print(f"\n{len(PASSED)} checks passed")
