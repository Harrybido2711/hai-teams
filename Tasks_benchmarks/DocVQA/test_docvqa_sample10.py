"""Offline test of `docvqa_eval_core`'s sample path, and proof that the path WITHOUT --sample is unchanged.

No network, no key, no real result file written: every run happens in a scratch copy, with a fake
`openai` package in front of the real one. Run from anywhere, with an interpreter that has
python-dotenv (the runners import it):

    python test_docvqa_sample10.py                 # every check; exit 1 if any fails, 2 if a part skipped
    python test_docvqa_sample10.py --baseline REF  # compare the legacy path against another commit

**A — legacy path, before vs after** (`--baseline`, default 1cc5bbd5: the last commit before the
sample path existed). The baseline's `docvqa_eval_core.py` and `gpt56luna_docvqa_eval.py` are read
with `git show` and run beside the working copies on identical inputs:
  A1 `core.run` with a deterministic fake call — a `--limit` run, a fresh shard, a resumed shard
     with empty rows, and a config-mismatch refusal: every file byte-identical, the same prompts
     asked in the same order, the same stdout.
  A2 the GPT runner as a subprocess (`--limit 6`, and shard 1 of 5 resumed from a copy of the real
     shard file with three rows blanked): result files byte-identical, the same client
     construction, the same requests (messages with images hashed, parameters), the same output.
  A3 the GPT runner WITH --sample: writes only smoke_sample10/, leaves results/ byte-identical,
     builds its own client with max_retries=0, and sends the same message the legacy path sent
     for the same question.
  A3 also ties every request to the client that sent it: all of the sample path's requests come from
     its max_retries=0 client, and `--save-every 0` is refused before any request.
**B — the sample path's guarantees**, with fake clients: the sample file against the data, the
output directories, the per-row fields, n_attempts and latency_s under retries, a failed row
written as failed, resume, refusals, halts and their smoke_ prefix, CutRetry, the billed usage of
every attempt (B9), numeric statuses as whole words (B10), Gemma's thinking check (B11),
`--save-every` (B12), a failed row's history across resubmits and a resubmit killed mid-row (B13,
B14, and the row in flight when a resubmit is killed or the run halts), billed costs (B15), Qwen's
and Gemma's thinking signals, inline think tags included (B16), calls without usage counted per HTTP
call, CutRetry stages included, never twice (B17), and parse_ok False meaning score 0 and ANLS 0 (B18).
B11 also checks that the first row showing thinking halts the run with THINKING_HALT.txt. **B4 needs the real openai SDK**: a local server that takes the request and
never finishes answering — silent, and keep-alive (a byte every 0.5 s, which defeats the SDK's own
timeout) — must be ended by the SIGALRM watchdog. Without the SDK, B4 is skipped and the exit is 2.
**C — the prompt**: `core.build_prompt` against the prompt in each of the four standalone runners.
"""

import argparse
import collections
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_REL = "Tasks_benchmarks/DocVQA"
GPT_DIR = "DOCVQA_GPT_5.6_Luna"
GPT_RUNNER = "gpt56luna_docvqa_eval.py"

RESULTS = []
SKIPS = []          # a part that could not run here; the exit code says so (2), it is never a pass


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    shown = detail if (not ok or len(detail) <= 100) else ""      # a pass shows only a short detail
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {detail}" if detail and not ok else
                                                   (f"  — {shown}" if shown else "")), flush=True)


def git_show(ref, rel):
    top = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], capture_output=True,
                         text=True, check=True).stdout.strip()
    return subprocess.run(["git", "-C", top, "show", f"{ref}:{REPO_REL}/{rel}"], capture_output=True,
                          check=True).stdout


def make_root(base, label, core_src, runner_src=None):
    """A DocVQA root in miniature: the core, the data and sample by symlink, optionally the GPT runner."""
    root = os.path.join(base, label)
    os.makedirs(os.path.join(root, GPT_DIR))
    with open(os.path.join(root, "docvqa_eval_core.py"), "wb") as fh:
        fh.write(core_src)
    os.symlink(os.path.join(HERE, "docvqa_output"), os.path.join(root, "docvqa_output"))
    os.symlink(os.path.join(HERE, "sample10.json"), os.path.join(root, "sample10.json"))
    if runner_src is not None:
        with open(os.path.join(root, GPT_DIR, GPT_RUNNER), "wb") as fh:
            fh.write(runner_src)
    return root


def load_core(root, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, "docvqa_eval_core.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def snapshot(d):
    out = {}
    for dirpath, _, files in os.walk(d):
        for f in files:
            p = os.path.join(dirpath, f)
            out[os.path.relpath(p, d)] = open(p, "rb").read()
    return out


def first_diff(a, b):
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            return k
    return None


# ---------------------------------------------------------------- A1: core.run, old vs new


def fake_answer(prompt, image_path, gold_of):
    """Deterministic, and exercises every scorer branch plus empty rows."""
    gold = gold_of.get((prompt, image_path), ["?"])[0]
    r = int(hashlib.sha1((prompt + image_path).encode()).hexdigest(), 16) % 6
    return ["", "Final Answer: wrong", f"It reads {gold}.", f"Final Answer: {gold.upper()}.",
            f"Final Answer: {gold}", f"The value is shown.\nFinal Answer:  {gold} "][r]


def test_a1(base, old_src, new_src):
    print("\n== A1  core.run without --sample: baseline core vs working core", flush=True)
    probe = load_core(make_root(base, "a1_probe", new_src), "a1_probe_core")
    data = probe.load_data()
    gold_of = {(probe.build_prompt(ex), ex["image_path"]): ex["answers"] for ex in data}
    model = "test/fake-model-1.0"
    cfg = {"model": model, "seed": 42, "max_tokens": 8192}

    seed_root = make_root(base, "a1_seed", old_src)
    seed_core = load_core(seed_root, "a1_seed_core")
    seed_dir = os.path.join(seed_root, "M")
    with contextlib.redirect_stdout(io.StringIO()):
        seed_core.run(seed_dir, model, lambda p, i: fake_answer(p, i, gold_of), config=cfg,
                      shard=2, total_shards=5)
    seeded = open(os.path.join(seed_dir, "results", "test-fake-model-1_0_shard2of5.jsonl")).read()
    rows = [json.loads(line) for line in seeded.splitlines()]
    for r in rows[5:8]:
        r["model_response"] = ""                  # three empties for the resume to call again
    resumable = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    other_cfg = resumable.replace("seed=42", "seed=7")

    modes = [("--limit 7", dict(limit=7, config=cfg), None),
             ("shard 2 of 5, fresh", dict(shard=2, total_shards=5, config=cfg), None),
             ("shard 2 of 5, resumed with 3 empty rows", dict(shard=2, total_shards=5, config=cfg),
              resumable),
             ("shard 2 of 5, stored config differs", dict(shard=2, total_shards=5, config=cfg), other_cfg)]
    for i, (label, kwargs, seed) in enumerate(modes):
        seen = {}
        for side, src in (("old", old_src), ("new", new_src)):
            root = make_root(base, f"a1_{i}_{side}", src)
            core = load_core(root, f"a1_{i}_{side}_core")
            mdir = os.path.join(root, "M")
            if seed is not None:
                os.makedirs(os.path.join(mdir, "results"))
                with open(os.path.join(mdir, "results", "test-fake-model-1_0_shard2of5.jsonl"), "w") as fh:
                    fh.write(seed)
            asked, out = [], io.StringIO()

            def call(prompt, image_path):
                asked.append((prompt, image_path))
                return fake_answer(prompt, image_path, gold_of)
            exit_msg = None
            with contextlib.redirect_stdout(out):
                try:
                    core.run(mdir, model, call, **kwargs)
                except SystemExit as e:
                    exit_msg = str(e).replace(root, "<root>")
            seen[side] = (snapshot(mdir), asked, out.getvalue(), exit_msg)
        o, n = seen["old"], seen["new"]
        check(f"A1 {label}: files byte-identical ({len(n[0])} files)", o[0] == n[0],
              f"first differing file: {first_diff(o[0], n[0])}")
        check(f"A1 {label}: same prompts in the same order ({len(n[1])} calls)", o[1] == n[1])
        check(f"A1 {label}: same stdout and exit", o[2] == n[2] and o[3] == n[3],
              f"old={o[3]!r} new={n[3]!r}")
        check(f"A1 {label}: no sample directory created",
              not any(k.startswith(("results_sample", "smoke_sample")) for k in n[0]))


# ---------------------------------------------------------------- A2/A3: the GPT runner, subprocess

FAKE_OPENAI = r'''
import hashlib, json, os
from types import SimpleNamespace

def _log(rec):
    with open(os.environ["FAKE_OPENAI_LOG"], "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")

def scrub(messages):
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            c = [dict(p, image_url=dict(p["image_url"], url="sha1:" + hashlib.sha1(
                     p["image_url"]["url"].encode()).hexdigest())) if p.get("type") == "image_url" else p
                 for p in c]
        out.append(dict(m, content=c))
    return out

_CLIENTS = [0]

class OpenAI:
    def __init__(self, **kw):
        _CLIENTS[0] += 1
        self.client_id = _CLIENTS[0]          # every request is logged with the client that sent it
        _log({"event": "client", "client_id": self.client_id,
              "kwargs": {k: (bool(v) if k == "api_key" else v) for k, v in kw.items()}})
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        msgs = kw["messages"]
        _log({"event": "create", "client_id": self.client_id,
              "params": {k: v for k, v in kw.items() if k != "messages"}, "messages": scrub(msgs)})
        c = msgs[-1]["content"]
        if isinstance(c, str):
            text = "ok"
        else:
            prompt = next(p["text"] for p in c if p.get("type") == "text")
            img = next(p["image_url"]["url"] for p in c if p.get("type") == "image_url")
            h = int(hashlib.sha1((prompt + img[-64:]).encode()).hexdigest(), 16)
            word = prompt.split("Question: ")[1].split("\n")[0].split()[-1].strip("?")
            text = ["", "Final Answer: wrong", "no marker here", "Final Answer: " + word][h % 4]
        usage = SimpleNamespace(prompt_tokens=1000 + len(str(c)) % 97, completion_tokens=len(text) + 3,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
        choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage, model=kw.get("model"))
'''


def fake_env(base):
    d = os.path.join(base, "fake_site")
    os.makedirs(os.path.join(d, "openai"))
    with open(os.path.join(d, "openai", "__init__.py"), "w") as fh:
        fh.write(FAKE_OPENAI)
    with open(os.path.join(d, "sitecustomize.py"), "w") as fh:
        fh.write("import time\ntime.sleep = lambda *a, **k: None   # the retry back-off, not the code under test\n")
    return d


def run_runner(root, fake_dir, log, args):
    env = dict(os.environ, PYTHONPATH=fake_dir, FAKE_OPENAI_LOG=log, OPENAI_API_KEY="test-key",
               PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, os.path.join(root, GPT_DIR, GPT_RUNNER), *args], env=env,
                       capture_output=True, text=True, cwd=os.path.join(root, GPT_DIR))
    events = [json.loads(line) for line in open(log)] if os.path.exists(log) else []
    return p.returncode, (p.stdout + p.stderr).replace(root, "<root>"), events


def test_a2_a3(base, old_core, new_core, old_runner, new_runner):
    print("\n== A2  the GPT runner without --sample: baseline vs working copy (fake openai)", flush=True)
    fake_dir = fake_env(base)
    real_shard = os.path.join(HERE, GPT_DIR, "results", "gpt-5_6-luna_shard1of5.jsonl")
    seeded = None
    if os.path.exists(real_shard):
        rows = [json.loads(line) for line in open(real_shard)]
        for r in rows[10:13]:
            r["model_response"] = ""
        seeded = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    runs = [("--limit 6", ["--limit", "6", "--sleep", "0"], None),
            ("--save-every 0 --limit 2 (still legal without --sample)",
             ["--save-every", "0", "--limit", "2", "--sleep", "0"], None)]
    if seeded:
        runs.append(("shard 1 of 5 resumed from the real shard file, 3 rows blanked",
                     ["--shard", "1", "--total-shards", "5", "--sleep", "0"], seeded))
    else:
        check("A2 real shard file present for the resume case", False, real_shard)
    legacy_messages = None
    for i, (label, args, seed) in enumerate(runs):
        seen = {}
        for side, core_src, runner_src in (("old", old_core, old_runner), ("new", new_core, new_runner)):
            root = make_root(base, f"a2_{i}_{side}", core_src, runner_src)
            if seed:
                os.makedirs(os.path.join(root, GPT_DIR, "results"))
                with open(os.path.join(root, GPT_DIR, "results", "gpt-5_6-luna_shard1of5.jsonl"), "w") as fh:
                    fh.write(seed)
            log = os.path.join(base, f"a2_{i}_{side}.log")
            code, out, events = run_runner(root, fake_dir, log, args)
            seen[side] = (code, out, events, snapshot(os.path.join(root, GPT_DIR, "results")),
                          sorted(os.listdir(os.path.join(root, GPT_DIR))))
        o, n = seen["old"], seen["new"]
        creates = [e for e in n[2] if e["event"] == "create"]
        check(f"A2 {label}: exit code and output identical (exit {n[0]})", o[0] == n[0] and o[1] == n[1],
              f"old exit {o[0]}, new exit {n[0]}\n--- old\n{o[1][-800:]}\n--- new\n{n[1][-800:]}")
        check(f"A2 {label}: result files byte-identical ({len(n[3])} files)", o[3] == n[3],
              f"first differing file: {first_diff(o[3], n[3])}")
        check(f"A2 {label}: same client construction and same requests ({len(creates)} create calls)",
              o[2] == n[2], f"old {len(o[2])} events, new {len(n[2])}")
        check(f"A2 {label}: nothing written beside results/", o[4] == n[4], f"{o[4]} vs {n[4]}")
        if i == 0:
            # one entry per question: the legacy path asks an empty reply again, up to 3 times
            legacy_messages = []
            for e in creates:
                if isinstance(e["messages"][-1]["content"], list) and (
                        not legacy_messages or legacy_messages[-1] != e["messages"]):
                    legacy_messages.append(e["messages"])
    print(f"      client as constructed: {[e for e in seen['new'][2] if e['event'] == 'client']}")

    print("\n== A3  the GPT runner WITH --sample (fake openai)", flush=True)
    root = make_root(base, "a3", new_core, new_runner)
    rdir = os.path.join(root, GPT_DIR, "results")
    os.makedirs(rdir)
    with open(os.path.join(rdir, "gpt-5_6-luna.jsonl"), "w") as fh:
        fh.write('{"sentinel": "a full-run row"}\n')
    before = snapshot(rdir)
    log = os.path.join(base, "a3.log")
    code, out, events = run_runner(root, fake_dir, log, ["--sample", "sample10", "--limit", "4", "--sleep", "0"])
    smoke = os.path.join(root, GPT_DIR, "smoke_sample10")
    check("A3 results/ untouched by a --sample run", snapshot(rdir) == before)
    check("A3 --limit under --sample writes smoke_sample10/, not results_sample10/",
          os.path.isdir(smoke) and not os.path.exists(os.path.join(root, GPT_DIR, "results_sample10")),
          out[-600:])
    clients = {e["client_id"]: e["kwargs"] for e in events if e["event"] == "client"}
    zero = [cid for cid, kw in clients.items() if kw.get("max_retries") == 0]
    creates = [e for e in events if e["event"] == "create"]
    check("A3 the sample client is built with max_retries=0", len(zero) == 1, str(clients))
    check(f"A3 every one of the {len(creates)} sample requests (probe included) came from that client",
          creates and zero and all(e["client_id"] == zero[0] for e in creates),
          str(sorted({e["client_id"] for e in creates})) + f" vs {zero}")
    log0 = os.path.join(base, "a3_save0.log")
    code0, out0, events0 = run_runner(root, fake_dir, log0,
                                      ["--sample", "sample10", "--limit", "1", "--save-every", "0"])
    check(f"A3 --save-every 0 with --sample is refused at parsing (exit {code0}, no request sent)",
          code0 == 2 and "must be 1 or more" in out0
          and not [e for e in events0 if e["event"] == "create"], out0[-300:])
    rows = [json.loads(line) for line in open(os.path.join(smoke, "gpt-5_6-luna.jsonl"))] if os.path.isdir(smoke) else []
    empties = sum(r["empty"] for r in rows)
    check(f"A3 exit code says whether rows are empty (exit {code}, {empties} empty of {len(rows)})",
          code == (1 if empties else 0))
    # the message the sample path sends == the message the legacy path sent, for the same questions
    core = load_core(root, "a3_core")
    data = core.load_data()
    ok = bool(legacy_messages)
    for ex, msg in zip(data[:6], legacy_messages or []):
        built = core.build_messages(ex)
        mine = json.loads(json.dumps(built))
        for p in mine[0]["content"]:
            if p.get("type") == "image_url":
                p["image_url"]["url"] = "sha1:" + hashlib.sha1(p["image_url"]["url"].encode()).hexdigest()
        ok = ok and mine == msg
    check(f"A3 core.build_messages == the legacy GPT message for the same {len(legacy_messages or [])} questions", ok)
    sent = [e for e in events if e["event"] == "create" and isinstance(e["messages"][-1]["content"], list)]
    check("A3 sample requests carry the settled GPT parameters",
          all(e["params"] == {"model": "gpt-5.6-luna", "max_completion_tokens": 16384,
                              "reasoning_effort": "low", "seed": 42} for e in sent) and sent,
          str(sent[:1]))


# ---------------------------------------------------------------- B: the sample path


class FakeClient:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
        self.probes = 0

    def create(self, **kw):
        self.probes += 1
        return None


def busy(seconds):
    end = time.time() + seconds
    while time.time() < end:
        pass


def sample_args(**kw):
    a = dict(sample="sample10", model="test/fake-model-1.0", limit=0, sleep=0, save_every=20, shard=0,
             total_shards=1)
    a.update(kw)
    return types.SimpleNamespace(**a)


def run_s(core, model_dir, once, **kw):
    fatal = kw.pop("fatal", ())
    wanted = kw.pop("wanted", {"max_tokens": 64, "seed": 42})
    extra = kw.pop("extra_config", None)
    call_timeout = kw.pop("call_timeout", 60)
    forbid = kw.pop("forbid_reasoning", False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            core.run_sample(sample_args(**kw), model_dir=model_dir, client=FakeClient(), once=once,
                            wanted=wanted, route="fake", fatal=fatal, call_timeout=call_timeout,
                            extra_config=extra, forbid_reasoning=forbid)
            code = None
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


def fake_resp(text, prompt=1000, completion=None, reasoning=None, finish="stop", usage=True,
              reasoning_text=None):
    """An OpenAI-SDK-shaped response. `reasoning=None` leaves completion_tokens_details out, as DeepInfra's
    Qwen does; `usage=False` is a response with no usage object at all."""
    details = types.SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None
    u = (types.SimpleNamespace(prompt_tokens=prompt,
                               completion_tokens=len(text) if completion is None else completion,
                               completion_tokens_details=details) if usage else None)
    msg = types.SimpleNamespace(content=text, reasoning_content=reasoning_text)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish)],
                                 usage=u, model="fake-served")


def scripted(core, script):
    """A runner's `once` whose n-th attempt for a question is script[n]: a response or an exception.
    Goes through core.openai_meta first, as the real runners do."""
    seen = {}

    def once(messages, model, params):
        q = qid_of(messages)
        k = seen[q] = seen.get(q, 0) + 1
        step = script[min(k, len(script)) - 1]
        if isinstance(step, BaseException):
            raise step
        meta = core.openai_meta(step)
        return step.choices[0].message.content, meta
    return once


def meta_for(qid, **extra):
    m = {"finish_reason": "stop", "prompt_tokens": int(qid), "completion_tokens": 5,
         "reasoning_tokens": None, "served_model": "fake-served"}
    m.update(extra)
    return m


def qid_of(messages):
    text = next(p["text"] for p in messages[-1]["content"] if p.get("type") == "text") \
        if isinstance(messages[-1]["content"], list) else ""
    return QID_BY_PROMPT.get(text)


QID_BY_PROMPT = {}


def test_b(base, new_src):
    print("\n== B  the sample path (fake clients)", flush=True)
    root = make_root(base, "b", new_src)
    core = load_core(root, "b_core")
    no_backoff(core)
    _test_b(core, root)


def no_backoff(core):
    """The core's own `time`, with sleep made instant: the retry back-off is not the code under test.
    Local to that module, so mock-server threads keep the real sleep; latencies are busy-waits."""
    core.time = types.SimpleNamespace(time=time.time, strftime=time.strftime, sleep=lambda *a, **k: None)


def _test_b(core, root):
    spec, rows = core.load_sample("sample10")
    data = core.load_data()
    for r in rows:
        QID_BY_PROMPT.setdefault(core.build_prompt(r["example"]), str(r["example"]["questionId"]))
    per = {}
    for r in rows:
        per[r["stratum"]] = per.get(r["stratum"], 0) + 1
    check(f"B1 sample10: {len(rows)} rows, {len(per)} strata, counts as in the file",
          len(rows) == 539 == spec["total_n"] and per == {k: v["n"] for k, v in spec["strata"].items()})
    check("B1 idx indexes the whole validation set, rows in data order",
          [r["idx"] for r in rows] == sorted({r["idx"] for r in rows})
          and all(str(data[r["idx"]]["questionId"]) == str(r["example"]["questionId"]) for r in rows))
    check("B1 every sample image resolves from the core's anchor",
          all(os.path.exists(core.image_file(r["example"]["image_path"])) for r in rows))
    check("B1 the stratum is question_types[0]",
          all(r["example"]["question_types"][0] == r["stratum"] for r in rows))

    def good(messages, model, params):
        q = qid_of(messages)
        return "Final Answer: x", meta_for(q)

    # smoke: --limit writes smoke_sample10/ only, every field on every row
    mdir = os.path.join(root, "M_smoke")
    os.makedirs(mdir)
    code, out = run_s(core, mdir, good, limit=3)
    smoke = os.path.join(mdir, "smoke_sample10")
    check("B2 --limit 3: exit 0, writes only smoke_sample10/",
          code == 0 and sorted(os.listdir(mdir)) == ["smoke_sample10"], f"exit {code}; {os.listdir(mdir)}\n{out[-500:]}")
    srows = [json.loads(line) for line in open(os.path.join(smoke, "test-fake-model-1_0.jsonl"))]
    need = set(core.USAGE_FIELDS) | set(core.BILLED_FIELDS) | set(core.RESUME_FIELDS) | {
        "parse_ok", "questionId", "stratum", "idx", "score", "anls", "seed", "config", "empty",
        "reasoning_chars", "think_tag_calls", "halted"}
    check("B2 every row carries every per-row field; `halted` is None on a row nothing interrupted",
          len(srows) == 3 and all(need <= set(r) for r in srows) and all(r["halted"] is None for r in srows),
          str([sorted(need - set(r)) for r in srows]))
    check("B2 rows are the first 3 sample questions, keyed by questionId",
          [r["questionId"] for r in srows] == [str(r["example"]["questionId"]) for r in rows[:3]])
    csv_text = open(os.path.join(smoke, "test-fake-model-1_0.csv")).read()
    check("B2 reasoning_tokens not reported -> None in the row, empty in the CSV (never 0)",
          all(r["reasoning_tokens"] is None for r in srows) and ",0,fake-served" not in csv_text)

    # metadata stays with its own row
    check("B3 each row's usage is its own call's (prompt_tokens == its questionId in the fake)",
          all(r["prompt_tokens"] == int(r["questionId"]) for r in srows))

    # retries: n_attempts counts every call, latency_s the successful one only
    state = {}

    def flaky(messages, model, params):
        q = qid_of(messages)
        n = state[q] = state.get(q, 0) + 1
        if n == 1:
            busy(0.5)
            raise RuntimeError("Error code: 502 - bad gateway")
        if n == 2:
            busy(0.3)
            return "", meta_for(q)
        busy(0.2)
        return "Final Answer: x", meta_for(q)
    mdir = os.path.join(root, "M_retry")
    os.makedirs(mdir)
    code, out = run_s(core, mdir, flaky, limit=2)
    rr = [json.loads(line) for line in open(os.path.join(mdir, "smoke_sample10", "test-fake-model-1_0.jsonl"))]
    check("B4 n_attempts = 3 after an exception and an empty reply",
          [r["n_attempts"] for r in rr] == [3, 3], str([r["n_attempts"] for r in rr]))
    check("B4 latency_s is the successful attempt alone (0.2 s, not 1.0 s)",
          all(0.15 <= r["latency_s"] < 0.45 for r in rr), str([r["latency_s"] for r in rr]))

    # a failed row is written as failed, not dropped
    def dead(messages, model, params):
        return "", meta_for(qid_of(messages), finish_reason="stop", completion_tokens=0)
    mdir = os.path.join(root, "M_dead")
    os.makedirs(mdir)
    code, out = run_s(core, mdir, dead, limit=2)
    dr = [json.loads(line) for line in open(os.path.join(mdir, "smoke_sample10", "test-fake-model-1_0.jsonl"))]
    check("B5 every attempt empty: rows written (2 of 2), empty=1, n_attempts=5, latency_s None, error set",
          len(dr) == 2 and all(r["empty"] == 1 and r["n_attempts"] == 5 and r["latency_s"] is None
                               and r["error"] and r["score"] == 0 and r["parse_ok"] is False for r in dr),
          json.dumps(dr[:1])[:400])
    check("B5 ... and the shard exits 1", code == 1, f"exit {code}")
    with open(os.path.join(mdir, "smoke_sample10", "test-fake-model-1_0_overall.csv")) as fh:
        allrow = [r for r in csv.DictReader(fh) if r["stratum"] == "ALL"]
    check("B5 the overall file counts them as empty (ALL: n=2, empty=2, parse_fail=2)",
          allrow and allrow[0]["n"] == "2" and allrow[0]["empty"] == "2" and allrow[0]["parse_fail"] == "2",
          str(allrow))

    # resume: only empty rows are called again; a different config is refused; foreign rows refused
    bad_q = {str(rows[3::20][1]["example"]["questionId"]), str(rows[3::20][4]["example"]["questionId"])}
    calls = []

    def half(messages, model, params):
        q = qid_of(messages)
        calls.append(q)
        return ("" if q in bad_q else "Final Answer: x"), meta_for(q)
    mdir = os.path.join(root, "M_resume")
    os.makedirs(mdir)
    code1, _ = run_s(core, mdir, half, shard=3, total_shards=20)
    path = os.path.join(mdir, "results_sample10", "test-fake-model-1_0_shard3of20.jsonl")
    first = [json.loads(line) for line in open(path)]
    n_first = len(calls)
    bad_q.clear()
    code2, out2 = run_s(core, mdir, half, shard=3, total_shards=20)
    second = [json.loads(line) for line in open(path)]
    recalled = calls[n_first:]
    check(f"B6 a real (non --limit) run writes results_sample10/ only ({len(first)} rows, shard 3 of 20)",
          sorted(os.listdir(mdir)) == ["results_sample10"] and len(first) == len(rows[3::20]))
    check(f"B6 resume calls exactly the 2 empty rows again (exit {code1} then {code2})",
          code1 == 1 and code2 == 0 and len(recalled) == 2 and len(second) == len(first), str(recalled))
    kept = {r["questionId"]: r for r in first if not r["empty"]}
    check("B6 rows done the first time are carried over unchanged",
          all(kept[r["questionId"]] == r for r in second if r["questionId"] in kept))
    code3, out3 = run_s(core, mdir, half, shard=3, total_shards=20, wanted={"max_tokens": 32, "seed": 42})
    check("B6 resume under a different config is refused", code3 and "refusing to resume" in str(code3),
          str(code3)[:200])
    shutil.copy(path, path.replace("shard3of20", "shard4of20"))
    code4, _ = run_s(core, mdir, half, shard=4, total_shards=20)
    check("B6 a file holding another shard's rows is refused, not overwritten",
          code4 and "not in this shard" in str(code4), str(code4)[:200])

    # halts: a runner's fatal marker, and DeepInfra's billing wording
    n = {"k": 0}

    def capped(messages, model, params):
        n["k"] += 1
        if n["k"] == 5:
            raise RuntimeError("Error code: 429 - Rate limit reached for requests per day (RPD)")
        return "Final Answer: x", meta_for(qid_of(messages))
    mdir = os.path.join(root, "M_fatal")
    os.makedirs(mdir)
    fatal = (("requests per day", "daily request cap reached"),)
    code, out = run_s(core, mdir, capped, shard=0, total_shards=20, fatal=fatal)
    fr = [json.loads(line) for line in open(os.path.join(mdir, "results_sample10",
                                                         "test-fake-model-1_0_shard0of20.jsonl"))]
    check("B7 a fatal marker stops at once, writes sample10_shard0_FATAL_HALT.txt, keeps the 4 rows done "
          "and writes the halting row as failed (5 rows)",
          "aborting" in str(code) and os.path.exists(os.path.join(mdir, "sample10_shard0_FATAL_HALT.txt"))
          and len(fr) == 5 and n["k"] == 5 and sum(r["empty"] for r in fr) == 1
          and fr[-1]["empty"] == 1 and str(fr[-1]["error"]).startswith("interrupted: SystemExit")
          and str(fr[-1]["halted"]).startswith("SystemExit: aborting") and fr[0]["halted"] is None,
          f"{code}; calls {n['k']}; rows {len(fr)}")
    check("B7 ... the halting call is counted once as a call without usage (not by both handlers), "
          "n_attempts 1", fr[-1]["attempts_without_usage"] == 1 and fr[-1]["n_attempts"] == 1,
          json.dumps({k: fr[-1].get(k) for k in ("attempts_without_usage", "n_attempts", "error")}))

    def broke(messages, model, params):
        raise RuntimeError("Error code: 402 - {'error': {'message': 'You need positive balance to do "
                           "inference. Please add balance manually or setup top-up'}}")
    mdir = os.path.join(root, "M_bill")
    os.makedirs(mdir)
    code, out = run_s(core, mdir, broke, limit=2)
    check("B7 DeepInfra's 402 stops as BILLING on the first call; a smoke's marker is smoke_sample10_*",
          "aborting" in str(code) and os.path.exists(os.path.join(mdir, "smoke_sample10_BILLING_HALT.txt"))
          and not os.path.exists(os.path.join(mdir, "sample10_BILLING_HALT.txt")), str(code))
    real_marker = os.path.join(mdir, "sample10_BILLING_HALT.txt")
    with open(real_marker, "w") as fh:
        fh.write("the real run's marker\n")
    code, out = run_s(core, mdir, good, limit=1)
    check("B7 a later smoke clears its own stale marker and leaves the real run's alone",
          code == 0 and os.path.exists(real_marker)
          and not os.path.exists(os.path.join(mdir, "smoke_sample10_BILLING_HALT.txt")), out[-300:])

    # CutRetry: first call cut, second cut, forced answer
    cut = core.CutRetry("CEILING", "RETRY", "FORCE", 99)
    sent_log = []

    def stage_call(sent, stage):
        sent_log.append((stage, sent))
        if stage == "first":
            return "partial one", "partial one", dict(meta_for("1"), finish_reason="length",
                                                      completion_tokens=100, reasoning_chars=0)
        if stage == "retry":
            return "partial two", "partial two", dict(meta_for("1"), finish_reason="length",
                                                      completion_tokens=50, reasoning_chars=0)
        return "Final Answer: x", "Final Answer: x", dict(meta_for("1"), completion_tokens=7, reasoning_chars=0)

    def cut_once(messages, model, params):
        return cut(messages, stage_call, model)
    mdir = os.path.join(root, "M_cut")
    os.makedirs(mdir)
    code, out = run_s(core, mdir, cut_once, limit=1, extra_config=cut.config())
    cr = json.loads(open(os.path.join(mdir, "smoke_sample10", "test-fake-model-1_0.jsonl")).readline())
    first_sent = sent_log[0][1]
    check("B8 CutRetry: ceiling is the first (system) message, the shared user message follows unchanged",
          first_sent[0] == {"role": "system", "content": "CEILING"} and first_sent[1]["role"] == "user"
          and len(first_sent) == 2)
    check("B8 CutRetry: cut twice -> forced answer; row marked and usage summed (100+50+7)",
          cr.get("cut_retry") and cr.get("forced_answer") and cr["completion_tokens"] == 157
          and cr["first_response"] == "partial one" and cr["retry_response"] == "partial two"
          and cr["model_response"] == "Final Answer: x" and [s for s, _ in sent_log] == ["first", "retry", "forced"]
          and sent_log[2][1][-2:] == [{"role": "assistant", "content": "partial two"},
                                      {"role": "user", "content": "FORCE"}]
          and cr["config"]["prompt_ceiling"] == "CEILING", json.dumps(cr)[:500])

    # billed usage: every attempt and every CutRetry stage that returned usage, failed ones included
    def one_row(once, **kw):
        mdir = tempfile.mkdtemp(dir=root, prefix="M_bill_")
        code, out = run_s(core, mdir, once, limit=1, **kw)
        path = os.path.join(mdir, "smoke_sample10", "test-fake-model-1_0.jsonl")
        one_row.last_dir = mdir
        line = open(path).readline() if os.path.exists(path) else ""
        return code, out, (json.loads(line) if line.strip() else {})     # a halted run writes no row

    code, out, r = one_row(scripted(core, [
        fake_resp("", completion=16384, reasoning=16384, finish="length"),     # HTTP 200, empty, billed
        fake_resp("Final Answer: x", completion=50, reasoning=0)]))
    check("B9 empty reply with 16,384 reasoning tokens, then an answer of 50: billed_completion_tokens "
          "16434, completion_tokens 50",
          code == 0 and r["billed_completion_tokens"] == 16434 and r["completion_tokens"] == 50
          and r["billed_reasoning_tokens"] == 16384 and r["reasoning_tokens"] == 0
          and r["billed_prompt_tokens"] == 2000 and r["prompt_tokens"] == 1000
          and r["attempts_without_usage"] == 0 and r["n_attempts"] == 2,
          json.dumps({k: r.get(k) for k in (*core.USAGE_FIELDS, *core.BILLED_FIELDS)}))
    code500, out500, r = one_row(scripted(core, [
        RuntimeError("Error code: 500 - {'error': 'internal', 'request_id': 'req_9f401ab2'}"),
        fake_resp("", usage=False),                                          # answered, no usage object
        fake_resp("Final Answer: x", completion=50)]))
    check("B9 an exception and a reply without usage are counted (attempts_without_usage 2), not billed; "
          "no reasoning reported anywhere -> billed_reasoning_tokens None",
          code500 == 0 and r["attempts_without_usage"] == 2 and r["billed_completion_tokens"] == 50
          and r["billed_reasoning_tokens"] is None and r["n_attempts"] == 3,
          json.dumps({k: r.get(k) for k in (*core.USAGE_FIELDS, *core.BILLED_FIELDS)}) + out500[-300:])
    stages = {"first": [fake_resp("partial", completion=8192, finish="length")],
              "retry": [RuntimeError("Error code: 502 - bad gateway"),
                        fake_resp("", completion=40), fake_resp("Final Answer: x", completion=60)]}
    cut2 = core.CutRetry("CEILING", "RETRY", "FORCE", 99)

    def staged(sent, stage):
        step = stages[stage].pop(0)
        if isinstance(step, BaseException):
            raise step
        meta = core.openai_meta(step)
        text = step.choices[0].message.content
        return text, text, meta
    code, out, r = one_row(lambda m, model, params: cut2(m, staged, model), extra_config=cut2.config())
    check("B9 CutRetry: a cut first call (8192), a retry that raised, one that was empty (40), one that "
          "answered (60) -> billed 8292, completion_tokens 8252 (first + answering retry), 1 without usage",
          code == 0 and r["billed_completion_tokens"] == 8292 and r["completion_tokens"] == 8252
          and r["attempts_without_usage"] == 1 and r["n_attempts"] == 3 and r.get("cut_retry")
          and r.get("first_completion_tokens") == 8192,
          json.dumps({k: r.get(k) for k in (*core.USAGE_FIELDS, *core.BILLED_FIELDS, "cut_retry")}))
    with open(os.path.join(one_row.last_dir, "smoke_sample10", "test-fake-model-1_0_overall.csv")) as fh:
        header = fh.readline().strip().split(",")
    check("B9 the overall file carries the billed totals and attempts_without_usage",
          all(k in header for k in core.BILLED_FIELDS) and "reasoning_chars_rows" in header, str(header))

    # numeric statuses are whole words: a hex request id is not a 401
    hexed = RuntimeError("Error code: 500 - {'error': 'internal', 'request_id': 'req_9f401ab2'}")
    check("B10 a 500 whose request id contains '401' is not an auth failure (it was, as a substring)",
          "401" in str(hexed) and not core.is_fatal_request(hexed) and not core.is_auth_failure(str(hexed)))
    check("B10 a real 401 still is",
          core.is_fatal_request(RuntimeError("Error code: 401 - Unauthorized"))
          and core.has_status("Error code: 502 - bad gateway", core.TRANSIENT_STATUS)
          and not core.has_status("x-request-id 5029ac", core.TRANSIENT_STATUS))
    check("B10 ... and in a run, that 500 was retried, not halted (answered on attempt 3, exit 0)",
          code500 == 0 and "MALFORMED REQUEST" not in out500 and "API error (1/5)" in out500, out500[-300:])

    # Qwen, Gemma: the first row showing thinking stops the run (marker, save, exit 3)
    calls = []

    def thinks_from(n_ok):
        def once(messages, model, params):
            calls.append(qid_of(messages))
            r = fake_resp("Final Answer: x", completion=30,
                          reasoning_text="Let me look..." if len(calls) > n_ok else None)
            meta = core.openai_meta(r)
            return r.choices[0].message.content, meta
        return once
    mdir = os.path.join(root, "M_think")
    os.makedirs(mdir)
    code, out = run_s(core, mdir, thinks_from(0), limit=3, forbid_reasoning=True)
    srows = [json.loads(line) for line in open(os.path.join(mdir, "smoke_sample10", "test-fake-model-1_0.jsonl"))]
    check("B11 smoke: the first row with reasoning text halts at once — 1 call of 3, the row saved, "
          "smoke_sample10_THINKING_HALT.txt, exit 3",
          code == 3 and len(calls) == 1 and len(srows) == 1 and srows[0]["reasoning_chars"] == 14
          and os.path.exists(os.path.join(mdir, "smoke_sample10_THINKING_HALT.txt"))
          and not os.path.exists(os.path.join(mdir, "sample10_THINKING_HALT.txt"))
          and "THINKING CAME BACK ON" in out, f"exit {code}; calls {len(calls)}\n{out[-400:]}")
    calls.clear()
    mdir = os.path.join(root, "M_think_run")
    os.makedirs(mdir)
    code, out = run_s(core, mdir, thinks_from(2), shard=3, total_shards=20, forbid_reasoning=True)
    rr = [json.loads(line) for line in open(os.path.join(mdir, "results_sample10",
                                                         "test-fake-model-1_0_shard3of20.jsonl"))]
    check("B11 real run: thinking on the 3rd row stops it there — 3 calls of 27, 3 rows saved, "
          "sample10_shard3_THINKING_HALT.txt, exit 3",
          code == 3 and len(calls) == 3 and len(rr) == 3
          and os.path.exists(os.path.join(mdir, "sample10_shard3_THINKING_HALT.txt")),
          f"exit {code}; calls {len(calls)}; rows {len(rr)}")
    calls.clear()
    stored = open(os.path.join(mdir, "results_sample10", "test-fake-model-1_0_shard3of20.jsonl")).read()
    os.remove(os.path.join(mdir, "sample10_shard3_THINKING_HALT.txt"))
    code, out = run_s(core, mdir, thinks_from(99), shard=3, total_shards=20, forbid_reasoning=True)
    check("B11 ... a resubmit on the same config halts before its first call: 0 calls, marker rewritten, "
          "file unchanged, exit 3",
          code == 3 and len(calls) == 0
          and os.path.exists(os.path.join(mdir, "sample10_shard3_THINKING_HALT.txt"))
          and open(os.path.join(mdir, "results_sample10", "test-fake-model-1_0_shard3of20.jsonl")).read() == stored,
          f"exit {code}; calls {len(calls)}\n{out[-300:]}")
    calls.clear()
    mdir = os.path.join(root, "M_think_ok")
    os.makedirs(mdir)
    code, out = run_s(core, mdir, thinks_from(0), limit=2)
    check("B11 without forbid_reasoning the same rows run through and exit 0 (only Qwen and Gemma set it)",
          code == 0 and len(calls) == 2, f"exit {code}")

    # --save-every below 1 is refused at parsing
    ap = core.sample_parser("t", "test/fake-model-1.0")
    with contextlib.redirect_stderr(io.StringIO()) as err:
        try:
            ap.parse_args(["--sample", "sample10", "--save-every", "0"])
            refused = None
        except SystemExit as e:
            refused = e.code
    check("B12 --save-every 0 is refused by the sample parser (exit 2)",
          refused == 2 and "must be 1 or more" in err.getvalue(), err.getvalue()[-200:])

    _test_resume_history(core, root, rows)
    _test_ledger_more(core, root)
    _test_parse_ok(core, rows)


def _test_parse_ok(core, rows):
    """parse_ok False must mean score 0 AND ANLS 0. Score: score_response returns 0 for an empty
    extraction. ANLS: anls_score gives an empty prediction 1.0 against an empty gold answer, so
    load_sample refuses a question that has one."""
    replies = ("", "   ", "Final Answer:", "Final Answer:   \n", "Final Answer: wrong", "no marker at all")
    violations, n = [], 0
    for row in rows:
        gold = row["example"]["answers"]
        for reply in replies + (f"Final Answer: {gold[0]}", gold[0], f"It says {gold[0]}"):
            text = reply.strip()
            rec = core.sample_row(row, text, {}, {}, "sample10", {})
            n += 1
            if not rec["parse_ok"] and (rec["score"] != 0 or rec["anls"] != 0):
                violations.append((rec["questionId"], reply))
    check(f"B18 parse_ok False => score 0 and ANLS 0, over {n:,} replies to the 539 sample questions",
          not violations, str(violations[:5]))
    real = core.load_data
    try:
        first = str(rows[0]["example"]["questionId"])
        core.load_data = lambda: [dict(ex, answers=[""]) if str(ex["questionId"]) == first else ex
                                  for ex in real()]
        try:
            core.load_sample("sample10")
            refused = ""
        except SystemExit as e:
            refused = str(e)
    finally:
        core.load_data = real
    check("B18 a sample question with an empty gold answer is refused at load (ANLS would credit an "
          "empty reply)", "gold answer is empty" in refused, refused[-200:])


def _test_resume_history(core, root, rows):
    """The reviewer's resume_probe.py (MMLU, 2026-11-15), adapted: a row that fails 5 times at 16,384
    billed tokens a call, then answers on the resubmit."""
    shard_rows = rows[3::20]
    doomed = str(shard_rows[1]["example"]["questionId"])
    other = str(shard_rows[4]["example"]["questionId"])
    mode = {"fail": {doomed}, "kill": set()}

    def once(messages, model, params):
        q = qid_of(messages)
        if q in mode["kill"]:
            raise SystemExit(143)                      # SIGTERM mid-row, as run_sample turns it
        r = (fake_resp("", completion=16384, finish="length") if q in mode["fail"]   # billed, empty
             else fake_resp("Final Answer: x", completion=50))
        meta = core.openai_meta(r)
        return r.choices[0].message.content, meta

    def rows_of(path):
        return {json.loads(line)["questionId"]: json.loads(line) for line in open(path)}

    mdir = os.path.join(root, "M_hist")
    os.makedirs(mdir)
    path = os.path.join(mdir, "results_sample10", "test-fake-model-1_0_shard3of20.jsonl")
    code1, _ = run_s(core, mdir, once, shard=3, total_shards=20)
    r1 = rows_of(path)
    d1 = r1[doomed]
    check(f"B13 run 1: the doomed row fails 5 times, billed 5 x 16,384 = 81,920 (exit {code1})",
          code1 == 1 and d1["empty"] == 1 and d1["n_attempts"] == 5 and d1["billed_completion_tokens"] == 81920
          and d1["prior_failed_runs"] == 0 and d1["resumed_after_failure"] is False,
          json.dumps({k: d1.get(k) for k in ("empty", "n_attempts", "billed_completion_tokens")}))
    code2, _ = run_s(core, mdir, once, shard=3, total_shards=20)          # resubmit, still failing
    d2 = rows_of(path)[doomed]
    check("B13 run 2 fails again: billed 163,840 over both runs, n_attempts 5 (this run), "
          "prior_n_attempts 5, prior_failed_runs 1",
          code2 == 1 and d2["empty"] == 1 and d2["billed_completion_tokens"] == 163840
          and d2["n_attempts"] == 5 and d2["prior_n_attempts"] == 5 and d2["prior_failed_runs"] == 1
          and d2["resumed_after_failure"] is True,
          json.dumps({k: d2.get(k) for k in ("billed_completion_tokens", "n_attempts", *core.RESUME_FIELDS)}))
    mode["fail"] = set()
    code3, _ = run_s(core, mdir, once, shard=3, total_shards=20)          # resubmit, now it answers
    r3 = rows_of(path)
    d3 = r3[doomed]
    check("B13 run 3 answers: n_attempts 1, completion_tokens 50, billed 163,890 (every run), "
          "prior_n_attempts 10, prior_failed_runs 2, resumed_after_failure",
          code3 == 0 and d3["empty"] == 0 and d3["n_attempts"] == 1 and d3["completion_tokens"] == 50
          and d3["billed_completion_tokens"] == 163890 and d3["billed_prompt_tokens"] == 11000
          and d3["prior_n_attempts"] == 10 and d3["prior_failed_runs"] == 2
          and d3["resumed_after_failure"] is True and d3["attempts_without_usage"] == 0,
          json.dumps({k: d3.get(k) for k in ("n_attempts", "completion_tokens", *core.BILLED_FIELDS,
                                             *core.RESUME_FIELDS)}))
    billed_rows = sum(r["billed_completion_tokens"] for r in r3.values())
    check(f"B13 the file's billed total is what the three runs actually cost ({billed_rows:,})",
          billed_rows == 163840 + 50 + 50 * (len(r3) - 1) and len(r3) == len(shard_rows))
    check("B13 a row that never failed carries prior 0 / 0 / false",
          all(r["prior_n_attempts"] == 0 and r["prior_failed_runs"] == 0 and r["resumed_after_failure"] is False
              for q, r in r3.items() if q != doomed))

    # killed in the middle of a resubmit: a stored failed row must still be in the file
    mdir = os.path.join(root, "M_kill")
    os.makedirs(mdir)
    path = os.path.join(mdir, "results_sample10", "test-fake-model-1_0_shard3of20.jsonl")
    mode.update(fail={doomed, other}, kill=set())
    run_s(core, mdir, once, shard=3, total_shards=20)
    before = rows_of(path)
    mode.update(fail=set(), kill={other})            # the resubmit answers `doomed`, dies on `other`
    code, out = run_s(core, mdir, once, shard=3, total_shards=20)
    after = rows_of(path)
    o = after.get(other, {})
    check(f"B14 resubmit killed mid-row (exit {code}): every row still in the file ({len(after)} of "
          f"{len(shard_rows)}); the row it died on is replaced by a failed row with its history folded in",
          code == 143 and len(after) == len(shard_rows) and o.get("empty") == 1
          and o.get("billed_completion_tokens") == before[other]["billed_completion_tokens"] == 81920
          and o.get("prior_n_attempts") == 5 and o.get("prior_failed_runs") == 1 and o.get("n_attempts") == 1
          and o.get("attempts_without_usage") == 1 and o.get("resumed_after_failure") is True
          and str(o.get("error")).startswith("interrupted: SystemExit") and o.get("halted") == "SystemExit: 143",
          f"exit {code}; rows {len(after)}; " + json.dumps({k: o.get(k) for k in (
              "empty", "billed_completion_tokens", "n_attempts", "attempts_without_usage", "error",
              *core.RESUME_FIELDS)}))
    check("B14 ... and the row it finished first was replaced, with its history folded in",
          after[doomed]["empty"] == 0 and after[doomed]["billed_completion_tokens"] == 81920 + 50
          and after[doomed]["prior_n_attempts"] == 5 and after[doomed]["resumed_after_failure"] is True)
    _test_interrupted_rows(core, root, shard_rows, rows_of)


def _test_interrupted_rows(core, root, shard_rows, rows_of):
    """The reviewer's probe_ledger.py P1-P3 (2026-11-15): calls billed in the row in flight when a
    resubmit is killed or the run halts. Each asserts the file's billed total equals what was billed."""
    doomed = str(shard_rows[1]["example"]["questionId"])
    billed = {"tokens": 0}

    def make_once(script):
        """script(q, k) -> response or exception, k = this question's call number in this run."""
        seen = {}

        def once(messages, model, params):
            q = qid_of(messages)
            k = seen[q] = seen.get(q, 0) + 1
            step = script(q, k)
            if isinstance(step, BaseException):
                raise step
            billed["tokens"] += step.usage.completion_tokens if step.usage else 0
            meta = core.openai_meta(step)
            return step.choices[0].message.content, meta
        return once

    def file_billed(path):
        return sum(r["billed_completion_tokens"] or 0 for r in rows_of(path).values())

    # P1: the resubmit is killed in its 3rd attempt on the stored failed row, after 2 billed attempts
    mdir = os.path.join(root, "M_kill_billed")
    os.makedirs(mdir)
    path = os.path.join(mdir, "results_sample10", "test-fake-model-1_0_shard3of20.jsonl")
    billed["tokens"] = 0
    fail = lambda q, k: fake_resp("", completion=16384, finish="length") if q == doomed \
        else fake_resp("Final Answer: x", completion=50)
    run_s(core, mdir, make_once(fail), shard=3, total_shards=20)
    kill3 = lambda q, k: (SystemExit(143) if k >= 3 else fake_resp("", completion=16384, finish="length"))
    code, _ = run_s(core, mdir, make_once(kill3), shard=3, total_shards=20)
    d = rows_of(path)[doomed]
    check(f"B14 killed in attempt 3 after 2 billed attempts (exit {code}): billed 5x + 2x 16,384 = 114,688, "
          f"n_attempts 3 (the 3rd was sent), 1 call without usage, prior 5 / 1",
          code == 143 and d["billed_completion_tokens"] == 114688 and d["n_attempts"] == 3
          and d["attempts_without_usage"] == 1 and d["prior_n_attempts"] == 5 and d["prior_failed_runs"] == 1
          and file_billed(path) == billed["tokens"],
          json.dumps({k: d.get(k) for k in ("billed_completion_tokens", "n_attempts", "attempts_without_usage",
                                             *core.RESUME_FIELDS)}) + f" file {file_billed(path)} vs {billed['tokens']}")
    code, _ = run_s(core, mdir, make_once(lambda q, k: fake_resp("Final Answer: x", completion=50)),
                    shard=3, total_shards=20)
    d = rows_of(path)[doomed]
    check(f"B14 ... it answers on the next resubmit: billed 114,738 = what was billed (file total "
          f"{file_billed(path):,} = {billed['tokens']:,}), prior_n_attempts 8, prior_failed_runs 2",
          code == 0 and d["billed_completion_tokens"] == 114738 and d["prior_n_attempts"] == 8
          and d["prior_failed_runs"] == 2 and file_billed(path) == billed["tokens"])

    # P3: SIGTERM inside sample_row, after the call returned — the stored row must not be deleted
    for label, times in (("once", 1), ("every time", 10 ** 6)):
        mdir = tempfile.mkdtemp(dir=root, prefix="M_kill_row_")
        path = os.path.join(mdir, "results_sample10", "test-fake-model-1_0_shard3of20.jsonl")
        billed["tokens"] = 0
        run_s(core, mdir, make_once(fail), shard=3, total_shards=20)
        stored = rows_of(path)[doomed]
        real_score, left = core.score_response, {"n": times}

        def score_then_sigterm(text, gold):
            if text and left["n"] > 0:
                left["n"] -= 1
                raise SystemExit(143)          # what the SIGTERM handler raises, inside sample_row
            return real_score(text, gold)
        core.score_response = score_then_sigterm
        try:
            code, out = run_s(core, mdir, make_once(lambda q, k: fake_resp("Final Answer: x", completion=50)),
                              shard=3, total_shards=20)
        finally:
            core.score_response = real_score
        after = rows_of(path)
        if times == 1:
            ok = (after[doomed]["empty"] == 0 and after[doomed]["billed_completion_tokens"] == 81920 + 50
                  and after[doomed]["resumed_after_failure"] is True)
            what = "the answered call is written, its history folded in"
        else:
            ok = after.get(doomed) == stored
            what = "writing it fails too, so the stored row is kept as it was"
        check(f"B14 SIGTERM inside sample_row ({label}, exit {code}): all {len(after)} rows kept; {what}",
              code == 143 and len(after) == len(shard_rows) and ok, out[-300:])

    # P2: a FAILURE halt (10 consecutive failed rows) — the 10th row and its billed calls are kept
    core._recent.clear()
    core._consecutive[0] = 0
    mdir = os.path.join(root, "M_failure_halt")
    os.makedirs(mdir)
    path = os.path.join(mdir, "results_sample10", "test-fake-model-1_0_shard3of20.jsonl")
    billed["tokens"] = 0
    code, out = run_s(core, mdir, make_once(lambda q, k: fake_resp("", completion=1000, finish="length")),
                      shard=3, total_shards=20)
    f = rows_of(path)
    check(f"B14 FAILURE halt after 10 failed rows: 10 rows in the file, the halting row included; file "
          f"billed {file_billed(path):,} = actually billed {billed['tokens']:,}",
          "stopped on failure" in str(code) and len(f) == 10 and all(r["empty"] for r in f.values())
          and file_billed(path) == billed["tokens"] == 50000
          and os.path.exists(os.path.join(mdir, "sample10_shard3_FAILURE_HALT.txt")),
          f"{str(code)[:80]}; rows {len(f)}")
    core._recent.clear()
    core._consecutive[0] = 0

    # Killed at `held.pop(qid, None)`: the new row is already in `records` and the stored one still in
    # `held`. Two guards keep the row written once — to_save() skips a held row already replaced, and
    # the in-flight recovery skips a row already in records — and only RAW LINES can show a duplicate:
    # a qid-keyed dict keeps one of them. The reviewer's probe_w3_lines.py (2026-11-15), folded in.
    mdir = os.path.join(root, "M_kill_at_pop")
    os.makedirs(mdir)
    path = os.path.join(mdir, "results_sample10", "test-fake-model-1_0_shard3of20.jsonl")
    run_s(core, mdir, make_once(fail), shard=3, total_shards=20)
    src, start = inspect.getsourcelines(core.run_sample)
    pop_line = [start + i for i, text in enumerate(src) if "held.pop(qid, None)" in text][0]
    fired = []

    def tracer(frame, event, arg):
        if frame.f_code is not core.run_sample.__code__:
            return None

        def local(fr, ev, a):
            if ev == "line" and fr.f_lineno == pop_line and not fired:
                fired.append(fr.f_lineno)
                raise SystemExit(143)           # the SIGTERM handler's exit, at that very line
            return local
        return local
    sys.settrace(tracer)
    try:
        code, out = run_s(core, mdir, make_once(lambda q, k: fake_resp("Final Answer: x", completion=50)),
                          shard=3, total_shards=20)
    finally:
        sys.settrace(None)
    lines = [json.loads(line) for line in open(path)]
    counts = collections.Counter(r["questionId"] for r in lines)
    dups = {q: n for q, n in counts.items() if n > 1}
    mine = [r for r in lines if r["questionId"] == doomed]
    check(f"B14 killed AT held.pop (exit {code}): {len(lines)} raw lines for {len(counts)} questions, no "
          f"duplicate; the doomed row once, answered, billed 81,970 summed over its lines",
          code == 143 and fired and len(lines) == len(counts) == len(shard_rows) and not dups
          and len(mine) == 1 and mine[0]["empty"] == 0
          and sum(r["billed_completion_tokens"] or 0 for r in mine) == 81920 + 50,
          f"fired {fired}; duplicates {dups}; doomed lines {len(mine)}")


def _test_ledger_more(core, root):
    def one_row(once, **kw):
        mdir = tempfile.mkdtemp(dir=root, prefix="M_led_")
        code, out = run_s(core, mdir, once, limit=1, **kw)
        path = os.path.join(mdir, "smoke_sample10", "test-fake-model-1_0.jsonl")
        line = open(path).readline() if os.path.exists(path) else ""
        return code, out, (json.loads(line) if line.strip() else {})

    def priced(text, cost=None, estimated=None, **kw):
        r = fake_resp(text, **kw)
        if cost is not None:
            r.usage.cost = cost
        if estimated is not None:
            r.usage.estimated_cost = estimated
        return r

    # costs: summed over every call; the per-call field is the last call's
    code, out, r = one_row(scripted(core, [priced("", cost=0.0012, completion=4000),
                                           priced("Final Answer: x", cost=0.0003, completion=50)]))
    check("B15 OpenRouter cost: billed_cost 0.0015 over both calls, last_call_cost 0.0003, "
          "billed_estimated_cost None (never reported)",
          code == 0 and r["billed_cost"] == 0.0015 and r["last_call_cost"] == 0.0003
          and r["billed_estimated_cost"] is None and "cost" not in r,
          json.dumps({k: r.get(k) for k in ("billed_cost", "last_call_cost", "billed_estimated_cost")}))
    code, out, r = one_row(scripted(core, [priced("", estimated=0.00002, completion=10),
                                           priced("Final Answer: x", estimated=0.00001, completion=10)]))
    check("B15 DeepInfra estimated_cost: billed 0.00003, last_call_estimated_cost 0.00001, billed_cost None",
          code == 0 and r["billed_estimated_cost"] == 0.00003 and r["last_call_estimated_cost"] == 0.00001
          and r["billed_cost"] is None,
          json.dumps({k: r.get(k) for k in ("billed_estimated_cost", "last_call_estimated_cost", "billed_cost")}))

    # Qwen: reasoning text in a CutRetry stage that is not the answering one still counts
    cut = core.CutRetry("CEILING", "RETRY", "FORCE", 99)
    stages = {"first": [fake_resp("partial", completion=8192, finish="length", reasoning_text="thinking..")],
              "retry": [fake_resp("Final Answer: x", completion=60, reasoning_text="")]}

    def staged(sent, stage):
        step = stages[stage].pop(0)
        meta = core.openai_meta(step)
        text = step.choices[0].message.content
        return text, text, meta
    code, out, r = one_row(lambda m, model, params: cut(m, staged, model), extra_config=cut.config(),
                           forbid_reasoning=True)
    check("B16 Qwen: reasoning text only in the cut first stage -> reasoning_chars 10 on the row, exit 3",
          code == 3 and r.get("reasoning_chars") == 10 and r.get("cut_retry")
          and r.get("reasoning_tokens") is None and "THINKING CAME BACK ON" in out,
          f"exit {code}; reasoning_chars {r.get('reasoning_chars')}\n{out[-300:]}")
    code, out, r = one_row(scripted(core, [fake_resp("Final Answer: x", completion=30, reasoning=12)]),
                           forbid_reasoning=True)
    check("B16 Gemma: reasoning_tokens 12 with no reasoning text -> exit 3",
          code == 3 and r.get("reasoning_chars") == 0 and r.get("reasoning_tokens") == 12
          and "THINKING CAME BACK ON" in out, f"exit {code}\n{out[-300:]}")
    code, out, r = one_row(scripted(core, [fake_resp("Final Answer: x", completion=30, reasoning=0)]),
                           forbid_reasoning=True)
    check("B16 reasoning_tokens 0 and no text -> exit 0", code == 0, f"exit {code}")

    # P4: `once` raises after openai_meta booked the call (HTTP 200, usage, `choices` empty)
    n4 = {"k": 0}

    def once4(messages, model, params):
        n4["k"] += 1
        r = fake_resp("", completion=7) if n4["k"] == 1 else fake_resp("Final Answer: x", completion=50)
        if n4["k"] == 1:
            r.choices = []
        meta = core.openai_meta(r)
        return r.choices[0].message.content, meta          # IndexError on the first, as every runner's
    code, out, r = one_row(once4)
    check("B17 a call that returned usage and then raised (empty choices) is billed and NOT counted as "
          "without usage: billed 57, attempts_without_usage 0",
          code == 0 and r["billed_completion_tokens"] == 57 and r["attempts_without_usage"] == 0
          and r["n_attempts"] == 2, json.dumps({k: r.get(k) for k in ("billed_completion_tokens",
                                                                       "attempts_without_usage", "n_attempts")}))

    # inline thinking: a think tag in `content`, in any of NEG_Gemma_DeepInfra's four spellings
    tags = ["<think>", "</think>", "<|think|>", "<|/think|>", "< / THINK >", "<Think>"]
    plain = ["thinking about it", "<thinking>", "I think so", "think>", "<thin k>"]
    check("B16 THINK_TAG matches the four spellings (any case, inner spaces) and not plain text",
          all(core.THINK_TAG.search(t) for t in tags) and not any(core.THINK_TAG.search(t) for t in plain),
          str([(t, bool(core.THINK_TAG.search(t))) for t in tags + plain]))
    for content in ("<think>Let me look at the table.</think>Final Answer: x",
                    "the totals row reads 4,000</think>\nFinal Answer: x",
                    "<|think|>scan the header<|/think|>Final Answer: x"):
        code, out, r = one_row(scripted(core, [fake_resp(content, completion=30, reasoning=0)]),
                               forbid_reasoning=True)
        check(f"B16 inline thinking {content[:22]!r}..., reasoning_tokens 0, no reasoning field -> exit 3",
              code == 3 and r.get("think_tag_calls") == 1 and "THINKING CAME BACK ON" in out,
              f"exit {code}\n{out[-300:]}")
    code, out, r = one_row(scripted(core, [fake_resp("<think>x</think>Final Answer: x", completion=30)]))
    check("B16 ... without forbid_reasoning (GPT, Gemini) it is recorded, not halted",
          code == 0 and r.get("think_tag_calls") == 1, f"exit {code}")

    # attempts_without_usage counts HTTP calls, CutRetry stages included
    cut = core.CutRetry("CEILING", "RETRY", "FORCE", 99)
    stages = {"first": [fake_resp("partial", finish="length", usage=False)],
              "retry": [fake_resp("partial two", completion=70, finish="length")],
              "forced": [fake_resp("Final Answer: x", usage=False)]}
    code, out, r = one_row(lambda m, model, params: cut(m, staged, model), extra_config=cut.config())
    check("B17 one attempt, three calls, two of them without usage -> attempts_without_usage 2, "
          "n_attempts 1 (calls = n_attempts + cut_retry + forced_answer = 3)",
          code == 0 and r["attempts_without_usage"] == 2 and r["n_attempts"] == 1
          and r.get("cut_retry") and r.get("forced_answer") and r["billed_completion_tokens"] == 70,
          json.dumps({k: r.get(k) for k in ("attempts_without_usage", "n_attempts", "billed_completion_tokens",
                                             "cut_retry", "forced_answer")}))


# ---------------------------------------------------------------- B4: the watchdog, on a blocked socket


def _read_request(conn):
    """One whole request off `conn`; False when the client closed it first."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        data = conn.recv(65536)
        if not data:
            return False
        buf += data
    head, body = buf.split(b"\r\n\r\n", 1)
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":")[1])
    while len(body) < length:
        data = conn.recv(65536)
        if not data:
            return False
        body += data
    return True


class _Server(socket.socket):
    """socket.socket has __slots__; this can carry the request counter."""


def mock_server(mode):
    """A local HTTP server that takes a request and never finishes answering it.

    "cut-then-keepalive": the first request is answered at once, finish_reason "length" with usage
    100 / 8,192 (a reply cut at the cap); every later one is "keepalive" — the reviewer's
    `wd_cutretry_probe.py` (MMLU, 2026-11-15).
    "silent": reads the request, sends nothing. "keepalive": sends the headers of a chunked 200, then a
    one-byte chunk every 0.5 s for ever — each chunk resets the client's read timeout, so the SDK's
    own `timeout=400` never fires and only the SIGALRM watchdog can end the call. The keep-alive case
    is the reviewer's `watchdog_probe.py` (2026-11-15), folded in here so the proof travels with the
    code; its TLS variant needs a certificate and is not repeated."""
    srv = _Server()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    served = srv.served = [0]                      # requests read in full, for the test to count

    def handle(conn):
        try:
            # Every request on the connection, not just the first: the SDK reuses a keep-alive
            # connection, so the retry after a cut reply arrives on the same socket.
            while _read_request(conn):
                served[0] += 1
                if not (mode == "cut-then-keepalive" and served[0] == 1):
                    break
                body = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                   "choices": [{"index": 0, "finish_reason": "length",
                                                "message": {"role": "assistant", "content": "partial ..."}}],
                                   "usage": {"prompt_tokens": 100, "completion_tokens": 8192,
                                             "total_tokens": 8292}}).encode()
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d"
                             b"\r\n\r\n" % len(body) + body)
            else:
                return                                 # closed before a full request arrived
            if mode == "silent":
                time.sleep(3600)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         b"Transfer-Encoding: chunked\r\n\r\n")
            while True:
                conn.sendall(b"1\r\n\n\r\n")
                time.sleep(0.5)
        except OSError:
            pass

    def accept():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    threading.Thread(target=accept, daemon=True).start()
    return srv


def test_watchdog(base, new_src):
    print("\n== B4  the SIGALRM watchdog, on an SDK call blocked in socket I/O (local mock server)", flush=True)
    try:
        import openai
    except ImportError as error:
        SKIPS.append(f"B4 watchdog SKIPPED: the real openai SDK is not importable ({error})")
        print(SKIPS[-1], flush=True)
        return
    if not hasattr(signal, "SIGALRM"):
        SKIPS.append("B4 watchdog SKIPPED: no SIGALRM on this platform")
        print(SKIPS[-1], flush=True)
        return
    root = make_root(base, "wd", new_src)
    core = load_core(root, "wd_core")
    no_backoff(core)
    # if the alarm never fires, say so and stop rather than hang the test for ever
    guard = threading.Timer(150, lambda: (print("FAIL  B4 gave up after 150 s: the watchdog never fired",
                                                flush=True), os._exit(1)))
    guard.daemon = True
    guard.start()

    def sdk_once(client):
        def once(messages, *_):
            r = client.chat.completions.create(model="m", messages=messages, max_tokens=5)
            meta = core.openai_meta(r)
            return r.choices[0].message.content, meta
        return once

    mdir = os.path.join(root, "M_wd")
    os.makedirs(mdir)
    core.HARD_CALL_TIMEOUT = 2
    for mode in ("silent", "keepalive"):
        srv = mock_server(mode)
        client = openai.OpenAI(base_url=f"http://127.0.0.1:{srv.getsockname()[1]}/v1", api_key="k",
                               timeout=400, max_retries=0)
        started = time.time()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            text, meta = core.call_with_retries(sdk_once(client), [{"role": "user", "content": "x" * 1000}],
                                                "probe", mdir, tries=2)
        took = time.time() - started
        check(f"B4 {mode}: ended by the watchdog, not the SDK's 400 s timeout (2 attempts x 2 s = {took:.1f} s)",
              text == "" and meta.get("n_attempts") == 2 and "hard limit" in str(meta.get("error"))
              and meta.get("attempts_without_usage") == 2 and 3.5 <= took < 15
              and out.getvalue().count("TIMEOUT (") == 2, f"{meta}\n{out.getvalue()[-300:]}")
        srv.close()
        core._consecutive[0] = 0
        core._recent.clear()

    srv = mock_server("keepalive")
    client = openai.OpenAI(base_url=f"http://127.0.0.1:{srv.getsockname()[1]}/v1", api_key="k",
                           timeout=400, max_retries=0)
    once = sdk_once(client)
    mdir = os.path.join(root, "M_wd_row")
    os.makedirs(mdir)
    started = time.time()
    code, out = run_s(core, mdir, lambda m, model, params: once(m), limit=1, call_timeout=1)
    took = time.time() - started
    path = os.path.join(mdir, "smoke_sample10", "test-fake-model-1_0.jsonl")
    r = json.loads(open(path).readline()) if os.path.exists(path) else {}
    check(f"B4 a whole sample row against the keep-alive server: written failed, exit 1 ({took:.1f} s)",
          code == 1 and r.get("empty") == 1 and r.get("n_attempts") == 5
          and str(r.get("error")).startswith("timeout") and r.get("attempts_without_usage") == 5
          and r.get("billed_completion_tokens") is None and took < 30,
          f"exit {code}; {json.dumps({k: r.get(k) for k in ('empty', 'n_attempts', 'error', 'attempts_without_usage')})}")
    srv.close()

    # the real Qwen runner's `once` (CutRetry): the first call is cut at the cap, the retry hangs
    qwen_src = open(os.path.join(HERE, "DOCVQA_Qwen_DeepInfra", "qwen_di_docvqa_eval.py"), "rb").read()
    os.makedirs(os.path.join(root, "DOCVQA_Qwen_DeepInfra"))
    with open(os.path.join(root, "DOCVQA_Qwen_DeepInfra", "qwen_di_docvqa_eval.py"), "wb") as fh:
        fh.write(qwen_src)
    os.environ.setdefault("DEEPINFRA_API_KEY", "test-key")   # the client is replaced before any call
    sys.modules.pop("docvqa_eval_core", None)                 # the runner must import THIS root's core
    spec = importlib.util.spec_from_file_location(
        "qwen_runner_under_test", os.path.join(root, "DOCVQA_Qwen_DeepInfra", "qwen_di_docvqa_eval.py"))
    qwen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwen)
    sys.modules.pop("docvqa_eval_core", None)
    qcore = qwen.core
    no_backoff(qcore)
    qcore.HARD_CALL_TIMEOUT = 2
    srv = mock_server("cut-then-keepalive")
    qwen.client = openai.OpenAI(base_url=f"http://127.0.0.1:{srv.getsockname()[1]}/v1", api_key="k",
                                timeout=1500, max_retries=0)
    ex = qcore.load_data()[0]
    started = time.time()
    with contextlib.redirect_stdout(io.StringIO()) as out:
        text, meta = qcore.call_with_retries(lambda m: qwen.once(m, "Qwen/Qwen3.5-9B", dict(qwen.WANTED)),
                                             qcore.build_messages(ex), "qwen", mdir, tries=2)
    took = time.time() - started
    check(f"B4 Qwen CutRetry: cut first call kept, each retry stage ended by the re-armed watchdog "
          f"({srv.served[0]} requests, {took:.1f} s)",
          text == "" and meta.get("n_attempts") == 2 and srv.served[0] == 3
          and meta.get("billed_prompt_tokens") == 100 and meta.get("billed_completion_tokens") == 8192
          and meta.get("attempts_without_usage") == 2 and "hard limit" in str(meta.get("error"))
          and 3.5 <= took < 15,
          json.dumps({k: meta.get(k) for k in ("n_attempts", "billed_prompt_tokens", "billed_completion_tokens",
                                                "attempts_without_usage", "error")}) + out.getvalue()[-300:])
    srv.close()
    guard.cancel()


# ---------------------------------------------------------------- C: the prompt


def legacy_prompt(path, question):
    """Evaluate the `prompt = (...)` expression of a standalone runner with `question` bound — read
    from the file, never by importing it (the standalone scripts run a whole benchmark at import)."""
    import ast
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "prompt"):
            return eval(compile(ast.Expression(node.value), path, "eval"), {"question": question})
    raise LookupError(path)


def test_c(base, new_src):
    print("\n== C  core.build_prompt against the standalone runners' prompt", flush=True)
    core = load_core(make_root(base, "c", new_src), "c_core")
    data = core.load_data()
    _, rows = core.load_sample("sample10")
    questions = [r["example"]["question"] for r in rows]
    for rel in ("qwen_DocVQA/qwen_eval.py", "gemma_DocVQA/gemma_eval_half1.py",
                "gemma_DocVQA/gemma_eval_half2.py", "gemini_eval.py", "openai_eval.py"):
        same = all(legacy_prompt(os.path.join(HERE, rel), q) == core.build_prompt({"question": q})
                   for q in questions)
        check(f"C {rel}: byte-identical prompt on all {len(questions)} sample questions", same)
    print("      rendered (sample question %s):\n%s" % (rows[0]["example"]["questionId"],
          "\n".join("      | " + line for line in core.build_prompt(rows[0]["example"]).split("\n"))))
    print(f"      sha1 {core.PROMPT_SHA1}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", default="1cc5bbd5",
                    help="git ref whose legacy core and GPT runner the working copies are compared with")
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = ap.parse_args()
    base = tempfile.mkdtemp(prefix="docvqa_s10_test_")
    try:
        new_core = open(os.path.join(HERE, "docvqa_eval_core.py"), "rb").read()
        new_runner = open(os.path.join(HERE, GPT_DIR, GPT_RUNNER), "rb").read()
        try:
            old_core = git_show(args.baseline, "docvqa_eval_core.py")
            old_runner = git_show(args.baseline, f"{GPT_DIR}/{GPT_RUNNER}")
        except (OSError, subprocess.CalledProcessError) as error:
            # A copy with no git history (Quest): A cannot run, and says so — a skip is not a pass.
            old_core = old_runner = None
            SKIPS.append(f"A SKIPPED: baseline {args.baseline} not readable from git ({type(error).__name__})")
            print(SKIPS[-1], flush=True)
        print(f"working copy:   core {hashlib.md5(new_core).hexdigest()}, "
              f"runner {hashlib.md5(new_runner).hexdigest()}")
        if old_core is not None:
            print(f"baseline {args.baseline}: core {hashlib.md5(old_core).hexdigest()}, "
                  f"runner {hashlib.md5(old_runner).hexdigest()}")
            test_a1(base, old_core, new_core)
            test_a2_a3(base, old_core, new_core, old_runner, new_runner)
        test_b(base, new_core)
        test_watchdog(base, new_core)
        test_c(base, new_core)
    finally:
        if args.keep:
            print(f"\nscratch kept at {base}")
        else:
            shutil.rmtree(base, ignore_errors=True)
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)} passed, {len(failed)} failed"
          + "".join(f"\n{line}" for line in SKIPS))
    sys.exit(1 if failed else 2 if SKIPS else 0)        # 2: nothing failed, but a part did not run


if __name__ == "__main__":
    main()
