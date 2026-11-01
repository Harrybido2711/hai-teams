"""Wonderbread — shared core: the pilot's 10% samples and the full sets, the human-labelled anchor
sets, the generation loop every tested model runs (`run_cli`), and upstream's own message builders,
judge prompts and scorers. Identical for every model and every judge; each model's runner holds only
its client and parameters (`judge_core.Generator`).

Scope: the two judged subtasks (LLM_as_judge/JUDGE_SELECTION.md §7.3-7.5) — Question Answering (Judge 1:
four 1-3 scores, 1 = BEST) and SOP Generation (Judge 2: line-to-line matching, tallied into
precision / recall / ordering). SOP Improvement's scorer does not execute (§7.5) and is out.

**Nothing upstream is re-typed.** Every prompt, message builder and scorer is imported from the
vendored package, and generation runs upstream's own `generate_response` / `run` functions with
only their transport (`_fetch_completion`) replaced — so what the model sees is what upstream built.
`wonderbread/helpers.py` imports anthropic, google-generativeai, pydrive2 and together at module
level; the functions used here call none of them, so a missing one is replaced by a stand-in that
raises if anything ever touches it (`_absent`). Quest's env lacks anthropic and pydrive2.

Declared deviations from upstream (LLM_as_judge/JUDGE_SELECTION.md §7.3-7.4):
- judges run at temperature 0 where the route allows it — upstream's SOP judge ran at the API
  default 1.0 (D3); 3 replicates, not 1;
- QA scores are parsed into {1,2,3} with a parse-fail status — upstream stored the raw string;
- SOP indices are parsed from JSON with fences stripped (Claude fences JSON mode on OpenRouter;
  upstream's bare json.loads would raise) and an index outside the list is a parse failure, not a
  match. Since 2026-10-31 (the full run; the pilot's rows were parsed without it): when the object
  is invalid JSON, its final `"index": N` field is read — Haiku 5.5 copies a line's quoted text
  ("01/01/2022") into the scratchpad string unescaped, and one unreadable line drops its whole SOP
  pair from a replicate's precision / recall;
- `preprocess_sop`'s strip-up-to-the-first-"." (B4) is KEPT, for comparability;
- each tested model runs at its settled config, declared in its runner — upstream's GPT-4 ran at
  temperature 0, which gpt-5.6-luna does not accept;
- generation (not the pilot's rows): every screenshot is downscaled to a 1188-px long side and sent
  as JPEG q85 (`IMAGE_POLICY`), for every model alike — the user's decision, 2026-10-31. Upstream's
  are 3456x2234-class PNGs, and at that size Gemini's route refuses requests over 30 MB, Together
  refuses Qwen's past 255 MiB decoded, and DeepInfra's Gemma hung on a 30-image request;
- a model with a per-request image cap (`Generator.max_images`: Qwen3.5-9B on Together, 10) is never
  sent a request over it: the unit is written `over_image_limit`, uncalled and unjudged, and that
  model is reported on the units within its cap — every row records `n_images`, so any model can be
  scored on the same subset (the user's decision, 2026-10-31).
"""

import argparse
import base64
import collections
import hashlib
import importlib
import io
import json
import os
import re
import sys
import tempfile
import time
import types

WB_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(WB_ROOT))
import judge_core as jc  # noqa: E402

DATA_DIR = os.path.join(WB_ROOT, "data")
# upstream's own location for demos (df_rankings.csv: path_to_demo_folder = data/demos/<folder>).
# Filled by fetch_sample_demos.py with only the sampled demos; gitignored.
DEMOS_DIR = os.path.join(DATA_DIR, "demos")
QA_CSV = os.path.join(DATA_DIR, "qa_dataset.csv")
RANKINGS_CSV = os.path.join(DATA_DIR, "df_rankings.csv")
_HC = os.path.join(DATA_DIR, "experimental_results")
ANCHOR_QA_HUMAN = os.path.join(_HC, "question_answering", "human_correlation", "qna-human-sample-final.csv")
ANCHOR_QA_GPT4 = os.path.join(_HC, "question_answering", "human_correlation", "qna-GPT4-sample-final.csv")
ANCHOR_SOP = os.path.join(_HC, "sop_generation", "human_correlation", "sample_human_eval.csv")
SAMPLE_PATH = os.path.join(WB_ROOT, "sample_uids.json")
ENV_PATH = os.path.join(WB_ROOT, ".env")

EXPECTED_QA = 120
EXPECTED_RANK1 = 162
EXPECTED_ANCHOR = 30
QA_CRITERIA = ("completeness", "soundness", "clarity", "compactness")
WITH_REFERENCE = ("completeness", "soundness")        # the two that see the Human Label (A2)
# upstream run_experiments.py for SOP generation: three non-pairwise ablations
SOP_ABLATIONS = {"td": dict(is_td=True, is_kf=False, is_act=False),
                 "td_kf": dict(is_td=True, is_kf=True, is_act=False),
                 "td_kf_act": dict(is_td=True, is_kf=True, is_act=True)}
HEADER_DEMARCATOR = "----------------------------------------"   # collect_results.py
SCORER_VERSION = "wb_upstream_v1"
ANCHOR_PRODUCER = "anchor-human-sample"   # upstream's GPT4 / GeminiPro / Claude3 responses


# ---------------------------------------------------------------- upstream, imported


def _absent(name):
    module = types.ModuleType(name)
    module.__path__ = []                       # importable as a package, so `a.b` resolves
    module.__absent__ = True                   # judge_core's clients check this: never a real client

    def __getattr__(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)

        def unavailable(*args, **kwargs):
            raise RuntimeError(f"{name}.{attr} is not installed here, and the judge pilot never "
                               f"calls it — something reached an upstream path it should not")
        return unavailable
    module.__getattr__ = __getattr__
    return module


_UPSTREAM_CLIENT_LIBS = ("anthropic", "openai", "together", "google", "google.generativeai",
                         "pydrive2", "pydrive2.auth", "pydrive2.drive")
_upstream = {}


def upstream():
    """The vendored modules this pilot uses, imported once."""
    if _upstream:
        return types.SimpleNamespace(**_upstream)
    stubbed = []
    for name in _UPSTREAM_CLIENT_LIBS:
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = _absent(name)
            parent, _, child = name.rpartition(".")
            if parent in sys.modules:
                setattr(sys.modules[parent], child, sys.modules[name])
            stubbed.append(name)
    if WB_ROOT not in sys.path:
        sys.path.insert(0, WB_ROOT)
    base = "wonderbread.benchmark.tasks"
    _upstream.update(
        helpers=importlib.import_module("wonderbread.helpers"),
        qa_prompts=importlib.import_module(f"{base}.knowledge_transfer.question_answering.prompts"),
        qa_gen=importlib.import_module(f"{base}.knowledge_transfer.question_answering.generate_responses"),
        sop_main=importlib.import_module(f"{base}.documentation.sop_generation.main"),
        sop_eval=importlib.import_module(f"{base}.documentation.sop_generation.eval"),
        eval_prompts=importlib.import_module(f"{base}.documentation.sop_generation.eval_prompts"),
        metrics=importlib.import_module(f"{base}.documentation.sop_generation.metrics"),
        stubbed=stubbed,
    )
    return types.SimpleNamespace(**_upstream)


# ---------------------------------------------------------------- data and the samples


def _pd():
    import pandas as pd
    return pd


def load_qa():
    df = _pd().read_csv(QA_CSV)
    if len(df) != EXPECTED_QA:
        raise SystemExit(f"{QA_CSV}: expected {EXPECTED_QA} rows, found {len(df)}")
    return {f"qa{i:03d}": row for i, row in df.iterrows()}


def task_site(task_id):
    task = upstream().helpers.get_webarena_task_json(int(task_id))
    if task is None:
        raise SystemExit(f"no WebArena task json for task_id {task_id}")
    return task["sites"][0]


def load_rank1():
    """The 162 rank-1 gold demos SOP Generation runs on (`run_experiment(is_use_rank_1_df=True)`)."""
    df = _pd().read_csv(RANKINGS_CSV)
    df = df[df["rank"] == 1]
    if len(df) != EXPECTED_RANK1:
        raise SystemExit(f"{RANKINGS_CSV}: expected {EXPECTED_RANK1} rank-1 demos, found {len(df)}")
    return {row["folder_name"]: int(row["task_id"]) for _, row in df.iterrows()}


def build_sample():
    """QA stratified by question template (11 strata), SOP Generation by website (4) — 10%, seeded,
    drawn once and committed."""
    qa = load_qa()
    qa_pick = jc.stratified_sample(list(qa), lambda u: qa[u]["Question Template"])
    qa_sizes = collections.Counter(r["Question Template"] for r in qa.values())
    rank1 = load_rank1()
    site = {demo: task_site(t) for demo, t in rank1.items()}
    sop_pick = jc.stratified_sample(list(rank1), site.get)
    sop_sizes = collections.Counter(site.values())
    doc = {"seed": jc.SAMPLE_SEED, "fraction": jc.SAMPLE_FRACTION,
           "qa": {"note": "uid = qa<row index in data/qa_dataset.csv>; stratum = Question Template",
                  "strata": {k: {"population": qa_sizes[k], "sampled": len(v), "uids": v}
                             for k, v in qa_pick.items()}},
           "sop": {"note": "uid = rank-1 demo folder (data/df_rankings.csv); stratum = WebArena site",
                   "strata": {k: {"population": sop_sizes[k], "sampled": len(v), "uids": v}
                              for k, v in sop_pick.items()}}}
    with open(SAMPLE_PATH, "w") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return doc


def load_sample(task):
    if not os.path.exists(SAMPLE_PATH):
        raise SystemExit(f"{SAMPLE_PATH} missing — run `python wb_eval_core.py --build-sample` once")
    with open(SAMPLE_PATH) as fh:
        strata = json.load(fh)[task]["strata"]
    return [u for k in sorted(strata) for u in strata[k]["uids"]]


def load_set(task, set_name):
    """QA uids or SOP demo folders. `full`: all 120 QA items / all 162 rank-1 demos, in a fixed
    order. `sample`: the pilot's (`load_sample`)."""
    if set_name == "sample":
        return load_sample(task)
    return sorted(load_qa()) if task == "qa" else sorted(load_rank1())


def demos_for(task, units):
    """The demo folders these units read: a QA item's Task ID(s), or the SOP demos themselves."""
    if task == "sop":
        return set(units)
    qa = load_qa()
    return {x.strip() for uid in units for x in str(qa[uid]["Task ID(s)"]).split(",")}


def demos_needed(set_name="sample"):
    """Every demo folder a set touches: its QA items' Task ID(s) plus its SOP demos — 31 for the
    sample, 255 for the full set."""
    return sorted(demos_for("qa", load_set("qa", set_name)) | demos_for("sop", load_set("sop", set_name)))


def check_demo(folder):
    """What upstream's path helpers assert, checked up front so a gap fails before any call."""
    h = upstream().helpers
    path = os.path.join(DEMOS_DIR, folder)
    problems = []
    if not os.path.isdir(path):
        return [f"{folder}: not on disk under {DEMOS_DIR}"]
    if h.get_path_to_sop_txt(path, is_no_assert=True) is None:
        problems.append(f"{folder}: no SOP*.txt")
    if h.get_path_to_trace_json(path, is_no_assert=True) is None:
        problems.append(f"{folder}: no trace .json")
    shots = os.path.join(path, "screenshots")
    if not os.path.isdir(shots) or not os.listdir(shots):
        problems.append(f"{folder}: no screenshots/")
    return problems


# ---------------------------------------------------------------- anchor sets (human-labelled)


def load_anchor_qa():
    """Upstream's 30-item human sample: responses from its own models, scored by humans; its GPT-4
    judge's scores on the same rows sit beside them (LLM_as_judge/JUDGE_SELECTION.md §7.3, A3)."""
    pd = _pd()
    human = pd.read_csv(ANCHOR_QA_HUMAN)
    gpt4 = pd.read_csv(ANCHOR_QA_GPT4)
    if len(human) != EXPECTED_ANCHOR or len(gpt4) != EXPECTED_ANCHOR:
        raise SystemExit(f"anchor QA: expected {EXPECTED_ANCHOR} rows in each file")
    if not (human["Response"].fillna("") == gpt4["Response"].fillna("")).all():
        raise SystemExit("anchor QA: the human and GPT-4 files do not hold the same responses in order")
    out = {}
    for i, row in human.iterrows():
        out[f"anchor{i:02d}"] = {
            "question": row["Question Instantiation"], "human_label": row["Human Label"],
            "response": row["Response"], "template": row["Question Template"],
            "human": {c: row[f"{c}_score"] for c in QA_CRITERIA},
            "upstream_gpt4": {c: gpt4.loc[i, f"{c}_score"] for c in QA_CRITERIA}}
    return out


def load_anchor_sop():
    """Upstream's 30-pair human sample: a generated SOP, its gold SOP (already through
    add_gold_sops), and human precision/recall beside GPT-4's (`auto_eval_*`)."""
    df = _pd().read_csv(ANCHOR_SOP)
    if len(df) != EXPECTED_ANCHOR:
        raise SystemExit(f"{ANCHOR_SOP}: expected {EXPECTED_ANCHOR} rows, found {len(df)}")
    return {f"anchor{i:02d}": {
        "demo": row["demo_name"], "ablation": row["ablation"], "pred_sop": row["pred_sop"],
        "gold_sop": row["gold_sop"],
        "human": {"precision": row["manual_eval_precision"], "recall": row["manual_eval_recall"]},
        "upstream_gpt4": {"precision": row["auto_eval_precision"], "recall": row["auto_eval_recall"],
                          "ordering": row["ordering"]}} for i, row in df.iterrows()}


# ---------------------------------------------------------------- generation: upstream builders


def gold_sop(demo):
    """upstream eval.add_gold_sops for one demo: the SOP file, title line dropped."""
    df = _pd().DataFrame([{"demo_name": demo}])
    return upstream().sop_eval.add_gold_sops(df, DEMOS_DIR)["gold_sop"].iloc[0]


def generated_sop(path_to_file):
    """collect_results.py: the SOP is everything after the header demarcator."""
    with open(path_to_file) as fh:
        contents = fh.read()
    return contents[contents.index(HEADER_DEMARCATOR) + len(HEADER_DEMARCATOR):].strip()


def count_images(messages):
    return sum(1 for m in messages for c in (m["content"] if isinstance(m["content"], list) else [])
               if c.get("type") == "image_url")


# ---------------------------------------------------------------- generation: the shared loop
# What the model sees is built by upstream, not here: QA calls
# `question_answering/generate_responses.py::generate_response`, SOP Generation calls
# `sop_generation/main.py::run`, each with its `_fetch_completion` replaced by `Transport`, which does
# what upstream's GPT4 path did (`helpers._fetch_openai_completion`): prepend `SYSTEM_PROMPT`, send
# the messages as built — screenshots as base64 `image_url` parts — and return the text. Upstream's
# `run` makes ONE call per SOP for any model but its Claude3 (which it splits past 10 images), so one
# row is one call. SOP files are written by `run` exactly as upstream writes them, then read back the
# way `collect_results.py` reads them.


IMAGE_LONG_SIDE = 1188
IMAGE_JPEG_QUALITY = 85
IMAGE_POLICY = {"long_side": IMAGE_LONG_SIDE, "format": "JPEG", "quality": IMAGE_JPEG_QUALITY}
OVER_LIMIT = "over_image_limit"
_shrunk = {}


def shrink_image_url(url):
    """One `data:` URL re-encoded per IMAGE_POLICY. Cached by digest: consecutive SOP ablations and
    QA items send the same screenshots."""
    key = hashlib.sha1(url.encode()).hexdigest()
    if key not in _shrunk:
        from PIL import Image
        im = Image.open(io.BytesIO(base64.b64decode(url.partition(",")[2]))).convert("RGB")
        scale = IMAGE_LONG_SIDE / max(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, "JPEG", quality=IMAGE_JPEG_QUALITY)
        if len(_shrunk) >= 256:
            _shrunk.pop(next(iter(_shrunk)))
        _shrunk[key] = "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()
    return _shrunk[key]


def shrink_images(messages):
    """A copy of `messages` with every image part re-encoded per IMAGE_POLICY; text untouched."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = [dict(c, image_url=dict(c["image_url"], url=shrink_image_url(c["image_url"]["url"])))
                       if c.get("type") == "image_url" else c for c in content]
        out.append(dict(m, content=content))
    return out


class GenerationFailed(Exception):
    pass


class Transport:
    """Stands in for upstream's `_fetch_completion(messages, model)` on the GPT4 path."""

    def __init__(self, gen, client, model, params, model_dir, dry_run=False):
        self.gen, self.client, self.model, self.params = gen, client, model, params
        self.model_dir, self.dry_run = model_dir, dry_run
        self.calls = []          # this unit's calls; cleared per unit
        self.largest = None      # the largest request of the whole run — a provider's size limit

    def __call__(self, messages, model=None, model_name=None, **kwargs):
        full = self.gen.messages(shrink_images(
            [{"role": "system", "content": upstream().helpers.SYSTEM_PROMPT}] + messages))
        info = {"n_images": count_images(full), "n_messages": len(full),
                "request_mb": round(len(json.dumps(full)) / 1e6, 2)}
        if self.gen.max_images is not None and info["n_images"] > self.gen.max_images:
            self.calls.append(dict(info, status=OVER_LIMIT))
            raise GenerationFailed(f"{info['n_images']} images > this model's cap of "
                                   f"{self.gen.max_images}: not sent")
        if self.largest is None or info["request_mb"] > self.largest["request_mb"]:
            self.largest = info
        if self.dry_run:
            self.calls.append(dict(info, status="dry_run"))
            return "1. DRY RUN"
        once = self.gen.once(self.client, self.model, full, self.params)
        text, meta, status = jc.call_with_retries(once, self.model, self.model_dir)
        self.calls.append(dict(info, meta=meta, status=status, text=text))
        if status != "ok":
            raise GenerationFailed(f"{status} after {meta.get('n_attempts')} attempts")
        return text


def _gen_qa(transport, path, config, done, uids):
    up = upstream()
    qa = load_qa()
    up.qa_gen._fetch_completion = transport
    for n, uid in enumerate(uids, 1):
        if uid in done:
            continue
        row = qa[uid]
        transport.calls.clear()
        try:
            up.qa_gen.generate_response(DEMOS_DIR, row["Task ID(s)"], row["Question Instantiation"],
                                        row["Evidence"], "GPT4")
        except GenerationFailed as error:
            print(f"  {uid}: {error}", flush=True)
        call = transport.calls[-1]
        meta = call.get("meta", {})
        out = {"uid": uid, "task": "qa", "status": call["status"], "response": call.get("text", ""),
               "question": row["Question Instantiation"], "human_label": row["Human Label"],
               "template": row["Question Template"], "evidence": row["Evidence"],
               "task_ids": row["Task ID(s)"], "n_images": call["n_images"],
               "request_mb": call["request_mb"], "config": config, **meta}
        if not transport.dry_run:
            jc.append_row(path, out)
            if call["status"] != OVER_LIMIT:          # never called: not a provider outcome
                jc.note_outcome(call["status"], transport.model, transport.model_dir)
        print(f"  [{n}/{len(uids)}] {uid} {row['Evidence'][:40]:40s} {call['status']:8s} "
              f"images={call['n_images']} {call['request_mb']}MB out={meta.get('completion_tokens')} "
              f"reasoning={meta.get('reasoning_tokens')} {meta.get('latency_s')}s", flush=True)
        if not transport.dry_run:
            time.sleep(1.0)
    return uids


def _gen_sop(transport, path, config, done, demos, upstream_dir):
    up = upstream()
    rank1 = load_rank1()
    up.sop_main._fetch_completion = transport
    # upstream writes the model name into the SOP file name: a "/" there would be a directory
    slug = jc.model_slug(transport.model)
    units = [(d, a) for d in demos for a in SOP_ABLATIONS]
    for n, (demo, ablation) in enumerate(units, 1):
        uid = f"{demo}|{ablation}"
        if uid in done:
            continue
        flags = SOP_ABLATIONS[ablation]
        transport.calls.clear()
        try:
            up.sop_main.run(os.path.join(DEMOS_DIR, demo), upstream_dir, rank1[demo], slug,
                            is_pairwise=False, is_verbose=False, **flags)
        except GenerationFailed as error:
            print(f"  {uid}: {error}", flush=True)
        call = transport.calls[-1]
        meta = call.get("meta", {})
        short = up.sop_main.kwarg_setting_to_ablation(dict(flags, is_pairwise=False, model=slug))
        sop_file = os.path.join(upstream_dir, demo, f"Generated-SOP - {short} - {demo}.txt")
        ok = call["status"] == "ok"
        out = {"uid": uid, "task": "sop", "demo": demo, "task_id": rank1[demo],
               "site": task_site(rank1[demo]), "ablation": ablation, "status": call["status"],
               "raw_response": call.get("text", ""),
               "pred_sop": generated_sop(sop_file) if ok else "",
               "gold_sop": gold_sop(demo), "n_images": call["n_images"],
               "request_mb": call["request_mb"], "config": config, **meta}
        if not transport.dry_run:
            jc.append_row(path, out)
            if call["status"] != OVER_LIMIT:          # never called: not a provider outcome
                jc.note_outcome(call["status"], transport.model, transport.model_dir)
        print(f"  [{n}/{len(units)}] {uid:46s} {call['status']:8s} images={call['n_images']} "
              f"{call['request_mb']}MB out={meta.get('completion_tokens')} "
              f"reasoning={meta.get('reasoning_tokens')} {meta.get('latency_s')}s "
              f"pred_lines={len(sop_lines(out['pred_sop'])) if ok else '-'}", flush=True)
        if not transport.dry_run:
            time.sleep(1.0)
    return [f"{d}|{a}" for d, a in units]


def run_cli(gen, model_dir, description):
    """`--task qa`: one answer per question. `--task sop`: one SOP per demo x upstream's three
    ablations (td, td+kf, td+kf+act). `--set full` (default) or the pilot's `sample`. Nothing here
    scores; judging is WB_Judges/.

    Exit 1 unless every unit has an `ok` row; exit 2 if any row hit the output cap
    (`finish_reason=length`), so a judge job waiting on `afterok` never starts on a truncated answer."""
    from dotenv import load_dotenv

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--task", required=True, choices=("qa", "sop"))
    ap.add_argument("--model", default=gen.model)
    ap.add_argument("--set", dest="set_name", default="full", choices=jc.SETS)
    ap.add_argument("--limit", type=int, default=0, help="first N items (qa) / demos (sop); smoke/")
    ap.add_argument("--dry-run", action="store_true", help="build every request; call nothing")
    args = ap.parse_args()

    units = load_set(args.task, args.set_name)[:args.limit or None]
    # every demo this run reads, checked before the first call — a gap fails here, not mid-run
    problems = [p for d in sorted(demos_for(args.task, units)) for p in check_demo(d)]
    if problems:
        raise SystemExit(f"{len(problems)} demo problem(s) — run `fetch_sample_demos.py --set "
                         f"{args.set_name}` first:\n  " + "\n  ".join(problems[:5]))
    load_dotenv(ENV_PATH)
    jc.HARD_CALL_TIMEOUT = gen.call_timeout
    smoke = args.limit > 0 or args.dry_run
    # QA and SOP share this folder, and smoke shares it with the real run: each names its markers
    jc.MARKER_PREFIX = ("smoke_" if smoke else "") + f"{args.set_name}_{args.task}_"
    if args.dry_run:
        client, params, notes = None, dict(gen.wanted), ["dry run: not negotiated"]
    else:
        jc.clear_stale_halt_markers(model_dir)
        client = gen.make_client()
        print(f"negotiating parameter surface for {args.model} ...", flush=True)
        params, notes = jc.negotiate(client, args.model, gen.wanted, model_dir, gen.required,
                                     gen.fixed)
        print(f"  accepted: {params}" + (f"  notes: {notes}" if notes else ""), flush=True)
    config = dict(gen.config(args.model, params, notes), task=args.task, set=args.set_name,
                  input="upstream builders verbatim, SYSTEM_PROMPT prepended as _fetch_openai_completion",
                  images=IMAGE_POLICY)
    if args.set_name == "sample":
        config.update(sample_seed=jc.SAMPLE_SEED, sample_fraction=jc.SAMPLE_FRACTION)
    parts = jc.set_parts(args.set_name)
    path = jc.out_path(model_dir, smoke, *parts, args.task, f"{jc.model_slug(args.model)}.jsonl")
    if not args.dry_run:
        jc.lock_output(path)
    done = {} if args.dry_run else jc.load_done(path, config)
    transport = Transport(gen, client, args.model, params, model_dir, dry_run=args.dry_run)
    print(f"{args.set_name}/{args.task}: {'DRY RUN, nothing is called' if args.dry_run else path}",
          flush=True)
    if args.task == "qa":
        uids = _gen_qa(transport, path, config, done, units)
    else:
        # upstream's run() writes SOP files; a dry run's placeholders must never land beside real ones
        upstream_dir = (tempfile.mkdtemp(prefix="wb_dry_run_") if args.dry_run else
                        os.path.join(model_dir, "smoke" if smoke else "results", *parts,
                                     "sop_upstream_files"))
        uids = _gen_sop(transport, path, config, done, units, upstream_dir)
    if transport.largest:
        print(f"largest request: {transport.largest['request_mb']} MB, "
              f"{transport.largest['n_images']} images", flush=True)
    if args.dry_run:
        return
    rows = [r for r in jc.read_rows(path) if r["uid"] in set(uids)]
    finish = collections.Counter(r.get("finish_reason") for r in rows)
    print(f"\nDONE {len(rows)}/{len(uids)} rows  status={jc.status_counts(rows)}  "
          f"finish={dict(finish)}  calls={jc.STATS}", flush=True)
    over = sum(r["status"] == OVER_LIMIT for r in rows)
    if over:
        print(f"  {over} unit(s) over this model's {gen.max_images}-image cap: not called, not "
              f"judged — a ceiling, not a failure", flush=True)
    odd = jc.unexpected_finish(rows)
    if odd:
        print(f"STOP: {sum(odd.values())} ok row(s) ended with finish_reason {dict(odd)}, not a known "
              f"complete answer — inspect them before judging.", flush=True)
    # length first: a row whose whole cap went on reasoning is `failed` AND finish=length, and the
    # decision it needs is the cap, not a resubmit
    if finish.get("length"):
        print(f"STOP: {finish['length']} row(s) hit the output cap — the cap bound. Decide whether "
              f"to raise it (archive the results first) before judging.", flush=True)
        sys.exit(2)
    # a unit over the model's image cap is finished (uncalled by design); anything else must be ok
    if odd or len(rows) < len(uids) or any(r["status"] not in ("ok", OVER_LIMIT) for r in rows):
        sys.exit(1)


# ---------------------------------------------------------------- judging: upstream prompts


def qa_judge_messages(criterion, question, human_label, response):
    """eval.py get_<criterion>_score → _fetch_completion(…, 'GPT4') → _fetch_openai_completion,
    which prepends SYSTEM_PROMPT. Same message shapes as upstream sent."""
    up = upstream()
    fn = getattr(up.qa_prompts, f"prompt__{criterion}_score")
    prompt = fn(question, human_label, response) if criterion in WITH_REFERENCE else fn(question, response)
    return [{"role": "system", "content": up.helpers.SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}]


def qa_skip(response):
    """Upstream writes "NA" without calling the judge when the response is NaN or < 5 characters."""
    return response is None or (isinstance(response, float) and response != response) \
        or len(str(response)) < 5


_SCORE_LEAD = re.compile(r"^\W*(?:score|rating)?\W*([123])(?!\d)", re.IGNORECASE)
_INTEGER = re.compile(r"(?<![\d.])\d+(?!\d|\.\d)")      # "1." is the integer 1; "1.5" is not


def parse_qa_score(text):
    """1, 2 or 3 — or None. Upstream asks for the bare number. Accepted: the number leading the
    answer with no rival on its first line ("2", "**2**", "Score: 2", "2 - mostly complete",
    "Rating: 2" then an explanation); otherwise exactly one integer in the whole answer, and it in
    range. Two candidates ("1 or 2") is ambiguous, never a guess."""
    t = jc.strip_fences(text).strip()
    m = _SCORE_LEAD.match(t)
    if m and len(_INTEGER.findall(t.split("\n", 1)[0])) == 1:
        return int(m.group(1))
    numbers = _INTEGER.findall(t)
    if len(numbers) == 1 and numbers[0] in ("1", "2", "3"):
        return int(numbers[0])
    return None


def sop_lines(sop):
    return upstream().sop_eval.preprocess_sop(sop)


def sop_judge_messages(query_line, list_of_lines):
    """metrics.PairwiseComparison → eval_completion.get_completion → helpers.fetch_openai_json_completion:
    SYSTEM_PROMPT, then the prompt as a plain-string user turn, JSON mode."""
    up = upstream()
    prompt = up.eval_prompts.map_query_to_one_prompt(query_line, list_of_lines)
    return [{"role": "system", "content": up.helpers.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]


SOP_RESPONSE_FORMAT = {"type": "json_object"}


# The object's LAST field, anchored at its closing brace: `"index"` quoted inside the free text
# earlier in the answer never matches.
_TRAILING_INDEX = re.compile(r'"index"\s*:\s*"?(-?\d+(?:\.\d+)?)"?\s*\}\s*$')


def make_index_parser(n_lines):
    def parse(text):
        obj = jc.first_json_object(text)
        if obj is not None:
            if "index" not in obj:
                return None
            raw = obj["index"]
        else:
            # invalid JSON only because the scratchpad holds unescaped quotes: read the last field —
            # of an answer that is an object, never of prose that happens to end in one
            body = jc.strip_fences(text).strip()
            m = _TRAILING_INDEX.search(body) if body.startswith("{") else None
            if not m:
                return None
            raw = m.group(1)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if value != int(value):
            return None
        index = int(value)
        return index if index == -1 or 0 <= index < n_lines else None
    return parse


def sop_metrics(pred_indices, gold_indices):
    """Upstream's own precision / recall / ordering (metrics.PairwiseComparison), fed our parsed
    indices instead of fetching; F1 as run_sop_generation.py computes it. None where upstream
    would divide by zero."""
    if not pred_indices or not gold_indices or None in pred_indices or None in gold_indices:
        return None
    PC = upstream().metrics.PairwiseComparison
    pc = PC.__new__(PC)                            # __init__ would call the API
    pc.pred_sop_len, pc.ref_sop_len = len(pred_indices), len(gold_indices)
    pc.completions = {**{f"pred_line_{i}": {"index": v} for i, v in enumerate(pred_indices)},
                      **{f"gold_line_{j}": {"index": v} for j, v in enumerate(gold_indices)}}
    p, r, o = pc.precision(), pc.recall(), pc.ordering()
    return {"precision": p, "recall": r, "ordering": o,
            "f1": (2 * p * r / (p + r)) if (p + r) else None,
            "n_lines_pred_sop": len(pred_indices), "n_lines_gold_sop": len(gold_indices)}


if __name__ == "__main__":
    if "--build-sample" in sys.argv:
        if os.path.exists(SAMPLE_PATH) and "--force" not in sys.argv:
            raise SystemExit(f"{SAMPLE_PATH} exists; the sample is drawn once. --force to redraw.")
        doc = build_sample()
        for task in ("qa", "sop"):
            for k, v in doc[task]["strata"].items():
                print(f"{task:3s} {v['sampled']:2d} of {v['population']:3d}  {k[:80]}")
        print(f"demos needed: {len(demos_needed())}")
    elif "--demos-needed" in sys.argv:
        print("\n".join(demos_needed()))
    else:
        print(__doc__)
