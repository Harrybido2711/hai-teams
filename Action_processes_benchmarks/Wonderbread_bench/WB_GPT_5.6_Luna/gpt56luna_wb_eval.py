"""GPT-5.6-Luna — Wonderbread QA answers and generated SOPs for the judge pilot. **The project's
OpenAI slot.**

`--task qa`: the 12 sampled questions (`../sample_uids.json`). `--task sop`: the 16 sampled rank-1
demos x upstream's three ablations (td, td+kf, td+kf+act) = 48 SOPs. Judging is
`../WB_Judges/wb_judge.py`; nothing here scores.

**What the model sees is built by upstream, not by this file.** QA calls
`question_answering/generate_responses.py::generate_response`, SOP Generation calls
`sop_generation/main.py::run`, each with its `_fetch_completion` replaced by `Transport`, which does
what upstream's GPT4 path did (`helpers._fetch_openai_completion`): prepend `SYSTEM_PROMPT`, send
the messages as built — screenshots as base64 `image_url` parts — and return the text. SOP files are
written by upstream's `run` exactly as upstream writes them, then read back the way
`collect_results.py` reads them.

**Settled config** (`model-parameters.md` § Settled — the project's GPT): `reasoning_effort="low"`,
`seed=42`; `temperature` is unsettable (upstream's GPT-4 ran at 0 — declared deviation).
`max_completion_tokens=8192`: upstream capped the visible answer at 4096 (`max_tokens=4096`), and
this cap also has to hold `low` reasoning; any `finish_reason=length` makes the run exit 2, so the
judges' `afterok` job does not start on a truncated answer. **Visible reasoning:** upstream's prompts ask for a 2-3
sentence answer (QA) or an SOP; nothing is added.

    python gpt56luna_wb_eval.py --task qa --dry-run     # build every request, call nothing
    python gpt56luna_wb_eval.py --task sop --limit 1    # smoke: 1 demo x 3 ablations, writes smoke/
    python gpt56luna_wb_eval.py --task sop              # the sample, writes results/
"""

import argparse
import collections
import json
import os
import sys
import tempfile
import time

from dotenv import load_dotenv

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import wb_eval_core as wb  # noqa: E402
jc = wb.jc


class GenerationFailed(Exception):
    pass


class Transport:
    """Stands in for upstream's `_fetch_completion(messages, model)` on the GPT4 path."""

    def __init__(self, client, model, params, dry_run=False):
        self.client, self.model, self.params, self.dry_run = client, model, params, dry_run
        self.calls = []

    def __call__(self, messages, model=None, model_name=None, **kwargs):
        full = [{"role": "system", "content": wb.upstream().helpers.SYSTEM_PROMPT}] + messages
        info = {"n_images": wb.count_images(full), "n_messages": len(full),
                "request_mb": round(len(json.dumps(full)) / 1e6, 2)}
        if self.dry_run:
            self.calls.append(dict(info, status="dry_run"))
            return "1. DRY RUN"
        once = jc.gen_once(self.client, self.model, full, self.params)
        text, meta, status = jc.call_with_retries(once, self.model, MODEL_DIR)
        self.calls.append(dict(info, meta=meta, status=status, text=text))
        if status != "ok":
            raise GenerationFailed(f"{status} after {meta.get('n_attempts')} attempts")
        return text


def run_qa(transport, path, config, done, limit):
    up = wb.upstream()
    qa = wb.load_qa()
    uids = wb.load_sample("qa")[:limit or None]
    up.qa_gen._fetch_completion = transport
    for n, uid in enumerate(uids, 1):
        if uid in done:
            continue
        row = qa[uid]
        transport.calls.clear()
        try:
            up.qa_gen.generate_response(wb.DEMOS_DIR, row["Task ID(s)"], row["Question Instantiation"],
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
        if call["status"] != "dry_run":
            jc.append_row(path, out)
            jc.note_outcome(call["status"], transport.model, MODEL_DIR)
        print(f"  [{n}/{len(uids)}] {uid} {row['Evidence'][:40]:40s} {call['status']:8s} "
              f"images={call['n_images']} {call['request_mb']}MB out={meta.get('completion_tokens')} "
              f"reasoning={meta.get('reasoning_tokens')} {meta.get('latency_s')}s", flush=True)
        if not transport.dry_run:
            time.sleep(1.0)
    return uids


def run_sop(transport, path, config, done, limit, upstream_dir, model):
    up = wb.upstream()
    rank1 = wb.load_rank1()
    demos = wb.load_sample("sop")[:limit or None]
    up.sop_main._fetch_completion = transport
    units = [(d, a) for d in demos for a in wb.SOP_ABLATIONS]
    for n, (demo, ablation) in enumerate(units, 1):
        uid = f"{demo}|{ablation}"
        if uid in done:
            continue
        flags = wb.SOP_ABLATIONS[ablation]
        transport.calls.clear()
        try:
            up.sop_main.run(os.path.join(wb.DEMOS_DIR, demo), upstream_dir, rank1[demo], model,
                            is_pairwise=False, is_verbose=False, **flags)
        except GenerationFailed as error:
            print(f"  {uid}: {error}", flush=True)
        call = transport.calls[-1]
        meta = call.get("meta", {})
        short = up.sop_main.kwarg_setting_to_ablation(dict(flags, is_pairwise=False, model=model))
        sop_file = os.path.join(upstream_dir, demo, f"Generated-SOP - {short} - {demo}.txt")
        ok = call["status"] == "ok"
        out = {"uid": uid, "task": "sop", "demo": demo, "task_id": rank1[demo],
               "site": wb.task_site(rank1[demo]), "ablation": ablation, "status": call["status"],
               "raw_response": call.get("text", ""),
               "pred_sop": wb.generated_sop(sop_file) if ok else "",
               "gold_sop": wb.gold_sop(demo), "n_images": call["n_images"],
               "request_mb": call["request_mb"], "config": config, **meta}
        if call["status"] != "dry_run":
            jc.append_row(path, out)
            jc.note_outcome(call["status"], transport.model, MODEL_DIR)
        print(f"  [{n}/{len(units)}] {uid:46s} {call['status']:8s} images={call['n_images']} "
              f"{call['request_mb']}MB out={meta.get('completion_tokens')} "
              f"reasoning={meta.get('reasoning_tokens')} {meta.get('latency_s')}s "
              f"pred_lines={len(wb.sop_lines(out['pred_sop'])) if ok else '-'}", flush=True)
        if not transport.dry_run:
            time.sleep(1.0)
    return [f"{d}|{a}" for d, a in units]


def main():
    ap = argparse.ArgumentParser(description="Wonderbread — gpt-5.6-luna generation (judge pilot)")
    ap.add_argument("--task", required=True, choices=("qa", "sop"))
    ap.add_argument("--model", default=jc.GEN_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="first N items (qa) / demos (sop); smoke/")
    ap.add_argument("--dry-run", action="store_true", help="build every request; call nothing")
    args = ap.parse_args()

    problems = [p for d in wb.demos_needed() for p in wb.check_demo(d)]
    if problems:
        raise SystemExit("demos missing — run fetch_sample_demos.py first:\n  " + "\n  ".join(problems[:5]))
    load_dotenv(wb.ENV_PATH)
    jc.HARD_CALL_TIMEOUT = 300
    smoke = args.limit > 0 or args.dry_run
    # QA and SOP share this folder, and smoke shares it with the real run: each names its markers
    jc.MARKER_PREFIX = ("smoke_" if smoke else "") + f"{args.task}_"
    if args.dry_run:
        client, params, notes = None, dict(jc.GEN_WANTED), ["dry run: not negotiated"]
    else:
        jc.clear_stale_halt_markers(MODEL_DIR)
        client = jc.gen_client()
        print(f"negotiating parameter surface for {args.model} ...", flush=True)
        params, notes = jc.negotiate(client, args.model, jc.GEN_WANTED, MODEL_DIR, jc.GEN_REQUIRED)
        print(f"  accepted: {params}" + (f"  notes: {notes}" if notes else ""), flush=True)
    config = {"model": args.model, "params": params, "negotiated": notes, "task": args.task,
              "input": "upstream builders verbatim, SYSTEM_PROMPT prepended as _fetch_openai_completion",
              "set": "sample", "sample_seed": jc.SAMPLE_SEED, "sample_fraction": jc.SAMPLE_FRACTION}
    path = jc.out_path(MODEL_DIR, smoke, args.task, f"{args.model}.jsonl")
    if not args.dry_run:
        jc.lock_output(path)
    done = {} if args.dry_run else jc.load_done(path, config)
    transport = Transport(client, args.model, params, dry_run=args.dry_run)
    print(f"{args.task}: {'DRY RUN, nothing is called' if args.dry_run else path}", flush=True)
    if args.task == "qa":
        uids = run_qa(transport, path, config, done, args.limit)
    else:
        # upstream's run() writes SOP files; a dry run's placeholders must never land beside real ones
        upstream_dir = (tempfile.mkdtemp(prefix="wb_dry_run_") if args.dry_run else
                        os.path.join(MODEL_DIR, "smoke" if smoke else "results", "sop_upstream_files"))
        uids = run_sop(transport, path, config, done, args.limit, upstream_dir, args.model)
    if args.dry_run:
        return
    rows = [r for r in jc.read_rows(path) if r["uid"] in set(uids)]
    finish = collections.Counter(r.get("finish_reason") for r in rows)
    print(f"\nDONE {len(rows)}/{len(uids)} rows  status={jc.status_counts(rows)}  "
          f"finish={dict(finish)}  calls={jc.STATS}", flush=True)
    if len(rows) < len(uids) or any(r["status"] != "ok" for r in rows):
        sys.exit(1)
    if finish.get("length"):
        # exit 2, so a judge job waiting on `afterok` does not start on truncated answers
        print(f"STOP: {finish['length']} row(s) hit max_completion_tokens — the cap bound. Decide "
              f"whether to raise it (archive results/ first) before judging.", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
