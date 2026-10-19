"""GPT-5.6-Luna — MultiChallenge response generation for the judge pilot. **The project's OpenAI slot.**

Generates one response per sampled item (`../sample_uids.json`, 27 of 273, stratified by axis) —
upstream's `--attempts 1`. The tested model gets the item's CONVERSATION verbatim, as upstream's
`OpenAIModel.generate` sends it: no system prompt, roles as stored, the last turn the user's.
Judging is `../MC_Judges/mc_judge.py`; nothing here scores.

**Settled config** (`model-parameters.md` § Settled — the project's GPT): `reasoning_effort="low"`,
`seed=42`, `temperature` not settable (upstream ran its models at temp=0 — a declared deviation).
`max_completion_tokens=8192` (`judge_core.GEN_WANTED`): MultiChallenge answers are free-form,
sometimes long rewrites; upstream's OpenAI path set no cap at all. `finish_reason=length` on any row
means the cap bound: the run then exits 2, so the judges' `afterok` job does not start on it.
**Visible reasoning:** the README gives no instruction and the conversation is sent as stored, so
nothing is added to ask for or suppress it.

    python gpt56luna_mc_eval.py --limit 3     # smoke, writes smoke/
    python gpt56luna_mc_eval.py               # the 27-item sample, writes results/
"""

import argparse
import collections
import os
import sys
import time

from dotenv import load_dotenv

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import mc_eval_core as mc  # noqa: E402
jc = mc.jc


def main():
    ap = argparse.ArgumentParser(description="MultiChallenge — gpt-5.6-luna responses (judge pilot)")
    ap.add_argument("--model", default=jc.GEN_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="first N sampled items; writes smoke/")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    load_dotenv(mc.ENV_PATH)
    jc.HARD_CALL_TIMEOUT = 300
    smoke = args.limit > 0
    jc.MARKER_PREFIX = "smoke_" if smoke else ""
    jc.clear_stale_halt_markers(MODEL_DIR)
    client = jc.gen_client()
    print(f"negotiating parameter surface for {args.model} ...", flush=True)
    params, notes = jc.negotiate(client, args.model, jc.GEN_WANTED, MODEL_DIR, jc.GEN_REQUIRED)
    print(f"  accepted: {params}" + (f"  notes: {notes}" if notes else ""), flush=True)

    config = {"model": args.model, "params": params, "negotiated": notes,
              "input": "upstream CONVERSATION verbatim, no system prompt",
              "set": "sample", "sample_seed": jc.SAMPLE_SEED, "sample_fraction": jc.SAMPLE_FRACTION}
    path = jc.out_path(MODEL_DIR, smoke, f"{args.model}.jsonl")
    jc.lock_output(path)
    done = jc.load_done(path, config)
    items = mc.load_sample()
    if smoke:
        items = items[:args.limit]
    todo = [it for it in items if it["QUESTION_ID"] not in done]
    print(f"{len(items)} items, {len(todo)} to call -> {path}", flush=True)

    for n, item in enumerate(todo, 1):
        once = jc.gen_once(client, args.model, item["CONVERSATION"], params)
        text, meta, status = jc.call_with_retries(once, args.model, MODEL_DIR)
        jc.append_row(path, {"uid": item["QUESTION_ID"], "axis": item["AXIS"], "status": status,
                             "response": text, "config": config, **meta})
        jc.note_outcome(status, args.model, MODEL_DIR)
        print(f"  [{n}/{len(todo)}] {item['QUESTION_ID']} {item['AXIS']:24s} {status} "
              f"finish={meta.get('finish_reason')} out={meta.get('completion_tokens')} "
              f"reasoning={meta.get('reasoning_tokens')} {meta.get('latency_s')}s", flush=True)
        time.sleep(args.sleep)

    rows = [r for r in jc.read_rows(path) if r["uid"] in {it["QUESTION_ID"] for it in items}]
    finish = collections.Counter(r.get("finish_reason") for r in rows)
    print(f"\nDONE {len(rows)}/{len(items)} rows  status={jc.status_counts(rows)}  finish={dict(finish)}"
          f"  calls={jc.STATS}", flush=True)
    if len(rows) < len(items) or any(r["status"] != "ok" for r in rows):
        sys.exit(1)
    if finish.get("length"):
        # exit 2, so a judge job waiting on `afterok` does not start on truncated answers
        print(f"STOP: {finish['length']} row(s) hit max_completion_tokens — the cap bound. Decide "
              f"whether to raise it (archive results/ first) before judging.", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
