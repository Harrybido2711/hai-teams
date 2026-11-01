"""MultiChallenge — one judge grades one model's stored responses.

Every judge runs this same file; `--judge` picks one of `judge_core.JUDGES` — `haiku-5.5` since the
user chose it on 2026-10-31 (`judge_core.JUDGE`). The prompt, the schema and the scorer are
upstream's, from `../mc_eval_core.py`, so the judge model is the only thing that changes.
`N_REPS` replicates per item, rep-major.

Two sets:
- `--set sample` (default) — the judge pilot: the 27 sampled items, gpt-5.6-luna's responses;
- `--set full` — all 273 items of one producer, named by its folder: `--producer MC_Qwen` reads
  the one `.jsonl` in `MC_Qwen/results/full/` (`smoke/full/` with `--limit`).

Output: `results/[full/]<judge>/<producer>.jsonl` — named after the model that produced it (the
judge) and holding its verdicts on one model's responses. One row per (item, replicate):
`status` is ok / parse_fail / refused / failed; a judge failure is never a verdict of NO and never
an axis of its own (LLM_as_judge/JUDGE_SELECTION.md §7.2, D2).

    python mc_judge.py --judge haiku-5.5 --set full --producer MC_Qwen --limit 3    # smoke
    python mc_judge.py --judge haiku-5.5 --set full --producer MC_Qwen              # all 273 x reps
    python mc_judge.py --judge haiku-4.5                                            # the pilot
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv

JUDGE_DIR = os.path.dirname(os.path.abspath(__file__))
MC_ROOT = os.path.dirname(JUDGE_DIR)
sys.path.insert(0, MC_ROOT)
import mc_eval_core as mc  # noqa: E402
jc = mc.jc

PILOT_PRODUCER = "MC_GPT_5.6_Luna"


def main():
    ap = argparse.ArgumentParser(description="MultiChallenge — one judge on one model's responses")
    ap.add_argument("--judge", default=jc.JUDGE, choices=sorted(jc.JUDGES))
    ap.add_argument("--set", dest="set_name", default="sample", choices=jc.SETS)
    ap.add_argument("--producer", default=PILOT_PRODUCER,
                    help="the generating model's folder, e.g. MC_Qwen (sample: the pilot's GPT only)")
    ap.add_argument("--responses", default=None, help="responses .jsonl; overrides --producer")
    ap.add_argument("--limit", type=int, default=0, help="first N items; writes smoke/")
    ap.add_argument("--reps", type=int, default=jc.N_REPS)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()
    smoke = args.limit > 0

    sub = os.path.join(MC_ROOT, args.producer, "smoke" if smoke else "results")
    if args.responses:
        responses_path = args.responses
    elif args.set_name == "full":
        responses_path = jc.responses_file(os.path.join(sub, "full"))
    else:
        responses_path = os.path.join(sub, f"{jc.GEN_MODEL}.jsonl")
    responses = {r["uid"]: r for r in jc.read_rows(responses_path)}
    stored_set = {r["config"].get("set") for r in responses.values()}
    if stored_set != {args.set_name}:
        raise SystemExit(f"{responses_path}: its rows were generated for set {sorted(map(str, stored_set))}, "
                         f"not '{args.set_name}' — refusing to judge them as that set.")
    items = mc.load_set(args.set_name)
    fallback = sum(1 for it in items if responses.get(it["QUESTION_ID"], {}).get("fallback"))
    if fallback:
        # Deepseek's answer read from reasoning_content (deepseek_mc_eval.answer) — judged as given
        print(f"NOTE: {fallback} response(s) are a reasoning_content fallback, judged as the answer",
              flush=True)
    if smoke:
        items = items[:args.limit]
    not_ready = [it["QUESTION_ID"] for it in items
                 if responses.get(it["QUESTION_ID"], {}).get("status") != "ok"]
    if not_ready:
        raise SystemExit(f"{responses_path}: {len(not_ready)} of {len(items)} items have no usable "
                         f"response (e.g. {not_ready[:3]}). Finish the generation run first — a "
                         f"missing response is never judged as a failure.")
    producer = os.path.splitext(os.path.basename(responses_path))[0]
    gen_config = next(iter(responses.values()))["config"]
    judged_sha = {f"{it['QUESTION_ID']}|rep{rep}": jc.text_sha(responses[it["QUESTION_ID"]]["response"])
                  for it in items for rep in range(args.reps)}

    load_dotenv(mc.ENV_PATH)
    parts = jc.set_parts(args.set_name)
    out_dir = os.path.join(JUDGE_DIR, "smoke" if smoke else "results", *parts, args.judge)
    jc.MARKER_PREFIX = f"{producer}_"
    jc.clear_stale_halt_markers(out_dir)
    path = jc.out_path(JUDGE_DIR, smoke, *parts, args.judge, f"{producer}.jsonl")
    jc.lock_output(path)
    config = dict(jc.judge_config(args.judge), n_reps=args.reps, prompt="upstream JUDGE_PROMPT",
                  response_format="JudgeResponse json_schema", scorer=mc.SCORER_VERSION,
                  responses=producer, responses_config=gen_config)
    done = jc.load_done(path, config)
    jc.check_judged(done, judged_sha, path)
    client = jc.judge_client()
    model = jc.JUDGES[args.judge]["model"]
    total = len(items) * args.reps
    print(f"{args.judge} ({model}) on {len(items)} items x {args.reps} reps = {total} calls, "
          f"{len(done)} already finished -> {path}", flush=True)

    for rep in range(args.reps):
        for item in items:
            qid = item["QUESTION_ID"]
            uid = f"{qid}|rep{rep}"
            if uid in done:
                continue
            response = responses[qid]["response"]
            once = jc.judge_once(client, args.judge, mc.judge_messages(response, item),
                                 mc.JUDGE_RESPONSE_FORMAT)
            text, verdict, meta, status = jc.call_and_parse(once, mc.parse_verdict, model, out_dir)
            passed = None if verdict is None else verdict == item["PASS_CRITERIA"]
            jc.append_row(path, {"uid": uid, "qid": qid, "axis": item["AXIS"], "rep": rep,
                                 "status": status, "verdict": verdict,
                                 "pass_criteria": item["PASS_CRITERIA"], "passed": passed,
                                 "judged_sha": judged_sha[uid], "raw_response": text,
                                 "config": config, **meta})
            jc.note_outcome(status, model, out_dir)
            print(f"  rep{rep} {qid} {item['AXIS']:24s} {status:10s} verdict={verdict} "
                  f"{meta.get('latency_s')}s cost={meta.get('cost')}", flush=True)
            time.sleep(args.sleep)

    rows = [r for r in jc.read_rows(path) if r["qid"] in {it["QUESTION_ID"] for it in items}]
    print(f"\nDONE {len(rows)}/{total} rows  status={jc.status_counts(rows)}  calls={jc.STATS}")
    for rep in range(args.reps):
        s = mc.score({r["qid"]: r["passed"] for r in rows if r["rep"] == rep})
        print(f"  rep{rep}: macro={s['overall_macro']}  micro={s['overall_micro']}  "
              f"unjudged={s['n_unjudged']}  axes={s['axis_scores']}")
    by_item = {}
    for r in rows:
        by_item.setdefault(r["qid"], []).append(r["passed"])
    s = mc.score({q: mc.majority(v) for q, v in by_item.items()})
    print(f"  majority of {args.reps}: macro={s['overall_macro']}  micro={s['overall_micro']}  "
          f"unjudged={s['n_unjudged']}  axes_missing={s['axes_missing']}")
    sys.exit(jc.judge_exit_code(rows, total))


if __name__ == "__main__":
    main()
