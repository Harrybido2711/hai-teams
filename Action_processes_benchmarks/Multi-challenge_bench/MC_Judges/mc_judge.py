"""MultiChallenge — one candidate judge grades stored responses, for the judge pilot.

Every judge runs this same file; `--judge` picks one of `judge_core.JUDGES`. The prompt, the schema
and the scorer are upstream's, from `../mc_eval_core.py`, so the judge model is the only thing that
changes between the four. `N_REPS` replicates per item, rep-major.

Output: `results/<judge>/<responses-producer>.jsonl` — named after the model that produced it (the
judge) and holding its verdicts on one model's responses. One row per (item, replicate):
`status` is ok / parse_fail / refused / failed; a judge failure is never a verdict of NO and never
an axis of its own (LLM_as_judge/JUDGE_SELECTION.md §7.2, D2).

    python mc_judge.py --judge haiku-4.5 --limit 3     # smoke: first 3 items of the smoke responses
    python mc_judge.py --judge haiku-4.5               # the sample, all replicates
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

DEFAULT_PRODUCER = os.path.join(MC_ROOT, "MC_GPT_5.6_Luna")


def main():
    ap = argparse.ArgumentParser(description="MultiChallenge — candidate judge (judge pilot)")
    ap.add_argument("--judge", required=True, choices=sorted(jc.JUDGES))
    ap.add_argument("--responses", default=None,
                    help="responses .jsonl; default: MC_GPT_5.6_Luna/{results|smoke}/gpt-5.6-luna.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="first N items; writes smoke/")
    ap.add_argument("--reps", type=int, default=jc.N_REPS)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()
    smoke = args.limit > 0

    responses_path = args.responses or os.path.join(
        DEFAULT_PRODUCER, "smoke" if smoke else "results", f"{jc.GEN_MODEL}.jsonl")
    responses = {r["uid"]: r for r in jc.read_rows(responses_path)}
    items = mc.load_sample()
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
    out_dir = os.path.join(JUDGE_DIR, "smoke" if smoke else "results", args.judge)
    jc.MARKER_PREFIX = f"{producer}_"
    jc.clear_stale_halt_markers(out_dir)
    path = jc.out_path(JUDGE_DIR, smoke, args.judge, f"{producer}.jsonl")
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
