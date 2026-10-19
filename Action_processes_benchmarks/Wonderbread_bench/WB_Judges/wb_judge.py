"""Wonderbread — one candidate judge grades QA answers or generated SOPs, for the judge pilot.

Every judge runs this same file; `--judge` picks one of `judge_core.JUDGES`. Prompts and scorers
are upstream's, through `../wb_eval_core.py`, so the judge model is the only thing that changes.
`N_REPS` replicates of every call, rep-major.

Two sets:
- `--set sample` — gpt-5.6-luna's answers / SOPs (`../WB_GPT_5.6_Luna/results/`), the pilot itself;
- `--set anchor` — upstream's 30-item human-labelled samples (QA: human 1-3 scores; SOP: human
  precision / recall), which need no generation. Each judge's agreement with the HUMANS there is
  what says which judge is better, not merely which judges agree (JUDGE_SELECTION.md §6.2).

`--task qa`: four calls per answer, one per criterion (completeness and soundness see the Human
Label; clarity and compactness do not), score 1-3 with **1 = best**. An answer shorter than 5
characters is not judged — status `skipped`, upstream's "NA".
`--task sop`: one call per line of each SOP, in both directions (generated → gold for precision
and ordering, gold → generated for recall); the judge returns the matching line's index or -1.

Output: `results/<judge>/<task>/<producer>.jsonl`, producer = gpt-5.6-luna or anchor-human-sample.

    python wb_judge.py --judge haiku-4.5 --task qa --set anchor --limit 2    # smoke, writes smoke/
    python wb_judge.py --judge haiku-4.5 --task sop --set sample
"""

import argparse
import os
import statistics
import sys
import time

from dotenv import load_dotenv

JUDGE_DIR = os.path.dirname(os.path.abspath(__file__))
WB_ROOT = os.path.dirname(JUDGE_DIR)
sys.path.insert(0, WB_ROOT)
import wb_eval_core as wb  # noqa: E402
jc = wb.jc

PRODUCER_DIR = os.path.join(WB_ROOT, "WB_GPT_5.6_Luna")


def qa_units(set_name, smoke, limit):
    """(uid, question, human_label, response) per answer."""
    if set_name == "anchor":
        anchor = wb.load_anchor_qa()
        units = [(u, a["question"], a["human_label"], a["response"]) for u, a in anchor.items()]
        return units[:limit or None], wb.ANCHOR_PRODUCER, None
    path = os.path.join(PRODUCER_DIR, "smoke" if smoke else "results", "qa", f"{jc.GEN_MODEL}.jsonl")
    rows = {r["uid"]: r for r in jc.read_rows(path)}
    uids = wb.load_sample("qa")[:limit or None]
    missing = [u for u in uids if rows.get(u, {}).get("status") != "ok"]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} of {len(uids)} answers missing or failed "
                         f"(e.g. {missing[:3]}). Finish the generation run first.")
    units = [(u, rows[u]["question"], rows[u]["human_label"], rows[u]["response"]) for u in uids]
    return units, jc.GEN_MODEL, rows[uids[0]]["config"]


def sop_units(set_name, smoke, limit):
    """(pair uid, pred_sop, gold_sop) per generated SOP."""
    if set_name == "anchor":
        anchor = wb.load_anchor_sop()
        units = [(u, a["pred_sop"], a["gold_sop"]) for u, a in anchor.items()]
        return units[:limit or None], wb.ANCHOR_PRODUCER, None
    path = os.path.join(PRODUCER_DIR, "smoke" if smoke else "results", "sop", f"{jc.GEN_MODEL}.jsonl")
    rows = {r["uid"]: r for r in jc.read_rows(path)}
    demos = wb.load_sample("sop")[:limit or None]
    uids = [f"{d}|{a}" for d in demos for a in wb.SOP_ABLATIONS]
    missing = [u for u in uids if rows.get(u, {}).get("status") != "ok"]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} of {len(uids)} SOPs missing or failed "
                         f"(e.g. {missing[:3]}). Finish the generation run first.")
    return [(u, rows[u]["pred_sop"], rows[u]["gold_sop"]) for u in uids], jc.GEN_MODEL, \
        rows[uids[0]]["config"]


def main():
    ap = argparse.ArgumentParser(description="Wonderbread — candidate judge (judge pilot)")
    ap.add_argument("--judge", required=True, choices=sorted(jc.JUDGES))
    ap.add_argument("--task", required=True, choices=("qa", "sop"))
    ap.add_argument("--set", dest="set_name", default="sample", choices=("sample", "anchor"))
    ap.add_argument("--limit", type=int, default=0, help="first N answers / SOPs; writes smoke/")
    ap.add_argument("--reps", type=int, default=jc.N_REPS)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()
    smoke = args.limit > 0

    if args.task == "qa":
        units, producer, gen_config = qa_units(args.set_name, smoke, args.limit)
    else:
        units, producer, gen_config = sop_units(args.set_name, smoke, args.limit)

    load_dotenv(wb.ENV_PATH)
    out_dir = os.path.join(JUDGE_DIR, "smoke" if smoke else "results", args.judge, args.task)
    jc.MARKER_PREFIX = f"{producer}_"       # the anchor and sample stages share this folder
    jc.clear_stale_halt_markers(out_dir)
    path = jc.out_path(JUDGE_DIR, smoke, args.judge, args.task, f"{producer}.jsonl")
    jc.lock_output(path)
    config = dict(jc.judge_config(args.judge), n_reps=args.reps, task=args.task,
                  set=args.set_name, scorer=wb.SCORER_VERSION, responses=producer,
                  responses_config=gen_config,
                  prompt=("upstream prompt__<criterion>_score" if args.task == "qa"
                          else "upstream map_query_to_one_prompt, json_object"))
    done = jc.load_done(path, config)
    client = jc.judge_client()
    model = jc.JUDGES[args.judge]["model"]
    # what each row judged, fingerprinted: checked on resume here and by compare_judges.py
    unit_sha = {u[0]: jc.text_sha(*u[1:]) for u in units}

    if args.task == "qa":
        calls = [(rep, u, c) for rep in range(args.reps) for u in units for c in wb.QA_CRITERIA]
        jc.check_judged(done, {f"{u[0]}|{c}|rep{rep}": unit_sha[u[0]] for rep, u, c in calls}, path)
        print(f"{args.judge} ({model}) QA/{args.set_name}: {len(units)} answers x 4 criteria x "
              f"{args.reps} reps = {len(calls)} calls, {len(done)} finished -> {path}", flush=True)
        for rep, (uid, question, label, response), criterion in calls:
            key = f"{uid}|{criterion}|rep{rep}"
            if key in done:
                continue
            base = {"uid": key, "item": uid, "criterion": criterion, "rep": rep,
                    "judged_sha": unit_sha[uid], "config": config}
            if wb.qa_skip(response):
                jc.append_row(path, dict(base, status="skipped", score=None, raw_response=""))
                continue
            once = jc.judge_once(client, args.judge,
                                 wb.qa_judge_messages(criterion, question, label, response))
            text, score, meta, status = jc.call_and_parse(once, wb.parse_qa_score, model, out_dir)
            jc.append_row(path, dict(base, status=status, score=score, raw_response=text, **meta))
            jc.note_outcome(status, model, out_dir)
            print(f"  rep{rep} {uid} {criterion:12s} {status:10s} score={score} "
                  f"{meta.get('latency_s')}s cost={meta.get('cost')}", flush=True)
            time.sleep(args.sleep)
    else:
        lines = {u: (wb.sop_lines(pred), wb.sop_lines(gold)) for u, pred, gold in units}
        calls = []
        for rep in range(args.reps):
            for uid, _, _ in units:
                pred, gold = lines[uid]
                calls += [(rep, uid, "pred", i, q, gold) for i, q in enumerate(pred)]
                calls += [(rep, uid, "gold", j, q, pred) for j, q in enumerate(gold)]
        jc.check_judged(done, {f"{uid}|{d}{i}|rep{rep}": unit_sha[uid]
                               for rep, uid, d, i, _, _ in calls}, path)
        empty = [u for u, (p, g) in lines.items() if not p or not g]
        if empty:
            print(f"WARNING: {len(empty)} SOP pair(s) with no lines after preprocess_sop, not "
                  f"judgeable: {empty[:3]}", flush=True)
        print(f"{args.judge} ({model}) SOP/{args.set_name}: {len(units)} SOPs, {len(calls)} line "
              f"calls over {args.reps} reps, {len(done)} finished -> {path}", flush=True)
        for rep, uid, direction, i, query, other in calls:
            key = f"{uid}|{direction}{i}|rep{rep}"
            if key in done:
                continue
            base = {"uid": key, "pair": uid, "direction": direction, "line": i, "rep": rep,
                    "n_list": len(other), "judged_sha": unit_sha[uid], "config": config}
            if not other:
                jc.append_row(path, dict(base, status="skipped", index=None, matched=None,
                                         raw_response=""))
                continue
            once = jc.judge_once(client, args.judge, wb.sop_judge_messages(query, other),
                                 wb.SOP_RESPONSE_FORMAT)
            text, index, meta, status = jc.call_and_parse(once, wb.make_index_parser(len(other)),
                                                          model, out_dir)
            jc.append_row(path, dict(base, status=status, index=index,
                                     matched=None if index is None else index != -1,
                                     raw_response=text, **meta))
            jc.note_outcome(status, model, out_dir)
            print(f"  rep{rep} {uid[:40]:40s} {direction}{i:<3d} {status:10s} index={index} "
                  f"{meta.get('latency_s')}s", flush=True)
            time.sleep(args.sleep)

    rows = jc.read_rows(path)
    wanted = {f"{u[0]}|" for u in units}
    rows = [r for r in rows if any(r["uid"].startswith(w) for w in wanted)]
    print(f"\nDONE {len(rows)}/{len(calls)} rows  status={jc.status_counts(rows)}  calls={jc.STATS}")
    if args.task == "qa":
        for c in wb.QA_CRITERIA:
            scores = [r["score"] for r in rows if r["criterion"] == c and r["score"] is not None]
            if scores:
                print(f"  {c:12s} mean={statistics.mean(scores):.3f} (1 = best)  n={len(scores)}")
    else:
        for rep in range(args.reps):
            per_pair = []
            for uid, _, _ in units:
                pred, gold = lines[uid]
                got = {(r["direction"], r["line"]): r["index"] for r in rows
                       if r["pair"] == uid and r["rep"] == rep}
                m = wb.sop_metrics([got.get(("pred", i)) for i in range(len(pred))],
                                   [got.get(("gold", j)) for j in range(len(gold))])
                if m:
                    per_pair.append(m)
            if per_pair:
                print(f"  rep{rep}: " + "  ".join(
                    f"{k}={statistics.mean(m[k] for m in per_pair if m[k] is not None):.3f}"
                    for k in ("precision", "recall", "f1", "ordering")) + f"  pairs={len(per_pair)}")
    sys.exit(jc.judge_exit_code(rows, len(calls)))


if __name__ == "__main__":
    main()
