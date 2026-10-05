# NegotiationToM — benchmark card

<!-- size-budget: 7500 -->
<!-- One job — the operational card. Over the 5 KB nudge since 2026-10-05, when the results table
     arrived: eight runs in two generations plus a Gemma caveat that changes how one cell is read. -->

Conflict management, under interpersonal processes. Upstream `HKUST-KnowComp/NegotiationToM`. No LLM
judge — scored by exact match and micro/macro F1. Six providers have results. This is the benchmark
the generic references were originally written around, so anything here that reads like a universal
truth is not one.

## Paths

| | Path |
|---|---|
| Local | `Interpersonal_processes_benchmarks/NegotiationToM` |
| Quest | `/gpfs/projects/p32983/Interpersonal_processes_benchmarks/NegotiationToM` |

**Quest was moved to match local on 2026-08-23**; before that it was flat at
`/gpfs/projects/p32983/NegotiationToM`. Anything still naming the old path is stale — that mismatch
is what blinded the pre-submit gate, whose contract is in [../quest-cluster.md](../../quest-cluster.md).

## Layout

```
NegotiationToM/
├── NEG_{GPT,Gemini,XAI,Qwen,Gemma,Deepseek}/
│   ├── <provider>_neg_eval.py     thin runner: supplies only its own call_api
│   ├── run_negotiation.sh         sbatch, full run
│   ├── run_pilot.sh               sbatch, 10% of the data
│   └── results/
├── neg_eval_core.py               the shared core every runner imports
├── preflight.py                   the one sanctioned route to a provider API
├── merge_neg_results.py
└── NegotiationToM.json            the data
```

**Sync the core and the runners together, or not at all.** A runner transferred without
`neg_eval_core.py` dies at import; a core transferred without the runners breaks whichever one used a
signature that changed. Check the whole set, not the file you edited: on 2026-07-29 a check found 6
of 32 files stale when only Qwen was suspected.

The concrete sync check for this benchmark — the generic principle and its two silent failure modes
are in [../quest-cluster.md](../../quest-cluster.md):

```bash
cd Interpersonal_processes_benchmarks/NegotiationToM
setopt null_glob
FILES=(*.py NEG_*/*.py NEG_*/*.sh); FILES=(${(u)FILES})
md5 -r "${FILES[@]}" | awk '{print $2, $1}' | sort -k1,1 > /tmp/l.md5
ssh quest "cd /gpfs/projects/p32983/Interpersonal_processes_benchmarks/NegotiationToM && md5sum ${FILES[*]}" \
  | awk 'NF==2{print $2, $1}' | sort -k1,1 > /tmp/q.md5
join -j1 -o 0,1.2,2.2 /tmp/l.md5 /tmp/q.md5 | awk '$2!=$3{print "DIFFER  " $1}'
join -v1 -j1 /tmp/l.md5 /tmp/q.md5      | awk '{print "MISSING ON QUEST  " $1}'
```

## Expected counts

A full run is **14,138 rows**: desire 4,760 + belief 4,760 + intention **4,618**.

- **intention at 4,760 means a known bug has returned** — odd-length dialogues annotate one target
  utterance, not two.
- `scored_rows` is **4,604** for desire and belief: 156 rows per task are excluded because their gold
  is the sentinel `"None"`, which marks an unannotated sample rather than "wants nothing".
- `All_EM` in the low single-digit percents is **expected**, not a defect: it ANDs all 5–6 rows of a
  dialogue, and intention gets no partial credit there even though its F1 does.

These are NegotiationToM's counts only. The workflows read each benchmark's counts from its own
page; never pass these for another one.

## Results — six runs complete, verified 2026-10-05

Recomputed from the per-shard `.jsonl` and equal to every `NEG_*/results/<slug>_negotiation_overall.csv`.
Every run: 14,138 rows, **0 empty responses**, 0 duplicate uids.

| Slot | Folder · model | Desire | Belief | Micro F1 | Macro F1 | All_EM | In `Final_result` |
|---|---|---|---|---|---|---|---|
| Deepseek | `NEG_Deepseek` · `deepseek-reasoner` | 0.6234 | **0.6197** | 0.4980 | 0.4742 | 0.0534 | yes |
| Gemma | `NEG_Gemma` · `gemma-4-31B-it`, reasoning off | **0.6494** | 0.5007 | 0.5345 | 0.5064 | **0.0773** | yes — belief caveat |
| Qwen | `NEG_Qwen` · `Qwen3.5-9B`, reasoning off | 0.5888 | 0.5374 | 0.4860 | 0.3992 | 0.0408 | yes |
| XAI | `NEG_XAI` · `grok-3-mini` | 0.5393 | 0.5510 | **0.5382** | **0.5366** | 0.0702 | no — XAI out since 2026-10-05 |
| — | `NEG_Gemini` · `gemini-2.5-flash` | 0.5315 | 0.5593 | 0.5369 | 0.5133 | 0.0647 | no — superseded |
| — | `NEG_GPT` · `gpt-4o-mini` | 0.5046 | 0.4522 | 0.4194 | 0.3376 | 0.0034 | no — superseded |

**The current Gemini and OpenAI slots have not run.** `NEG_Gemini_Flash3.5lite_OpenRouter` and
`NEG_GPT_5.6_Luna` are written but unpiloted, so those two `Final_result.xlsx` cells are blank.

**Gemma's belief is a thinking-off number.** Paired on 463 uids against an archived reasoning-on
pilot, belief was 0.6177 on and 0.4903 off (McNemar p=1.2e-08). Off is Gemma's settled config on
every sheet, so the user reported it as is on 2026-10-05; the caveat is on the `Provenance` sheet
and in `ISSUES.md`. Qwen is off too and has not been measured for the same effect.

Per-metric cells and their sources live in the two workbooks' `NegotiationToM` sheets.

## Output, logs and markers

```
results/pilot/<task>/*.jsonl                pilot
results/<task>/<model>_shard{N}of{M}.jsonl  full run — the shard tag is mandatory
log_pilot.txt · log_shard%a.txt · .err
NEG_*/{BILLING,QUOTA,FAILURE}_HALT.txt      check these before grepping any log
```

Tasks are `desire`, `belief`, `intention`.

## Run order

`preflight.py` → `sbatch run_pilot.sh` → review the pilot → `sbatch run_negotiation.sh` →
`bash run_merge.sh` if sharded.

## Its own traps

- **Verify from the `.jsonl`, not the `.csv`.** Reasoning models emit newlines inside `raw_response`,
  so `cut -d, -f1` on the CSV mis-parses and can report *more* unique uids than rows. That false
  alarm looked model-specific and nearly triggered a needless re-run.
- **A finished job proves nothing about usable data.** Grok's five shards all exited `COMPLETED 0:0`
  with a perfect 14,138 rows while belief and intention were 100% empty — its credits had run out.
  Report the non-empty `raw_response` rate and the null-`pred` count, not row counts.
- **Qwen needs `reasoning={"enabled": False}`.** A pilot burned 3h10m for 315 rows and 105 empty
  responses because that fix never left the laptop.

## Where the rest lives

`negotiation.md` (current results, the silent-failure catalogue, reasoning-token cost) ·
`DATA_NOTES.md` (cutoff tiling, the sentinel, expected counts) · `ISSUES.md` (what broke, what was
rejected, what shipped, the false alarms) · `Negotiation_script.md` (what each task *means*; its file
listings go stale — verify them against the tree).
