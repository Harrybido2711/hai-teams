# DocVQA — benchmark card

Document visual question answering. Upstream docvqa.org. Scored by **ANLS**, no LLM judge.

**Four of the six have results, and four is the ceiling.** `gemini_eval.py`, `openai_eval.py`,
`qwen_DocVQA/` and `gemma_DocVQA/` — exactly the four models that accept an image
([model-calls.md](../../model-calls.md)). **XAI and Deepseek are blank here because they are
text-only, not because the run is outstanding.** Nothing is waiting to be scheduled for them.

## Paths

| | Path |
|---|---|
| Local | `Tasks_benchmarks/DocVQA` |
| Quest | `/gpfs/projects/p32983/Tasks_benchmarks/DocVQA` |

## Layout

```
DocVQA/
├── openai_eval.py · openai_eval_submit_array.sh · openai_eval_run_merge.sh
├── merge_openai_results.py · cleanup_shards.py       sharding support
├── gemini_eval.py · gemini_eval_script.sh
├── qwen_DocVQA/qwen_eval.py · gemma_DocVQA/gemma_eval_half{1,2}.py
├── docvqa_output/docvqa_validation.json · images/    the data — 3.5 GB, mostly page images
├── openai_partial_results/                           shard state from the interrupted run
├── OpenAI_tesing/                                    retry probes (sic — the folder is misspelled)
└── OPENAI_EVAL_NOTES.md
```

## Expected counts

**5,349 validation questions.** The interrupted run left ~3,021 of them with empty responses, so a
result set here is only meaningful alongside its empty count.

## Its own traps — all four from one incident

Recorded in `OPENAI_EVAL_NOTES.md`, which is authoritative on this benchmark:

1. **Shards share one API key's daily quota.** Five parallel shards drew on the same 10,000 RPD cap
   and exhausted it; each then logged the same refusal ~210 times. Shard count was cut 10 → 5.
2. **An RPD refusal must not be retried.** Retrying a daily-cap error burns the retry budget against
   a wall that only clears at midnight.
3. **Resume skips already-answered questions**, so a partial shard is worth keeping — but only if the
   empty rows are pruned first, or resume treats them as answered.
4. **`timeout=30` was added to the API calls** after hanging requests; the notes' §"Problem 3" ties
   the hang directly to the shard design.

The notes also carry a cost estimate and a written resume procedure — read them before restarting,
not after.
