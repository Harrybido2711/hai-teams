# OpenAI DocVQA Evaluation — Issue Log & Fix Notes

## Problem: ~3021 out of 5349 questions have empty responses

### Root Cause

Five shards run in parallel and share the same API key's daily RPD quota (10,000 requests/day).
The old retry logic wasted quota on every failed request:

```
Question 1: success → valid        (costs 1 RPD)
Question 2: fail → retry → retry → retry → empty  (wastes 3 RPD)
Question 3: fail → retry × 3      → empty  (wastes 3 RPD)
...
```

This produced a very regular pattern visible in the CSV data:

```
VALID  × 1
EMPTY  × 6   ← 6 questions failed because quota was burned by retries
VALID  × 1
EMPTY  × 6
...  (repeated ~210 times per shard)
```

Net result: for every 1 question answered, ~6 questions were skipped due to exhausted RPD —
consuming roughly 90 RPD per cycle instead of 7.

### Data State (after the failed runs)

| Shard | Total rows | Valid responses | Empty responses |
|-------|-----------|----------------|----------------|
| shard0 | 1070 | 456 | 614 |
| shard1 | 1070 | 478 | 592 |
| shard2 | 1070 | 459 | 611 |
| shard3 | 1070 | 479 | 591 |
| shard4 | 1069 | 456 | 613 |
| **Total** | **5349** | **2328** | **3021** |

No duplicate questionIds, no cross-shard contamination — only the empty responses need to be retried.

---

## Fix Applied

### 1. Stop retrying on RPD errors (`openai_eval.py`)

Old behavior: retry 3 times on any API error, including RPD exhaustion.

New behavior: return `None` immediately when RPD is hit — no wasted quota.

```python
if 'requests per day' in err:
    print("RPD limit exhausted — stopping to preserve quota for resume.")
    return None  # do not retry
```

### 2. Resume logic — skip already-answered questions

On startup, `openai_eval.py` reads the existing CSV and builds `done_ids` from rows that have
a non-empty `model_response`. Only unanswered questions are sent to the API.

```python
if os.path.exists(out_csv):
    existing = pd.read_csv(out_csv)
    existing = existing[existing["model_response"].notna() &
                        (existing["model_response"].astype(str).str.strip() != "")]
    done_ids = set(str(x) for x in existing["questionId"].tolist())
    results = existing.to_dict("records")
```

### 3. Reduced parallel shards (10 → 5)

Running 10 shards simultaneously was pushing combined TPM (tokens/min) over the Tier 1 limit.
Reduced to 5 shards to stay within limits while still parallelizing.

### 4. Added `timeout=30` to API calls

The OpenAI SDK default timeout is 600 seconds. Without a timeout, a hanging request would
block a shard for up to 10 minutes. Now each call times out after 30 seconds.

---

## How to Resume the Run

1. **Wait for RPD reset** — resets at 00:00 UTC (08:00 Beijing time) every day.

2. **Upload the fixed files to Quest:**
   ```bash
   scp openai_eval.py openai_eval_submit_array.sh openai_eval_run_merge.sh merge_openai_results.py \
       <netid>@quest.northwestern.edu:/projects/p32983/.../DocVQA/
   ```

3. **Submit the array job** — resume logic handles the rest automatically:
   ```bash
   sbatch openai_eval_submit_array.sh
   ```
   Each shard will skip its ~456 already-answered questions and only call the API for the
   remaining ~604 empty ones (~3021 total across all shards).

4. **After all shards finish, merge results:**
   ```bash
   sbatch openai_eval_run_merge.sh
   ```
   This produces `openai_docvqa_results.csv` and `openai_docvqa_overall.csv`.

---

## Cost Estimate

| Item | Value |
|------|-------|
| Remaining questions | ~3,021 |
| Avg input tokens/question | ~850 (image + prompt) |
| Avg output tokens/question | ~200 |
| Input cost (gpt-4o-mini) | $0.15 / 1M tokens |
| Output cost (gpt-4o-mini) | $0.60 / 1M tokens |
| **Estimated cost** | **~$0.60** |

Full run from scratch (5,349 questions) costs ~$1.05.
