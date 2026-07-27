# NegotiationToM Benchmark Notes

## Dataset Overview

- Total samples: 2,380
- Unique dialogues: 396 (each dialogue sampled at multiple turn cutoff points)
- Venue: EMNLP 2024 — [NegotiationToM paper](https://arxiv.org/abs/2404.13627)
- Setting: agent_1 and agent_2 negotiate over three item types: Food, Water, Firewood

---

## Three Sub-Tasks

### 1. Desire Prediction
Predict an agent's preference priority (High / Medium / Low) over the three item types.

- Label set: `Food`, `Water`, `Firewood`, `Not Given`, `None`
- **Metric: Exact Match** — all three slots (High / Medium / Low) must be correct simultaneously; any single error yields 0

### 2. Belief Prediction
Predict what an agent *believes* the other agent's preferences are (i.e., the agent's subjective model of the opponent).

- Label set: same as above
- **Metric: Exact Match** — same strict three-slot rule
- Key challenge: an agent's belief may differ from the opponent's true desire; the model must infer the subjective perspective, not the ground truth

### 3. Intention Classification
Multi-label classification of the intent behind a specified utterance.

- Base intent labels (9): `Build-Rapport`, `Callout-Fairness`, `Describe-Need`, `Discover-Preference`, `No-Intention`, `No-Need`, `Promote-Coordination`, `Show-Empathy`, `Undermine-Requirements`
- A single utterance may carry multiple intents (e.g., `Build-Rapport,Describe-Need,Promote-Coordination`)
- **Metric: Micro F1 + Macro F1** — partial credit for partially correct label sets

---

## Composite Metrics

| Metric | Description |
|--------|-------------|
| **All (Exact Match)** | A sample is correct only if Desire, Belief, and Intention are all correct simultaneously (AND logic, not average) |
| **Consistency** | Whether the model's mental state predictions remain coherent across multiple turn cutoffs of the same dialogue (reported separately for Desire and Belief) |

---

## Model Performance Results (Table 4)

| Model | Desire EM (%) | Belief EM (%) | Intent Micro F1 (%) | Intent Macro F1 (%) | All EM (%) | Consistency Desire (%) | Consistency Belief (%) |
|-------|--------------|--------------|--------------------|--------------------|-----------|----------------------|----------------------|

---

## Script Design

### Files

| File | Purpose |
|------|---------|
| `NEG_GPT/openai_neg_eval.py` | Full evaluation — all 2,380 samples, all 3 tasks, with sharding |
| `NEG_GPT/openai_neg_pilot.py` | Pilot — 50 random samples, prints cost/time projection |
| `NEG_GPT/merge_shards.py` | Post-run — merges shard outputs and computes final scores |
| `NEG_GPT/run_negotiation.sh` | SLURM job: `--array=0-4`, 5 shards × all tasks |
| `NEG_GPT/run_pilot.sh` | SLURM job: single job, 50 samples |
| `NEG_GPT/run_merge.sh` | Runs `merge_shards.py` after all array jobs finish |

Model: `gpt-4o-mini` via `openai.OpenAI`. API key loaded from `.env`.

---

### Data Loading

- Password-protected zip: extract with `unzip -P "NegotiationToM" NegotiationToM.zip`
- Scripts expect the extracted `NegotiationToM.json` (2,380 samples)
- `dialogue_id` format: `"<dialogue>-<turn_cutoff>"` (e.g. `"0-5"` = dialogue 0, 5 turns visible)
- Each sample carries ground-truth desire, belief, and intent labels for both agents

---

### Three Sub-Tasks and Prompts

#### 1. Desire (`--task desire`)

2 API calls per sample (one per agent) → **4,760 total calls**

**System prompt:** Assigns exactly one of High / Medium / Low to each item. JSON format: `{"high": "<item>", "medium": "<item>", "low": "<item>"}`.

**User prompt:** Full dialogue + ask for `agent_X`'s priorities over Food, Water, Firewood.

**Scoring — Exact Match:** All three slots must match simultaneously. Any single wrong slot → score 0.

#### 2. Belief (`--task belief`)

2 API calls per sample (agent_1's belief about agent_2, and vice versa) → **4,760 total calls**

**System prompt:** Same structure as Desire but `"Not Given"` is valid when a belief cannot be inferred from the dialogue.

**User prompt:** Full dialogue + ask what `agent_X` believes about `agent_Y`'s priorities.

**Scoring — Exact Match:** Same strict three-slot rule. `"Not Given"` must match the gold label exactly.

#### 3. Intention (`--task intention`)

2 API calls per sample (second-to-last utterance + last utterance) → **up to 4,760 total calls**

**System prompt:** Select one or more intents from the 9-label set. JSON format: `{"intents": ["label1", "label2"]}`.

**User prompt:** Full dialogue + target utterance highlighted separately.

**Scoring — Micro F1 + Macro F1:** Labels binarized against the 9-label set; scored with `sklearn.metrics.f1_score`. Partial credit for partially correct sets.

**9 intent labels:** `Build-Rapport`, `Callout-Fairness`, `Describe-Need`, `Discover-Preference`, `No-Intention`, `No-Need`, `Promote-Coordination`, `Show-Empathy`, `Undermine-Requirements`

---

### API Call and Retry Logic

Two-layer retry system:

**Layer 1 — `call_api()`** (mirrors EmoBench):
- Up to 3 attempts per call
- Retries on empty string response (`if not text: continue`)
- `time.sleep(2.0)` after every successful call to stay under RPM/TPM limits
- Dynamic backoff: parses `"try again in X(ms|s)"` from rate-limit errors
- Hard stops: `insufficient_quota` → `SystemExit`; `requests per day` → `return None`

**Layer 2 — `call_and_parse()`** (wraps `call_api`):
- Up to 3 attempts to get a valid JSON response
- If `parse_json()` returns `None` on a non-empty response, fires a fresh `call_api` call
- All task runners use `call_and_parse` — no bare `call_api` + `parse_json` at call sites
- On exhausting all retries: records `raw_response` as-is, scores as 0

The pilot version of `call_and_parse` additionally accumulates `prompt_tokens` + `completion_tokens` across all retry attempts for accurate cost reporting.

---

### Output Normalization

Model outputs are normalized before scoring to handle casing, whitespace, and key-name variants:

| Helper | Handles |
|--------|---------|
| `norm_item(s)` | `"food"` → `"Food"`, `"not given"` → `"Not Given"`, extra spaces |
| `norm_intent(s)` | `"build-rapport"` → `"Build-Rapport"`, extra spaces |
| `_pred_item(pred, key)` | Checks both `"high"` and `"High"` keys in the pred dict |
| `pred_intent_bitmask(list)` | Normalizes each label before bitmask comparison |
| `intent_bitmask(str)` | Used for gold labels only (already canonical, but normalized for safety) |

---

### Checkpoint / Resume

Each task writes a `.jsonl` checkpoint to `results/<task>/`. On restart, completed UIDs are skipped.

UID format:
- Desire/Belief: `"<dialogue_id>_<agent>_<task>"` (e.g. `"0-5_agent_1_desire"`)
- Intention: `"<dialogue_id>_utt<1|2>_intention"` (e.g. `"0-5_utt2_intention"`)

---

### Sharding and Merging

`--shard N --total-shards 5` splits the flat item list into 5 equal slices (~476 samples each).
Each SLURM array job (`--array=0-4`) runs one shard across all 3 tasks.

**After all 5 jobs finish**, run merge to aggregate:

```bash
bash run_merge.sh
# reads: results/<task>/gpt-4o-mini_shard{0..4}of5.jsonl
# writes: results/<task>/gpt-4o-mini_all.jsonl
#         results/<task>/gpt-4o-mini_all.csv
#         results/<task>/gpt-4o-mini_final_overall.csv
#         results/negotiation_gpt-4o-mini_results.csv   ← summary
```

`merge_shards.py` deduplicates by UID (handles checkpoint overlaps) and warns on any missing shard files before computing final scores.

---

### Output Files

```
NEG_GPT/results/
  pilot/
    pilot_desire.csv
    pilot_belief.csv
    pilot_intention.csv
  desire/
    gpt-4o-mini_shard{0..4}of5.jsonl   # per-shard raw output
    gpt-4o-mini_all.jsonl               # merged (after run_merge.sh)
    gpt-4o-mini_all.csv
    gpt-4o-mini_final_overall.csv       # {"metric": "Desire_EM", "score": ...}
  belief/
    (same structure)
  intention/
    (same structure — two metrics: Micro F1 and Macro F1)
  negotiation_gpt-4o-mini_results.csv   # ← auto-generated summary (Score, GPT-4o-mini)
```

---

## Quest Run Workflow

### Step 1 — Pilot (50 samples, ~5% of dataset)

```bash
sbatch run_pilot.sh
```

- Runs 50 random samples (seed=42) across all 3 tasks
- Output: `results/pilot/pilot_desire.csv`, `pilot_belief.csv`, `pilot_intention.csv`
- Log: `log_pilot.txt` — check for errors and the printed cost/time projection
- Decide whether to proceed and which partition/time limit to use

### Step 2 — Full Run (2,380 samples, 5 shards)

```bash
sbatch run_negotiation.sh
```

- Submits 5 array jobs (`--array=0-4`), each processing ~476 samples across all 3 tasks
- Checkpoints saved every 20 items to `results/<task>/gpt-4o-mini_shard{N}of5.jsonl`
- Monitor progress: `squeue -u $USER` and `wc -l results/desire/*.jsonl`

### Step 3 — Merge and Score

```bash
bash run_merge.sh
```

- Merges all 5 shards per task, deduplicates by UID
- Computes final scores and writes `results/negotiation_gpt-4o-mini_results.csv`

---

### Shard Correctness (verified)

Sharding uses ceiling division: `size = ceil(total_items / total_shards)`. For 5 shards:

| Task | Total items | Items/shard |
|------|------------|-------------|
| Desire | 4,760 (2,380 × 2 agents) | 952 |
| Belief | 4,760 (2,380 × 2 agents) | 952 |
| Intention | ~4,760 (most samples have 2 utterances) | ~952 |

UIDs are consistent between `openai_neg_eval.py` and `merge_shards.py`, so checkpoint deduplication and merging work correctly across shards.

---

## Known Concerns and Solutions

### 1. Unknown cost and run time before committing to the full job

**Concern:** 2,380 samples × ~6 API calls each = ~14,280 calls. Unknown token count per call and unknown GPT rate limits make it hard to estimate cost or wall time upfront.

**Solution:** Run `run_pilot.sh` first (50 random samples, seed=42). The pilot script tracks actual `prompt_tokens` + `completion_tokens` via `resp.usage` on every call, computes the real pilot cost, then scales to the full 2,380-sample dataset and prints:
- Estimated total cost (USD)
- Estimated wall time (single-threaded)
- Per-task breakdown

Decide whether to proceed and which partition/time limit to request on Quest only after seeing the pilot output in `log_pilot.txt`.

---

### 2. Quest parallel job limit (≤ 5 simultaneous)

**Concern:** With ~14,280 total API calls and a 2s sleep between each, a single-threaded run would take ~8 hours. Need parallelism, but Quest caps array jobs at 5 simultaneous.

**Solution:** Split into 5 shards (`--array=0-4`, ~476 samples/shard). Each job runs all 3 tasks for its shard. 5 jobs fit exactly within Quest's limit in one `sbatch` submission. After all finish, run `run_merge.sh` to aggregate results.

If the dataset or model changes and more shards are needed (e.g., 10), submit in two batches of 5: `--array=0-4` then `--array=5-9`.

---

### 3. Empty or unparseable model responses

**Concern:** GPT occasionally returns an empty string or a response that cannot be parsed as JSON (e.g., plain text explanation instead of `{"high": "Food", ...}`). Without handling this, the sample silently scores 0 even though a valid answer might be obtainable on retry.

**Solution:** Two-layer retry in `call_and_parse()`:
- `call_api()` already retries up to 3× on exceptions and network errors
- `call_and_parse()` adds a second loop (also up to 3 attempts): if `parse_json()` returns `None` on a non-empty response, the entire API call is retried with the same prompt
- Only after exhausting all parse retries does the sample record `pred=None` and score 0
- The `raw_response` field is always saved so failed responses can be inspected manually

---

### 4. Case and whitespace inconsistency in model outputs

**Concern:** Gold labels use canonical casing (`"Food"`, `"Not Given"`, `"Build-Rapport"`). Model outputs may return `"food"`, `"not given"`, `"build-rapport"`, `" Food "`, or capitalize dict keys differently (`"High"` vs `"high"`). Direct string equality would silently mark correct predictions as wrong.

**Solution:** Normalization layer applied to all model outputs before scoring:
- `norm_item()`: lowercases and looks up in a canonical map (`"food"` → `"Food"`, `"not given"` → `"Not Given"`); falls back to `.title()` for unknown values
- `norm_intent()`: same pattern for the 9 intent labels (`"build-rapport"` → `"Build-Rapport"`)
- `_pred_item()`: checks both `"high"` and `"High"` key variants in the pred dict before normalizing the value
- `pred_intent_bitmask()`: normalizes each predicted intent label before bitmask comparison (kept separate from `intent_bitmask()` which handles gold labels)

Gold labels are never modified — normalization is applied only to model outputs.

---
