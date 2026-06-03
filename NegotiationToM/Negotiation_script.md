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

## Script Design (NEG_GPT/openai_neg_eval.py)

### Model and SDK

| File | Model | SDK |
|------|-------|-----|
| `NEG_GPT/openai_neg_eval.py` | `gpt-4o-mini` | `openai.OpenAI` |

Same pattern as EmoBench: load API key from `.env`, iterate over samples, call model with JSON-structured output prompt, save per-task CSVs plus an overall results CSV.

---

### Data Loading

- Dataset is password-protected: extract with `unzip -P "NegotiationToM" NegotiationToM.zip`
- Script expects the extracted `NegotiationToM.json` (2,380 samples)
- Each sample: `dialogue_id` (e.g. `"0-5"` = dialogue 0, turn cutoff 5), full `dialogue` list, ground truth desire/belief/intent labels

---

### Three Sub-Tasks and Prompts

#### 1. Desire (`--task desire`)

One API call per agent per sample (2 calls/sample → 4,760 total).

**System prompt:** Instructs the model to assign exactly one of High / Medium / Low to each of Food, Water, Firewood. Reply in JSON `{"high": "<item>", "medium": "<item>", "low": "<item>"}`.

**User prompt:** Provides the full dialogue and asks for the specified agent's priorities.

**Scoring — Exact Match:** All three slots (high/medium/low) must match the ground truth simultaneously. Single-slot errors score 0.

#### 2. Belief (`--task belief`)

One API call per agent per sample (2 calls/sample → 4,760 total).

**System prompt:** Same structure as Desire but allows `"Not Given"` for slots where the belief cannot be inferred from the dialogue.

**User prompt:** Provides the dialogue and asks what `agent_X` believes about `agent_Y`'s priorities.

**Scoring — Exact Match:** Same strict three-slot rule. `"Not Given"` is a valid answer and must match the gold label exactly.

#### 3. Intention (`--task intention`)

Two API calls per sample (one per the last two utterances → up to 4,760 total).

**System prompt:** Instructs the model to select one or more intents from the 9-label set. Reply in JSON `{"intents": ["label1", "label2"]}`.

**User prompt:** Provides the full dialogue plus the target utterance highlighted separately.

**Scoring — Micro F1 + Macro F1:** Predictions are binarized against the 9-label set and scored with `sklearn.metrics.f1_score`. Partial credit is given for partially correct label sets.

**9 intent labels:** `Build-Rapport`, `Callout-Fairness`, `Describe-Need`, `Discover-Preference`, `No-Intention`, `No-Need`, `Promote-Coordination`, `Show-Empathy`, `Undermine-Requirements`

---

### Retry and Rate-Limit Logic

Mirrors EmoBench's `call_api()`:
- Up to 3 attempts per call
- `time.sleep(2.0)` after every successful call to stay under RPM/TPM limits
- Parses `"try again in X(ms|s)"` from rate-limit errors for dynamic backoff
- Hard stops on `insufficient_quota` (raises `SystemExit`) or `requests per day` (returns `None`)

---

### Checkpoint / Resume

Each task writes a `.jsonl` file to `results/<task>/`. On restart, completed UIDs are skipped. UID format: `"<dialogue_id>_<agent>_<task>"` for desire/belief, `"<dialogue_id>_utt<1|2>_intention"` for intention.

---

### Sharding

`--shard N --total-shards M` splits the flat item list into M equal slices so multiple jobs can run in parallel. Example (4 shards):

```bash
for i in 0 1 2 3; do
  python openai_neg_eval.py --task desire --shard $i --total-shards 4 &
done
```

---

### Output Files

```
NEG_GPT/results/
  desire/
    gpt-4o-mini.jsonl         # per-item results
    gpt-4o-mini.csv           # same as JSONL but CSV
    gpt-4o-mini_overall.csv   # {"metric": "Desire_EM", "score": ...}
  belief/
    (same structure)
  intention/
    (same structure — includes Micro F1 and Macro F1)
```

Top-level `negotiation_results.csv` is filled in manually after all tasks complete, following the `emobench_results.csv` format.

---
