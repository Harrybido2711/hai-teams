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
