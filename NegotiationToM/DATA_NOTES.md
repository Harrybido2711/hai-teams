# NegotiationToM.json — structure notes

Facts about the dataset (2,380 samples, 396 dialogues) that are **not** obvious from reading the
eval code, each verified against the data. Several of these silently change scoring, so read this
before touching `neg_eval_core.py`.

---

## Cutoff tiling

`dialogue_id = "<dialogue>-<cutoff>"`. Each dialogue is sliced every 2 turns (2, 4, 6, 8, 10 …),
plus one final sample holding the complete dialogue. A sample's targets are its last 1–2
utterances, and across all cutoffs of a dialogue **every utterance is asked exactly once** —
verified: 396/396 dialogues tile perfectly.

So `intents` carries labels for the whole dialogue while any single sample uses only 1–2 of them.

## Odd vs even length → 1 or 2 target utterances

Dialogues alternate agent_1/agent_2, so intermediate cutoffs are always even-length and end on a
complete exchange → **2 targets**. The final uncut sample can be odd-length when the original
dialogue ends mid-exchange → **1 target** (142 of 2,380 samples; all 142 verified to be their
dialogue's longest).

| | `utterance1_intent` | `utterance2_intent` |
|---|---|---|
| even length (2,238) | ↔ `turns[-2]` | ↔ `turns[-1]` |
| odd length (142) | ↔ **`turns[-1]`** | the string `"None"` |

Detect with `sample["utterance2_intent"] == "None"`. It is the **string** `"None"`, not JSON null,
and not one of the 9 intent labels — so a naive implementation maps it to an all-zero bitmask
without raising, creating phantom rows that make every prediction a false positive.

**Correct total intention rows: 2,238×2 + 142×1 = 4,618** (not 4,760).

Targets are chosen by **position, not speaker identity** — `utterance1_agent` is `agent_2` in 1,106
samples. 41 dialogues are not strictly alternating (the same agent speaks twice), which does not
affect the positional rule.

## `"None"` means "never annotated", not "wants nothing"

Three pieces of evidence:

1. It never appears mixed with a real label — all three priority slots are `"None"` together, with
   **0** mixed cases across all four field groups.
2. Desire and belief are `"None"` on the **same** samples.
3. The `agent{N}_desire` dict still holds real values for those samples. Example `1-5`: flat fields
   all `"None"`, dict `{'High':'Food','Low':'Water','Medium':'Firewood'}`.

Counts: 84 samples for agent_1, 72 for agent_2 ⇒ **156 rows each in desire and belief** (3.3%).
Those rows have no correct answer, so they are excluded from the metrics — the same convention as
`utterance2_intent == "None"` — while still being written to CSV.

Consequence for normalisation: gold `"None"` and a model answering `"None"` are different things.
Since all-`None` gold rows are excluded, gold in any *scored* row is one of
Food / Water / Firewood / Not Given, so a model emitting `"None"` is simply wrong.

## Which gold fields to use

Two parallel encodings, and they are **not** equivalent:

| Field | Shape | Contents |
|---|---|---|
| `agent{N}_desire` | dict | always a complete Food/Water/Firewood permutation, never "Not Given" |
| `agent{N}_desire_{high,medium,low}` | flat strings | cutoff-aware; "Not Given" ×2,338, "None" ×252 |
| `agent{N}_belief_{high,medium,low}` | flat strings | agent N's belief about the *other* agent |

**Desire must be scored against the flat cutoff-aware fields.** The spec's Desire label set is
`{Food, Water, Firewood, Not Given, None}`; the dict can never express the last two. Using the dict
asks the model for the full true ordering at a cutoff where it is not inferable.

Note that `Output_template/openai_desire.csv` stores the dict — it was produced by the buggy path
and is not independent evidence.

## Task metrics

| Task | Metric | Notes |
|---|---|---|
| Desire | Exact Match | all three slots simultaneously; partial credit is not given |
| Belief | Exact Match | same, for agent N's model of the opponent |
| Intention | Micro + Macro F1 | multi-label over 9 intents; **partial credit applies** |
| All | Exact Match | AND across every scored row of all three tasks sharing a `dialogue_id` |

Consistency_Desire / Consistency_Belief are out of scope for this project.
