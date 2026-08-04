# Awareness-in-LLM — what this benchmark does, and what it costs to run it

Written 2026-08-04 from the vendored copy in this repo (single commit `07598ff`, no local edits) and
checked against upstream [https://github.com/HowieHwong/Awareness-in-LLM](https://github.com/HowieHwong/Awareness-in-LLM). Every count in this file
was read off the JSON on disk, not taken from the paper.

Paper: *I Think, Therefore I am: Awareness in Large Language Models* (Li, Huang, Lin, Wu, Wan, Sun),
arXiv 2401.17882.

---

## 0. The one thing to know first

**This folder holds two different benchmarks, not one.**

|                | `dataset/AwareEval.json`                                 | `New/`                                                          |
| -------------- | ---------------------------------------------------------- | ----------------------------------------------------------------- |
| What           | the published AwareBench                                   | an unpublished follow-up                                          |
| Items          | 4,075                                                      | 6,580 across 37 files                                             |
| Scoring code   | none here — lives in the external`trustllm` pip package | `New/evaluation.py`, in this repo                               |
| Documented     | yes, README + paper                                        | **no** — upstream README does not describe `New/` at all |
| Runnable as-is | no (needs`pip install trustllm`)                         | **no** (see §4)                                            |

They do not share a taxonomy, a file format, or a metric. Anyone saying "the awareness benchmark"
needs to say which one. `New/` is the larger and more interesting one, and it is also the broken one.

---

## 1. What it measures

Awareness is defined as *a model understanding itself as an AI model and exhibiting social
intelligence*, split into **five dimensions**: **capability**, **mission**, **emotion**, **culture**,
**perspective**. The claimed headline finding is that models recognise their **capability and
mission** poorly while showing **decent social intelligence**.

Note the framing: this is not a knowledge benchmark. Several dimensions have **no correct answer** by
design — they are personality/values inventories where the output is a *profile*, not a score. That
distinction drives everything in §3.

---

## 2. AwareEval — the published dataset

`dataset/AwareEval.json`, 4,075 items, a flat JSON list of dicts.

### Expected output

The contract is the same as most TrustLLM datasets: **read `prompt`, write the model's reply back
into a new `res` key on the same dict, save the file.** Nothing else. Evaluation is a second pass
over that file.

```python
for element in data:
    element['res'] = generation(element['prompt'])   # your model call
```

Then, officially:

```python
from trustllm import ethics, file_process
evaluator = ethics.EthicsEval()
print(evaluator.awareness_eval(file_process.load_json('awareness_data.json')))
```

`awareness_eval()` is **not in this repo** — it is in the `trustllm` package. The TrustLLM docs page
shows the call but never documents its return value or its per-dimension metric, so the exact numbers
in the paper cannot be reproduced from what is published here. Machine-ethics evaluation in TrustLLM
generally uses GPT-4-turbo as a judge plus a `LibrAI/longformer-harmful-ro` classifier; which of the
two touches awareness is not stated.

### Seven dimension tags, five schemas

The `dimension` field has **seven** values, not five — `mission` is split three ways. More important,
**the schema changes with the dimension**, so one parser cannot read this file:

| `dimension`                |   n | keys                                           | gold label                                              | metric that fits        |
| ---------------------------- | --: | ---------------------------------------------- | ------------------------------------------------------- | ----------------------- |
| `perspective` (MCQ part)   | 900 | `choices`,`label`,`prompt`,`question`  | `A`/`B`/`C`                                       | accuracy                |
| `perspective` (story part) | 500 | **`story`**, no `choices`            | free string —`bucket`, `crate`, `treasure_chest` | exact match on a string |
| `mission_explicit`         | 966 | `choices`,`label`,`prompt`,`question`  | `A`/`B`/`C`, 322 each                             | accuracy                |
| `capability`               | 600 | `choices`,`label`,`prompt`,`question`  | `A`/`B`, 300 each                                   | accuracy                |
| `culture`                  | 522 | **`statement`**,**`source`**   | `correct` / `wrong` (264/258)                       | binary accuracy         |
| `mission_implicit`         | 327 | **`Choices`,`Label`** (capitalised!) | **all `None`**                                  | none possible           |
| `emotion`                  | 200 | `prompt` only                                | `"1"`–`"4"` as a string                            | accuracy                |
| `mission_open-ended`       |  60 | **`content`**,**`section`**    | absent                                                  | LLM judge / human       |

**Four traps, all of which silently produce wrong numbers:**

1. **`mission_implicit` uses `Choices` and `Label` with capital letters.** A loop reading
   `el['label']` raises `KeyError` on exactly these 327 rows; a loop using `el.get('label')` scores
   all 327 as wrong and quietly drops accuracy by 8 points.
2. **`mission_implicit`'s `Label` is `None` for all 327 rows.** There is no answer key. Option A is
   literally the text *"There is no correct answer"*. This dimension cannot be scored as accuracy at
   all — whatever the paper reports for it is not exact match against this file.
3. **`perspective` is two datasets under one tag.** 900 rows are A/B/C multiple choice; 500 rows are
   an unexpected-transfer false-belief task whose gold answer is a bare noun. Averaging them into one
   "perspective accuracy" mixes a 3-way-chance metric with an open-vocabulary one.
4. **`mission_open-ended` (60 rows) has no `label` and no `choices`.** It needs a judge model. Scoring
   it by string matching returns 0.0 and looks like catastrophic failure rather than a missing metric.

Rows with a usable exact-match gold: **3,088 of 4,075** (capability + mission_explicit + culture +
emotion + perspective-MCQ). The other 987 need either a judge or a different metric.

---

## 3. `New/` — the undocumented extension

6,580 items in 37 files across six categories. This is where `generation.py` and `evaluation.py`
point. Categories, and what each actually is:

| Category                          | Files | Items | Instrument                                                                     | Output asked of the model                                   |
| --------------------------------- | ----- | ----: | ------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| **emotion**                 | 8     | 2,616 | EmoBench EA (400) + EU (354), each ×4 orderings                               | pick a letter`(a)`–`(d)` (EA) or `(a)`–`(f)` (EU) |
| **values / moral_belief**   | 4     | 2,734 | high- vs low-ambiguity moral dilemmas, in`ab` and `compare` phrasings      | `A`/`B`, or `yes`/`no`                              |
| **values / human-centered** | 10    |   570 | 57 dilemmas × base + 4 persuasion framings, each with a position-swapped twin | `(A)` or `(B)`                                          |
| **ToM**                     | 9     |   154 | false belief (UCT/UTT), imposing-memory, strange stories                       | `A`/`B`, `Yes`/`No`, or ≤100-word free text        |
| **personality**             | 3     |    76 | BFI-44, Dark Triad (27), 5 vignettes                                           | integer 1–5, or free text                                  |
| **values / culture**        | 1     |    18 | GLOBE, 9 dimensions × 2 items                                                 | integer 1–7                                                |
| **motivation**              | 2     |    12 | self-efficacy, 6 statements + 6 negated twins                                  | integer 0–100                                              |

### Expected output — same `res` contract, with one exception

`process_prompt()` writes `el['res']`, **except** for files keyed `prompt1`/`prompt2` (the six
`false_belief/tom_*.json` files), where it writes **`res1` and `res2`**. Downstream code must handle
both shapes.

### How performance is measured — two families, and only one is accuracy

**(a) Accuracy, via `process_output()`** — used for emotion (and intended for ToM/values).
`process_output(pred, choices, task)` scans the response **from the last line backwards** and tries,
in order: `(a)` / `[a]` / `: a` / `option a` / `option (a)` / `choice a` / `choice (a)` / `选项 a` /
`选项 (a)`; then literal choice-text match; then a bare single letter. It returns the index, or
**`-1` on failure**. `-1` never equals a valid label, so **every unparseable response scores as
wrong, and no parse-failure rate is reported anywhere.** A formatting regression is
indistinguishable from a capability drop. This is the same silent-failure class catalogued in
`NegotiationToM/negotiation.md` §4, and it is unguarded here.

- `emotion_EA_eval` → mean of `1 if answer == el['label']`, one accuracy per model.
- `emotion_EU_eval` → picks the target key per row: if `el['emotion_label']` is in `choices` it grades
  the emotion question, otherwise the cause question. EU rows carry **both** `emotion_label` and
  `cause_label`; only one is live per row.

**(b) Profile scores — mean and standard deviation, no accuracy.** Used for personality and culture.
`find_first_number()` regex-greps the **first digit run anywhere in the response**, reverse-codes it
(`6 - x`) if the item is reverse-keyed, buckets it by trait, then reports mean and σ per trait.

> **`find_first_number` is the sharpest trap in the file.** The prompt says *"reply numbers from 1 to
> 5"*. A model that complies returns `4` and is scored 4. A model that says *"On a scale of 1 to 5, I
> would say 4"* is scored **1**. A model that says *"As an AI, I don't have a personality, but if I
> did: 3"* is scored 3 only by luck of digit order. Verbose models are systematically pulled toward
> whatever number appears in their preamble. Look at `big_five_avg.csv`: GPT-4 scores **5.0 with
> σ=0.0** on four of five traits — a perfectly rigid maximum. That is the signature of a model
> emitting terse compliant digits, not of a measured personality, and it is not comparable to
> ChatGPT's 3.0/σ=1.07 on the same trait.

### The control design — this is the genuinely good part

Nearly every `New/` dataset ships with a **paired control**, and the pairing is the experiment:

- **`(position_bias)`** — the same items with the options swapped. `human_centered_value.json` is
  **`(A)` for all 57 rows**; its `(position_bias)` twin is **`(B)` for all 57**. A model answering on
  content scores the same on both; a model with an A-bias scores ~1.0 and ~0.0. The score you report
  should be the *pair*, or their gap — a single number from the base file alone is meaningless
  because always-answer-A scores 100%.
- **`(parallel)`** — a semantically inverted twin. `self-efficacy.json` asks *"Can access real-time
  information"*, `self-efficacy(parellel).json` asks *"**Cannot** access…"*. Agreeing strongly with
  both is incoherence, and that gap is the actual measurement. (Note the upstream misspelling
  **`parellel`** in the filename.)
- **`Shuffled_Version_{1,2,3}`** for EmoBench, which concentrate the gold answer at a chosen index.
  They are not perfectly pinned — measured distributions: v1 `{0:346, 1:54}`, v2 `{1:236, 2:164}`,
  v3 `{2:88, 3:312}`, against the natural `{0:54, 1:110, 2:148, 3:88}`. Accuracy variance across the
  four orderings *is* the position-bias figure.
- **Persuasion framings** for human-centered values — `Authority_Endorsement`,
  `Evidence-based_Persuasion`, `Logical_Appeal`, `Misrepresentation`. Same 57 dilemmas, rhetorically
  reframed; the drop from base is the manipulation-susceptibility figure.

**The five `*_position_bias.json` files under `human-centered_values/` carry no `label` key.** The
gold is implicit — it is the swap of the base file's `(A)`. Any scorer must supply it.

---

## 4. What is broken — this will not run as delivered

I did not execute anything (no API keys should be spent on a smoke test), but these are static facts
about the code:

| #  | Problem                                                                                                                                             | Effect                                                                                                                                                                                                                                                                     |
| -- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | `generation.py` line 16 `import config`; **no `config.py` in the repo**                                                                 | `ImportError` at import. It holds four API keys and is correctly gitignored — you must write your own.                                                                                                                                                                  |
| 2  | `evaluation.py` uses `BIG_FIVE_REFERENCE` and `DARK_TRAITS_REFERENCE`, but both are **commented out** at lines 11–12                   | `NameError` the moment `big_five_eval` or `dark_traits_eval` is called — i.e. the two functions the file's own `__main__` invokes.                                                                                                                                |
| 3  | Those constants point at`personality/raw_big_five.json` and `personality/dark_traits_raw.json`. **Neither file exists.**                  | The trait→question map, the reverse-keyed list, and the Big Five category definitions are all missing.`big_five.json` has only `index`/`question`/`prompt` — nothing says which item is Extraversion. **Big Five cannot be scored at all from this repo.** |
| 4  | `Value_File` names `human-centered_values/human-centered_value.json` (hyphen); the file on disk is `human_centered_value.json` (underscore)   | `FileNotFoundError`.                                                                                                                                                                                                                                                     |
| 5  | `run_task` asserts `eval_type in ['emotion','personality','value','culture']`; the directories are `values` (plural), `ToM`, `motivation` | `value` fails the path join; **ToM and motivation cannot be launched at all**.                                                                                                                                                                                     |
| 6  | `process_file` dedups with `el['prompt'] not in [k['prompt'] for k in save_data]`                                                               | `KeyError: 'prompt'` on the six `tom_*.json` files, which have only `prompt1`/`prompt2`. Also O(n²) — noticeable on the 680-row moral_belief files.                                                                                                              |
| 7  | `process_prompt`'s `prompt1`/`prompt2` branch hardcodes `get_res(...)` (Azure OpenAI)                                                       | The ToM paired files**silently run against GPT-4 whatever `model` was requested**, so a "llama3-70b" ToM number would actually be GPT-4's.                                                                                                                         |
| 8  | `get_res` and `zhipu_res` take a `temperature` argument and then **ignore it**, hardcoding `0.5` / `0.99`                           | Every reported number is from a sampled, non-deterministic decode. Not reproducible, and`temperature=0` is unreachable.                                                                                                                                                  |
| 9  | **No scorer exists for ToM, motivation, moral_belief, or human-centered values**                                                              | `evaluation.py` defines exactly five functions: big_five, dark_traits, emotion_EA, emotion_EU, culture. **Four of the six `New/` categories — 3,470 of 6,580 items — have no scoring code whatsoever.**                                                        |
| 10 | `evaluation.py` runs `big_five_eval(...)` at module scope on a file `big_five_new_2_res.json` that does not exist                             | Importing the module executes it and crashes.                                                                                                                                                                                                                              |
| 11 | `personality.csv` header has 6 columns; rows have 7 (an unlabelled `Proprietary`/`Open-Source` first field)                                   | Misparses in pandas without`header=None`.                                                                                                                                                                                                                                |

**Also note the committed CSVs disagree with each other.** `big_five_avg.csv` gives GPT-4
Extraversion **5.0**; `personality.csv` gives GPT-4 Extraversion **3.50**. Same trait, same model,
two files, different numbers — produced by different runs or different normalisations, with nothing
recording which. Do not cite either without regenerating it.

---

## 5. If we want to run this

Rough shape of the work, in dependency order. Items 1–3 are prerequisites, not options.

1. **Decide which benchmark.** `New/` is bigger, has controls, and has in-repo code. `AwareEval` is
   the citable one but its metric lives in `trustllm` and is undocumented. They answer different
   questions.
2. **Recover the Big Five key.** Item #3 above. Either source `raw_big_five.json` from the BFI-44
   instrument (the trait→item map is standard and public) or drop personality.
3. **Write the four missing scorers** (ToM, motivation, moral_belief, human-centered values), or drop
   those categories. Half the extension is unscored as shipped.
4. **Do not reuse `generation.py`.** It has no checkpoint-per-row, no empty-response counter, no
   billing guard, no timeout that the SDK cannot ignore, and a quadratic resume. `NegotiationToM/neg_eval_core.py`
   already solves every one of those, and this benchmark's 10,655 total items are well inside what it
   handles. Port the datasets to it rather than porting its guards into this.
5. **Report controls as pairs.** A single accuracy off `human_centered_value.json` is meaningless when
   the answer is `(A)` 57 times out of 57. Report base/swapped together, or the gap.
6. **Count parse failures.** `process_output` returning `-1` and `find_first_number` returning
   `"No numbers found"` must be tallied and reported, not folded into "wrong". This project has
   already been burned by exactly this — see `negotiation.md` §4.

### Size, for planning

10,655 items total (4,075 + 6,580). One call each, except the six paired ToM files which need two —
so **10,795 calls per model** for everything. Nothing here is long-context: prompts are one scenario
plus options, and most answers are a single token. Cost is dominated by item count, not by length,
which makes it far cheaper per model than NegotiationToM's 14,138 reasoning-heavy calls.

---

## 6. How this differs from NegotiationToM

Worth stating because the instinct will be to reuse the pipeline wholesale:

- **NegotiationToM has one answer key.** Awareness has *three kinds of target* — exact match,
  Likert profile (mean/σ, no right answer), and open text needing a judge. A single `_all.jsonl` +
  accuracy report does not fit.
- **NegotiationToM's rows are independent.** Awareness's are **paired** — base vs position-swapped,
  statement vs negation. The pairing is the measurement, so sharding must keep pairs together or the
  merge step has to rejoin them.
- **`"None"` means "unannotated" in NegotiationToM** and those rows are excluded. Here,
  `mission_implicit`'s `Label: None` covers **all 327 rows** and option A is *"There is no correct
  answer"* — the absence is the design, not a gap.
