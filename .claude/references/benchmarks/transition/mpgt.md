<!-- size-budget: 16000 -->
# mpgt-eval — benchmark card

Multi-party goal tracking, for goal specification. Upstream `AddleseeHQ/mpgt-eval`, vendored
2026-08-19. **No LLM judge.** The paper's metric is exact match on the gold `[MASK]` strings with the
residue reviewed by a person (`LLM_as_judge/JUDGE_SELECTION.md` §1); **no scoring code ships in the
repo** — the `gpt/` scripts only write predictions to a CSV. **Our runners exist since 2026-11-09**:
zero-shot, the five selected models, scored by one automatic lenient matcher instead of a person
(the user's decisions, 2026-11-09). **Smoke-tested locally 2026-11-11 (§ Smoke): all five run end to
end, but Deepseek still hits its cap on 1 row in 12. Not on Quest, and not until the user has
verified the scripts.** No results.

## Paths

| | Path |
|---|---|
| Local | `Transition_processes_benchmarks/Multi-party_Goal_Tracking_bench` |
| Quest | none yet — nothing has been transferred. AwareBench's copy sits at `/gpfs/projects/p32983/Transition_processes_benchmarks/Awareness_in_LLM`, so the mirror would be beside it; confirm on Quest, never infer |

## Layout

```
Multi-party_Goal_Tracking_bench/
├── mpgt_eval_core.py   ours: data, upstream's six prompts, the loop (run_cli), the lenient scorer
├── test_mpgt_scorer.py ours: the scorer's checks on the real data — run before trusting a number
├── MPGT_GPT_5.6_Luna/ · MPGT_Gemini_Flash3.5lite_OpenRouter/ · MPGT_Qwen_DeepInfra/
│   MPGT_Gemma_DeepInfra/ · MPGT_Deepseek/      one client each + run_mpgt.sh
├── gpt/            zero_shot_dst_only_gpt.py · few_shot_dst_only_gpt.py
│                   zero_shot_goal_only_gpt.py · few_shot_goal_only_gpt.py
├── Preprocessing/  train-test-split.py · dst_and_goal_masking.py · inject_noise_to_dialogue.py
├── mapping/        elan-to-dlm.py · count-turns.py
└── README.md
```

**The call layer is imported, not ported:** `mpgt_eval_core.py` puts
`Action_processes_benchmarks/` on the path and uses `judge_core.py` (halts, watchdog, retries,
negotiation, checkpoint), as MultiChallenge and Wonderbread do. **A sync of mpgt to Quest includes
`judge_core.py`**, and an edit to `judge_core.py` changes three benchmarks. `.env` is local and
gitignored, copied from MultiChallenge's; Quest needs its own copy.

Upstream ships **four prompt conditions** — zero-shot and few-shot, each for DST-only and goal-only.
They are four separate scripts, not four flags. **Ours run zero-shot only, both tasks** — the
user's choice for the first run, 2026-11-09; few-shot can be added without redoing it.

## Expected counts — verified 2026-11-09

Measured from `Preprocessing/data/split-diogs/` and the four `gpt/` scripts. One row is one dialogue
— `src` holds `[MASK]`s inside a `[start]`…`[end]` window, `tgt` is the gold — and one row is one
API call. Every script loops over **six prompt styles** (basic, specific, story, roleplay,
annotation, reasoning); ours run all six (the user, 2026-11-09).

| Condition | Test file | Rows | `[MASK]` slots | Calls per model, six styles |
|---|---|---|---|---|
| zero-shot DST-only | `zero-shot/dst-only/dst-only.jsonl` | 29 | 389 (3–27 per row) | 29 × 6 = 174 |
| zero-shot goal-only | `zero-shot/goal-only/goals-only.jsonl` | 29 | 225 (2–16 per row) | 174 |
| few-shot DST-only | `pcent-20/dst-only/run{1,2,3}-test.jsonl` | 23 each | 319 · 302 · 293 | 23 × 6 × 3 = 414 |
| few-shot goal-only | `pcent-20/goal-only/run{1,2,3}-test.jsonl` | 23 each | 178 · 176 · 172 | 414 |

- **A full run of ours is 348 rows per model** (the two zero-shot rows above). `mpgt_eval_core`
  refuses to run if the files hold anything but 29 rows and 389 / 225 slots.
- Few-shot prepends the first **two** train dialogues (`range(0, 2)`) and uses runs 1–3 of the five
  splits on disk; `pcent-50` and `pcent-80` are not read by any script.
- `goal-given-dst` and `multi-task` splits exist on disk (29 rows; 210 and 542 slots zero-shot) but
  no script in `gpt/` reads them.
- Upstream calls `gpt-3.5-turbo` at temperature 0.4 through the pre-1.0 `openai.ChatCompletion` API,
  sleeping 45–90 s after every call (it differs per script) — a reference for the prompts, not a
  runner to reuse. Ours take the six prompt builders from that source text (`ast`), byte for byte.

## Model parameters

| Model | Folder | Set |
|---|---|---|
| GPT | `MPGT_GPT_5.6_Luna` | `gpt-5.6-luna`, `reasoning_effort="low"`, `max_completion_tokens=8192`, `seed=42` |
| Gemini | `MPGT_Gemini_Flash3.5lite_OpenRouter` | OpenRouter, reasoning `{"effort": "minimal"}`, `max_tokens=8192`, `seed=42` |
| Qwen | `MPGT_Qwen_DeepInfra` | DeepInfra, reasoning `{"enabled": False}`, `temperature=0`, `max_tokens=8192`, `seed=42` |
| Gemma | `MPGT_Gemma_DeepInfra` | DeepInfra, `temperature=0`, `max_tokens=8192`, `seed=42` |
| Deepseek | `MPGT_Deepseek` | `deepseek-flash`, prompt ceiling, `temperature=0`, **`max_tokens=16384`** |

**Deepseek's cap is twice the others' — the user, 2026-11-11, "raise it a bit, not too much".**
One call annotates a whole window, and the ceiling does not keep the reasoning short: at 8192, 7 of
12 smoke rows ended at `length` with no answer. At 16384 one in 12 still does (§ Smoke).

## Scoring — `mpgt_lenient_v4`

Upstream's masking (`Preprocessing/dst_and_goal_masking.py`) replaces each turn's annotation inside
the window — from its `#` (DST) or `@` (goal) to the turn's end — with one `[MASK]`; the utterances
are untouched and `tgt` is the window unmasked. So a token alignment of the window against `tgt`
yields every gold slot exactly (all 614 checked). **The window is the noise condition** — the
README: "the task in the paper is window-based de-noising" — so there is no separate noise arm.

**The dialogue rewritten is the answer; everything else only fills gaps** (v4, 2026-11-11, after
five reviews — A1–A6, F1–F6, G1–G6, H1–H6, R1–R5, the fifth `safe-to-run`; the code's header and
`_predict` hold the detail).
- **turns** — copies of the window among the response's turns, matched by speaker and utterance;
  echoed context is never read for a slot. An annotation right after a turn's utterance is a
  **strong** answer; a copy covering at least half the masked turns has its strong answers replace
  an earlier copy's, and a later blank replaces nothing. An annotation found after prose is
  **weak** — looked for only where the turn opens its own line (`- LC :`, `> LC :`); a turn quoted
  inside prose (`after \`LC : ...\`, this turn is # x`) answers nothing.
- **fillers** — weak answers, small copies (notes, corrections quoting a turn or two), then a
  **list** (the last run of numbered items counted from 1 with one item per slot, at least half
  answered; an item that opens with a window turn answers only for that turn) or, where nothing is
  numbered, a **row** (markers outside the dialogue's turns, each shaped like the scheme — a goal
  `G/AG/CG ( speaker , ... )`, a DST item with brackets or a name of the scheme, no placeholder
  like `intent ( slot )` — exactly one per slot: a table, JSON, answers in a row). A filler only
  fills a slot still unread. **Where it disagrees with the answer kept, that is a conflict:
  counted, never applied** — and a list or row with any conflict fills nothing, since it may be
  shifted by a slot. `#3`-style references are not markers.

A bare word is an annotation only right after a marker, after a joiner where the annotation then
ends, or as a name of the scheme (the 27 top-level names in the gold, `→ affirm`, `offer help ( )`)
— so `here the speaker`, `* then`, `, probably .` are prose, never items. Two annotations with
words between them (`# x rather than # y`, `# x (or possibly # y)`) are left unread.

- **lenient** — the headline. A slot is correct when the predicted annotation equals the gold as a
  *set* of items, after dropping spacing, case, the marker, `_`/`-`, an empty `( )` left off or
  added (`request ( help ( ) )` is `request ( help )` — the gold never writes both), and the order
  of items and of speakers in `LC+RP` (`lc_rp`, `lc rp` too — goal 19's gold writes `lc_rp`), `_`/`-`
  written as spaces (`closing time`, `offer help`). Content is strict: an item too many — a second
  intent, a hedge joined to the answer (`# x or # y`) — or too few is wrong. A note elsewhere
  offering another tag never changes the answer; at most it is a conflict.
- **exact** — the same tokens up to spacing and case; reported beside, never the headline.
- **An unread slot counts as wrong**, and `--summarize` prints `CHECK` for a style with more than
  20% of its slots unread — read those responses before believing that style's number. It also
  prints `via:` — how many slots each reading answered — and `conflict=` with `DISAGREE` for a style
  where responses answered a slot twice, differently, or a later full copy changed an answer
  (`revised=`). A misread is never flagged by `CHECK`; an unexpected route in `via` or a `DISAGREE`
  is where one would show, and is a reason to read those responses.
- **Headline per task = the plain mean of the six styles' slot accuracy.** A task missing a style or
  a dialogue prints `INCOMPLETE` and is not a headline.
- `test_mpgt_scorer.py` holds all five reviews' cases, on all 58 dialogues: 61 shapes of the gold
  that must read at 100%; 4 that answer twice, where the dialogue's answer must stand and the
  conflict be counted; 5 with repeated turns that may leave a slot unread but read none wrong; 11
  answerless responses from which nothing may be read; 22 that must credit nothing; and R1, R3, R4
  checked one by one.
- `python mpgt_eval_core.py --summarize MPGT_*/results/zero-shot/*.jsonl` rescores stored rows; the
  runner prints the same table at the end of a run.

**Goal scores 0 for every model in the smoke, and that is the task, not the reader.** Models get
the `G`/`AG`/`CG` type and the speaker right but name the goal in their own words — gold
`G ( LC , drink ( coffee ) )`, a model `G ( LC , get-info ( coffee ( place ) ) )`. A structure
score (type and speaker only) beside the headline is proposed and waits for the user.

## Smoke — 2026-11-11, dialogue 00, 12 calls a model

| Model | Rows | Seconds a call | Unread rows | Note |
|---|---|---|---|---|
| GPT | 12 ok | ~10 | dst/story | answered as a 19-item list for 23 slots |
| Gemini | 12 ok | ~2 | goal/story, goal/reasoning | 5 and 2 answers for 6 slots; goal/story's preamble holds a format example, `such as @ G ( speaker , action ( parameters ) )`, which is not read |
| Gemma | 12 ok | ~10 | none | |
| Qwen | 12 ok | 30–450 | 8 of 12 | echoes the input unfilled, lists every annotation of the dialogue, numbers 42 items for 23 slots, or reasons in prose without tags; goal/story looped to its 8192 cap; goal/roleplay rewrites the window and then lists answers that disagree on 3 slots — the list skips a turn, so it fills nothing (`conflict=3`) |
| Deepseek | 11 ok, 1 `length` | 35–90 | none | reasoning 5.9k–15.8k tokens; dst/story ran out at 16384 |

Every unread row was checked by hand: the model's answer, not the reader's miss. Misreads were
looked for too: under v3 no slot of the 60 rows is read from a word without brackets or from a
format example, and each style is read by the route its shape implies (`via`: rewrites by turns;
the goal story style of GPT, Gemma and Deepseek and the goal roleplay of GPT and Deepseek by list;
Gemini's bare goal row by row). Smoke rows are not
committed (no benchmark's are); the 8192 Deepseek rows sit locally in
`MPGT_Deepseek/archive/smoke_max_tokens8192_2026-11-09/`.

## Output and logs

`MPGT_<Model>/results/zero-shot/<model id, "/" → "-">.jsonl`, one row per (task, style, dialogue),
uid `task/style/NN`; `smoke/zero-shot/` for any `--limit` run, never `results/`. sbatch logs are
`log_<jobid>.txt`/`.err` in the model folder. Results are a model-named folder of ours; upstream ships
none. A run exits 1 unless every unit has an `ok` row, and **2 if any row hit the output cap**.

## Run order

1. `python test_mpgt_scorer.py` — ALL PASS, or nothing is believed.
2. In a model folder: `python <runner> --dry-run`, then `--limit 1` (12 calls, writes `smoke/`).
3. On Quest, after the user's verification: `sbatch run_mpgt.sh` from inside the model folder.

## Showing reasoning

The README says nothing about it. Upstream's own `reasoning` style asks the model to step through
the annotation; that is the prompt, not an addition. Nothing is added to any style. Deepseek alone
gets a system-message prompt ceiling (`prompt-ceiling.md`), as on AwareBench and MultiChallenge.

## Traps

- **The gold has 11 unbalanced annotations** — nine left open, two with an extra `)`
  (`r-repet ( directions ( cafe ) ) )`, DST dialogue 5). The reader closes what is open — at a line
  break, a quote, a marker, or prose (a gold bracket never holds a word before a word or a full
  stop, nor any sign but `( ) ; , +`) — and skips a `)` with nothing open; `exact` misses the two
  extra-`)` slots even for the gold itself. Words in a row inside a bracket are read as one name
  (`closing time`), so an item left open runs into words after it: `# x ( y , I think`, `# x ( y
  rather than # z`, `# x ( y (confident)` read wrong — 9 slots, and only if a model copies the
  gold's slip.
- **The dialogue repeats itself** — "Thank you .", ARI giving the same directions verbatim — so an
  echoed context can look like a copy of the window, and two window turns can read the same. The
  first is why echoed context is never read for a slot. The second cannot always be settled: when
  a response quotes only one of two identical turns, the answer goes to the one in sequence, and
  the other is unread. Total credit is right; which of the two slots holds it may not be.
- **Models answer in many shapes**: the window rewritten, a `[MASK]` kept beside the answer, a
  numbered list with or without speakers or the marker, an analysis list then an answer key, a
  table, JSON. Each shape seen is a case in the scorer test. A shape the reader cannot place comes
  out unread and is flagged `CHECK`; one it misreads would not be, and shows only in `via` — so a
  new shape goes into the test before a number that depends on it is believed. Known unread: a row
  of answers with neither marker nor numbering, and an annotation with neither marker nor brackets
  whose name is not one of the scheme's; a one-line rewrite with prose before each annotation
(only line-opening turns may have prose first); two annotations with words between them —
including `# x ( ) and also # y ( )`, and a remark on the line right after the last turn that
mentions a marker (`(each # is followed by the intent)`), which costs that last slot.
- **One known misread: a later full copy that is not the answer.** "A common mistake would be:"
  followed by a whole wrong rewrite reads exactly like a draft and its final version; without the
  English it cannot be told apart, so the later copy is read. It is never silent: every slot it
  changes is counted as `revised`, and the style shows `DISAGREE`.
