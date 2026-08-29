# BIG-Bench Hard (BBH) — what it runs and how this folder is laid out

General task ability, not a team process. It exists so a team-process number can be read against
what the model can do at all. **No LLM judge:** the model is asked to end with
`Final Answer: <answer>` and that string is compared to the gold target.

## What the tasks actually are

**20 task files, 4,833 items** (counted from the JSONs, 2026-08-29). Every task is 250 items except
`causal_judgement` (187) and `penguins_in_a_table` (146). Each JSON is `{"examples": [{"input",
"target"}]}` — `input` is the whole prompt including its answer options, `target` is the gold string.

Upstream BBH ships 27 tasks; **seven are not vendored here** (`disambiguation_qa`, `hyperbaton`,
`movie_recommendation`, `ruin_names`, `salient_translation_error_detection`, `snarks`,
`sports_understanding`). A "BBH score" from this folder is over 20 tasks and is not comparable to a
published 27-task average.

What the 20 ask for, by answer shape — this is what makes the matching problem below:

| Answer shape | Tasks | Gold looks like |
|---|---|---|
| Multiple choice, lettered | `date_understanding` · `geometric_shapes` · `logical_deduction_{three,five,seven}_objects` · `penguins_in_a_table` · `reasoning_about_colored_objects` · `temporal_sequences` · `tracking_shuffled_objects_{three,five,seven}_objects` | `(B)` |
| Binary | `boolean_expressions` (`True`/`False`) · `causal_judgement` · `navigate` · `web_of_lies` (`Yes`/`No`) · `formal_fallacies` (`valid`/`invalid`) | `No` |
| Free-form string | `dyck_languages` (closing brackets) · `multistep_arithmetic_two` (an integer) · `object_counting` (a count) · `word_sorting` (a sorted word list) | `] ) ]` · `-14` · `arm bumblebee cat` |

The free-form row is where models are penalised for formatting rather than reasoning, and it is why
two different scorers exist in this folder.

## The matching problem, and what the scripts do about it

Every runner does the same first step: `re.search(r"Final Answer:\s*(.*)", output, re.IGNORECASE)`,
falling back to **the entire response** when the model never emitted the marker — a fallback that
scores 0 on anything but a one-word answer. From there the eight runners split in two.

**Five score strictly** — `deepseek`, `gemini`, `kimi`, `llama`, `openai`:

```python
return int(final_answer.lower().strip() == gold_answer.lower().strip())
```

Case-folded equality and nothing else. `B` against gold `(B)` scores 0. So does `barn, damp` against
`barn damp`.

**Three score leniently** — `xai`, `gemma`, `qwen`. `xai_eval.py::score_response` is the original;
gemma's and qwen's are copies of it. Six branches, tried in order, any one of which scores 1:

1. **Exact**, case-folded and stripped — plus `.strip("\"'`")` on the extracted answer first, so
   `"(B)"` and `` `No` `` survive quoting.
2. **Letter for letter.** Gold matches `^\([A-Z]\)$`; take the first letter of the answer with the
   parens optional, so `B`, `(B)` and `(B) a hexagon` all match gold `(B)`.
3. **Option text for letter.** Same gold shape, but the model answered with the *content*: the
   question is re-parsed with `re.findall(r'\(([A-Z])\)\s*([^\n(]+)')` into a letter→text map and
   the gold letter's text is compared. `hexagon` matches gold `(B)`.
4. **Comma-vs-space.** `final.replace(",", " ").split() == gold.split()` — `barn, damp` matches
   `barn damp`. Token-level, so it also absorbs runs of whitespace.
5. **Answer restated with the prompt's prefix.** For `dyck_languages` the gold is only the *closing*
   brackets; a model that helpfully repeats the whole sequence is matched by splicing the question's
   `Input:` line onto the gold and comparing to that.
6. **Both sides comma-normalised** — branch 4 again with `.replace(",", " ")` applied to the gold as
   well, for a gold that itself carries commas.

**Nothing in a result CSV records which scorer produced its `score` column.** Before comparing two
providers here, check which side of this split each one is on — a strict-scored model's number is
partly a formatting score. The same six-branch matcher, extracted and generalised, is the reference
implementation cited by `.claude/references/script-skeleton.md`.

## Layout

Reorganised 2026-08-29 from a flat directory to one folder per model.

```
bbh/
├── <task>.json          × 20      the data, shared — read by every runner
├── .env                            all provider keys
├── <model>/                        deepseek gemini gemma kimi llama openai qwen xai
│   ├── <model>_eval.py             the runner
│   ├── <model>_finish.py           gemma, qwen only — a re-run pass over an existing CSV
│   ├── <model>_eval_script.sh      the sbatch job
│   ├── <model>_<task>.csv          one per task: question, gold_answer, model_response, score
│   ├── <model>_overall_results.csv per-task means
│   ├── <model>_outlog / _errlog    that model's SLURM logs, no longer shared
│   └── _superseded/                an older duplicate kept rather than deleted — see below
└── bin/ · dotenv/ · python_dotenv-1.1.1.dist-info/   a pip install vendored in place
```

**Run it from the bbh root:** `sbatch qwen/qwen_eval_script.sh`. The `--output`/`--error` paths in
the sbatch are relative to the submit directory. The runners themselves resolve every path from
`__file__` (`MODEL_DIR`, `BBH_ROOT`), so their reads and writes do not depend on the cwd at all —
`kimi/kimi_outlog` is the record of what happens when they do: 20 splits, each
`No such file or directory: 'boolean_expressions.json'`, and an empty results file at the end.

## Traps

- **Two `.sh` files do not submit the runner their name implies.** `gemma_eval_script.sh` runs
  `gemma_finish.py` and `qwen_eval_script.sh` runs `qwen_finish.py`. The `_finish` twins differ in
  `max_tokens`, client timeout, retry behaviour and task list. Read the `.sh` before the runner.
- **The `splits` list in several runners is a narrowed re-run list** that was never restored —
  `deepseek` runs 5 tasks, `gemma_eval.py` 4, `kimi` 1, `openai` 14. A CSV on disk is not evidence
  the script beside it would regenerate it.
- **One failed call discards a whole task, in eight of ten runners.** A `None` response reaches
  `re.search` and raises `TypeError`; the per-split handler catches it, moves to the next split, and
  the rows collected so far are dropped without a CSV. `gemma_finish.py` and `qwen_finish.py` return
  `""` instead and are the exceptions.
- **`_superseded/` holds an older copy that was NOT deleted.** `gemma/` had been re-run into its own
  folder while stale flat copies of three tasks stayed at the root; the byte-identical duplicates
  were removed and the three genuinely older CSVs plus the old `gemma_overall_results.csv` (4 tasks
  vs the current 20) and `gemma_outlog` were parked there. `kimi/_superseded/` holds the 1-byte
  results stub and the failed-cwd-run logs. Read the current file, not these.
- **Result CSVs exist for eight prefixes; fewer are reported.** Some runs were abandoned and their
  CSVs were never deleted. Take coverage from `PLAN.md` / `Final_Result.xlsx`, never from `ls`.
- **No script sets `seed`, and `kimi` runs at `temperature=1`.** Nothing here is reproducible.
  Per-model generation parameters: `.claude/references/benchmarks/tasks/bbh-parameters.md`.
