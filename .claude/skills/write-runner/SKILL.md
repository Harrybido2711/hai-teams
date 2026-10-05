---
name: write-runner
description: Write or change the per-model runner for one model on one benchmark, up to a passing local --limit smoke test and a reviewer verdict. Use when adding a model to a benchmark, switching its provider, or changing a runner's decoding, retry, checkpoint or scoring code.
---

# write-runner

Phase 2 · run by `executor` (or the planner) · reviewed by `reviewer` · used by `new-runner`.
Ends **before** Quest: the user verifies the scripts, not you.

## Steps

1. **Retrieve.** `check_docs.py --model <model-id>` and `check_docs.py --brief <benchmark>`. A gap
   in `--model` (no invocation recipe, no parameter row) is filled first — in `model-calls.md` and
   `model-parameters.md` — not worked around in the runner.
2. **Copy the closest existing model folder of the same benchmark** and swap the client: `base_url`,
   key variable, model id, from `model-calls.md`. Keep the folder shape `<BENCH>_<Model>/`.
3. **Set every limit `model-parameters.md` requires** — thinking cap, output cap, seed — even when
   it equals the default. Reasoning visibility is resolved from the benchmark's README at run time.
4. **Import the scorer from the benchmark's shared core** (`*_eval_core.py`). If runners still carry
   their own, moving it into the core is the first change.
5. **Check the diff against the skeleton:** empty responses retried, every `except` logs and calls
   `halt_on_billing` first, a timeout that derives from `BaseException`, checkpoint keyed by a stable
   uid, a shard tag on every artefact, `PYTHONUNBUFFERED=1` in the sbatch script.
6. **Compile and smoke-test locally:** `python3 -m py_compile`, `bash -n`, then the runner with
   `--limit 3` (or the page's equivalent). Open the rows: non-empty response, parsed prediction,
   scored. Each parameter you set has now been accepted by one real call.
7. **Hand the diff to `reviewer`.**

## Done when

- The smoke rows are non-empty and scored — output pasted.
- `--model <id>` reports no gap.
- `reviewer` returns `STATUS: safe-to-run`.

## Never

- Run the full set locally — real runs execute on Quest.
- Drop a parameter because its name appeared in an error: a refused *value* is not a refused
  *parameter* (`model-parameters.md` rule 7).
- Put a model-specific prompt tweak into the shared builders.

## Detail

[script-skeleton.md](../../references/script-skeleton.md) ·
[model-calls.md](../../references/model-calls.md) ·
[model-parameters.md](../../references/model-parameters.md) ·
[provider-gotchas.md](../../references/provider-gotchas.md)
