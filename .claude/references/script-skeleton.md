# Eval script skeleton

The order below is the one the six NegotiationToM runners follow. Copy the closest existing folder
and swap the client rather than starting from this list.

**1. Paths and client.** Resolve from the script's own location, never from cwd:

```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # benchmark root
load_dotenv(os.path.join(ROOT, ".env"))
client = Together(api_key=os.getenv("TOGETHER_API_KEY"), timeout=180)  # see provider-gotchas.md
```

**2. Prompt builders.** System prompt from the benchmark's `src/configs/prompts.yaml` and
`response.yaml`; choices as a lettered menu; answer requested as JSON.

**3. `parse_json(text)`.** Accept bare JSON and ```` ```json ```` fenced output; return `None` on
failure.

**4. `call_api(messages, model, max_retries=3)` — the shared retry contract.**

- up to 3 attempts, `temperature=0`, and an explicit `max_tokens` chosen from what the model
  actually needs (see [provider-gotchas.md](provider-gotchas.md)) rather than habit
- **retry on empty-string responses, not just exceptions** — HTTP 200 with an empty body is the
  most common failure here and raises nothing:
  ```python
  content = (resp.choices[0].message.content or "").strip()
  if not content: time.sleep(5); continue
  ```
- dynamic backoff from the provider's own message:
  `re.search(r'try again in ([\d.]+)(ms|s)', err)`, else 5s
- **every `except` block calls `halt_on_billing(error, model, SCRIPT_DIR)` first** — see
  [provider-gotchas.md](provider-gotchas.md) for why the narrow string test is not enough
- `time.sleep(2.0)` after every success
- **log the exception** in every `except` block. Without it a `TypeError` from a bad call signature
  is retried as if it were a network fault, then scores 0, with nothing in the SLURM log

**5. `call_and_parse()` — second retry layer.** If `parse_json()` returns `None` on a *non-empty*
response, re-issue the call, up to 3 times. Call sites use this, not bare `call_api` + `parse_json`.

**6. Checkpoint / resume.** `.jsonl` keyed by a stable UID (`qid`, or
`"<dialogue_id>_<agent>_<task>"`). On start, load completed UIDs and skip them. `--save-every 20`.
Always persist the raw response so failures can be inspected.

> **Check for stale checkpoints before every full run.** Resume skips any UID already present, so
> leftovers from an older code version make a run "succeed" in seconds while emitting old, wrong
> data. NEG_GPT held 14,280 such rows. Archive to a timestamped directory; never delete outright.
>
> Rows written *empty* are skipped forever by a plain resume too, because `load_checkpoint` adds
> every uid to the done set regardless of whether its response was usable. Prune before resubmitting.

**7. Normalisation — copy the bbh approach.** Generous about how an answer is written, strict about
what it says. `bbh/xai_eval.py::score_response` is the reference; the extracted version is
`neg_eval_core.py::clean_surface`:

```python
text = value.strip().strip("\"'`").strip()   # quotes and backticks are packaging
text = re.sub(r"\s+", " ", text)             # collapse whitespace runs
text = text.strip(" .,;:!")                  # trailing punctuation
return ALIASES.get(text.lower(), text.title())
```

Add aliases for what the models **actually emit** — check a pilot rather than guessing. Measured
here: GPT returned lowercase items 486 times (17% of its values), plus `Wood` for `Firewood` and
`unknown`/`N/A` for `Not Given`. Check both `"high"` and `"High"` key casings.
**Normalise model output only, never gold labels** — gold is canonical, and rewriting it changes
the answer key.

**8. `evaluate()`.** Per-sample CSV plus an overall CSV. Sanitize model names with
`.replace(".", "_").replace("/", "-")`.

**9. Entry point.** `argparse` with `--model`, `--task all`, `--save-every 20`.

## Invariants a change is checked against

These are settled; a change that breaks one is wrong unless it argues otherwise explicitly.

| Invariant | Why |
|---|---|
| Gold labels are never rewritten — normalisation applies to model output only | rewriting gold changes the answer key |
| Shard-level and merged metrics call the **same** scoring function | two implementations drift silently |
| An empty API response is retryable, not a scored zero | HTTP 200 + empty body raises nothing |
| Every `except` block logs the exception | otherwise a `TypeError` is retried as a network fault and scores 0 |
| A timeout exception derives from `BaseException` | `except Exception` in each runner would swallow it |
| Job scripts `export PYTHONUNBUFFERED=1` | or the log is empty while the job runs |
| Shard outputs carry a shard tag | untagged `_overall.csv` files overwrite each other |
| Model-specific prompt tweaks stay in that model's runner | the shared builders must serve all six identically |
| NegotiationToM intention rows total **4,618**, not 4,760 | odd-length dialogues have one target, not two |
| 156 unannotated rows each are excluded from desire and belief | they have no correct answer |
