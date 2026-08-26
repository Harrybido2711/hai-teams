<!-- size-budget: 6000 -->
<!-- One row per provider plus its failure modes; it grows when a provider is added or
     replaced, which is the file working, not the file sprawling. -->
# Provider gotchas

Every one of these has cost a debugging cycle, and most fail with **HTTP 200 and no exception** —
the run looks complete while scoring 0.

| Provider | Client | Must do |
|---|---|---|
| OpenAI `gpt-4o-mini` | `openai.OpenAI` | baseline |
| DeepSeek `deepseek-v4-flash` | `openai.OpenAI`, `base_url="https://api.deepseek.com"`, `timeout=7200` | legacy `deepseek-reasoner` retired 2026-07-24; pass `extra_body={"thinking":{"type":"disabled"}}` for this classification benchmark |
| Gemini `gemini-3.5-flash-lite` **via OpenRouter** | `openai.OpenAI`, `base_url="https://openrouter.ai/api/v1"` | the project's Gemini since 2026-08-23. `extra_body={"reasoning":{"effort":"minimal"}}`; a seed alone does not reproduce, the backend decides (below) |
| ~~Gemini `gemini-2.5-flash`~~ *(superseded, still in the bbh/NegToM/DocVQA runners)* | `google.genai.Client` | no `system` role in messages; `thinking_budget=0`; do **not** set `max_output_tokens` (256 truncated JSON mid-object) |
| xAI `grok-3-mini` | `xai_sdk.Client` | no message dicts — `chat.create(model=...)`, `chat.append(xai_system(...))`, `chat.append(xai_user(...))`, `chat.sample()`. It does accept `max_tokens`/`temperature` |
| Qwen `Qwen/Qwen3.5-9B` | `together.Together`, **`timeout=180`** | hybrid model: pass `reasoning={"enabled": False}`; retain `max_tokens=8192` for visible JSON headroom |
| Gemma `google/gemma-4-31B-it` | `together.Together`, **`timeout=300`** | intermittent empty string at HTTP 200 — retry up to 5× |

## Timeouts, and why `timeout=` is not a guard

**Never inherit `timeout=18000` from the EmoBench scripts.** Five hours per request means one hung
call stalls the job: Gemma sat inside a single request for over two hours with SLURM reporting
RUNNING and an empty log.

Worse, **Together's client did not honour `timeout=` at all** — lowering it to 300 still produced a
~70-minute hang. The reliable guard is a SIGALRM watchdog (420s) — implemented for NegotiationToM as
`neg_eval_core.py::guarded_call`, and needed in equivalent form by any new runner
that interrupts the blocking read whatever the library does.

`CallTimeout` **must derive from `BaseException`**: as an `Exception` it is caught by each runner's
own `except Exception`, and since the alarm has already fired and been cleared, the rest of that
`call_api` runs unprotected — the guard evaporates after one use.

## Qwen's unstable reasoning

Thinking length varied from ~700 to past 32,768 output tokens for the same prompt at
`temperature=0`; one pilot produced 60 rows in 7 hours. Qwen3.5-9B is a Together hybrid model, so
the shipped fix is the provider control `reasoning={"enabled": False}`, not prompt wording or a
larger token budget. **Keep the shared prompt identical across models.** Log `finish_reason` and
usage on every empty response, including billed tokens from failed attempts.

## google-genai: the thinking field depends on the installed SDK

**Verify against the interpreter that will run the job, not the one you tested on.** google-genai
**2.19** accepts `types.ThinkingConfig(thinking_level="minimal")`. **Quest's is 1.49.0**, whose
`ThinkingConfig` exposes only `include_thoughts` and `thinking_budget` — sending `thinking_level`
there raises a pydantic **`ValidationError` on every call**.

- **Select the field from `ThinkingConfig.model_fields`; do not try one and catch an exception.** The
  EmoBench runner guarded this and caught `TypeError`. google-genai is pydantic, so the guard never
  fired, the error reached the generic retry handler, and a permanent config error was retried three
  times an item — 20 rows, all empty, 26 s each, before anyone looked. 2026-08-23.
- **A malformed request must be fatal.** `INVALID_ARGUMENT`, `Extra inputs are not permitted` and
  `ValidationError` are not transient; retrying them spends the run to learn one fact.
- On `gemini-3.5-flash-lite`, `thinking_budget=0` is rejected **400 INVALID_ARGUMENT** — thinking
  cannot be switched off.
- **`thinking_budget` is a request, not a ceiling.** With it set to 128 over a 400-item EmoBench run,
  48 items thought anyway and 25 went past the budget, one to 532 tokens. A six-item probe had shown
  zero and was simply too small to see it. OpenRouter's `reasoning.effort="minimal"` on the *same
  model* produced zero across all 400 — so the two routes are not the same condition, and a run
  capped this way still needs its thinking tokens summed from the rows rather than assumed.

## Health checks must use real prompts

A synthetic probe (system `"You are a JSON API."`) was rejected by grok with
`PERMISSION_DENIED / SAFETY_CHECK_TYPE_BIO` while the genuine eval prompts passed. `preflight.py`
builds its probe from the real prompt builders — use it rather than writing a new probe script.

## Classifying a refusal

Every `except` block must call `halt_on_billing(error, model, SCRIPT_DIR)` first. It is the shared
classifier in `neg_eval_core.py`: a billing refusal or an exhausted *daily* cap stops the run at the
first occurrence and writes `BILLING_HALT.txt` / `QUOTA_HALT.txt`; everything else returns and the
normal retry continues.

Do not hand-roll `if "insufficient_quota" in text` — that narrow test is exactly what let xAI write
9,378 empty rows, because its wording is "used all available credits or reached its monthly spending
limit".
