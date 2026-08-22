# Provider gotchas

Every one of these has cost a debugging cycle, and most fail with **HTTP 200 and no exception** —
the run looks complete while scoring 0.

| Provider | Client | Must do |
|---|---|---|
| OpenAI `gpt-4o-mini` | `openai.OpenAI` | baseline |
| DeepSeek `deepseek-v4-flash` | `openai.OpenAI`, `base_url="https://api.deepseek.com"`, `timeout=7200` | legacy `deepseek-reasoner` retired 2026-07-24; pass `extra_body={"thinking":{"type":"disabled"}}` for this classification benchmark |
| Gemini `gemini-2.5-flash` | `google.genai.Client` | no `system` role in messages; use `thinking_budget=0`; do **not** set `max_output_tokens` (256 truncated JSON mid-object) |
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
