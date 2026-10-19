<!-- size-budget: 9600 -->
<!-- One row per provider plus its failure modes; it grows when a provider is added or
     replaced, which is the file working, not the file sprawling. Raised to 9000 on 2026-10-19 for
     the OpenRouter section the judge-pilot probes produced. -->
# Provider gotchas

Every one of these has cost a debugging cycle, and most fail with **HTTP 200 and no exception** —
the run looks complete while scoring 0.

| Provider | Client | Must do |
|---|---|---|
| OpenAI `gpt-4o-mini` | `openai.OpenAI` | baseline |
| DeepSeek `deepseek-v4-flash` | `openai.OpenAI`, `base_url="https://api.deepseek.com"`, `timeout=7200` | legacy `deepseek-reasoner` retired 2026-07-24; pass `extra_body={"thinking":{"type":"disabled"}}` for this classification benchmark |
| Gemini `gemini-3.5-flash-lite` **via OpenRouter** | `openai.OpenAI`, `base_url="https://openrouter.ai/api/v1"` | the project's Gemini since 2026-08-23. `extra_body={"reasoning":{"effort":"minimal"}}`; a seed alone does not reproduce, the backend decides (below) |
| ~~Gemini `gemini-2.5-flash`~~ *(superseded, still in the bbh/NegToM/DocVQA runners)* | `google.genai.Client` | no `system` role in messages; `thinking_budget=0`; do **not** set `max_output_tokens` (256 truncated JSON mid-object) |
| xAI `grok-3-mini` | `xai_sdk.Client` | no message dicts — `chat.create(model=...)`, `chat.append(xai_system(...))`, `chat.append(xai_user(...))`, `chat.sample()`. It does accept `max_tokens`/`temperature` | **RETIRED — xAI serves neither this id nor any alias of it, checked 2026-09-08 against `api.x.ai/v1/language-models`; the stored results stay valid but a new run fails. What it serves now: grok-4.3/4.5/4.6/4.20/build-0.1, all text+image.**
| Qwen `Qwen/Qwen3.5-9B` | `together.Together`, **`timeout=180`** | hybrid model: pass `reasoning={"enabled": False}`; retain `max_tokens=8192` for visible JSON headroom |
| Gemma `google/gemma-4-31B-it` | `together.Together`, **`timeout=300`** | intermittent empty string at HTTP 200 — retry up to 5× |

## OpenRouter: dropped settings, and refusals that look like empties

Probed 2026-10-19 on the four candidate judges (`Action_processes_benchmarks/judge_core.py`):

- **A setting no backend supports is silently dropped.** `temperature=0` to Claude Haiku 5.5 returns
  200 and an answer, while the model page lists no `temperature`. With
  `extra_body={"provider": {"require_parameters": True}}` the same request is refused 404 "No
  endpoints found that can handle the requested parameters" — so send that on every call whose
  settings are part of the result, and treat the 404 as a fatal config error.
- **A content-filter refusal is HTTP 200 with `content: null`.** `finish_reason: "content_filter"`,
  `native_finish_reason: "refusal"`, the reason in `message.refusal`, sometimes no `usage`. A loop
  that only reads `content` sees an empty body and retries it as transient; read `message.refusal`
  and record the row as refused. Sonnet 5.5 does this on every Wonderbread SOP judge call (no
  `usage`, unbilled) and on 11 of 81 MultiChallenge judge calls (billed) — the same prompt refused
  on one replicate and answered on the next, so it is per call, not per item.
- **One model id is many backends, at different precisions.** Kimi K2.6 is served by 17 providers
  at int4, fp4, fp8 and bf16, and one unpinned run used two. Pin with `"provider": {"only":
  ["<name>"], "allow_fallbacks": False}` wherever the comparison is between models; the endpoint
  list with each provider's quantization is `GET /api/v1/models/<id>/endpoints`. The maker's own
  endpoint is not automatically the safe pin: Moonshot AI's returns whitespace in JSON mode.
- **Claude fences JSON mode** — `response_format={"type": "json_object"}` comes back inside
  ```` ```json … ``` ````; a bare `json.loads` raises. Strip fences before parsing.

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
- **`MALFORMED_RESPONSE` is a real finish reason and it is deterministic.** On DocVQA,
  2026-09-10, 5 of 5,349 page images came back HTTP 200 with thought tokens spent (83–267) and
  **no output part at all**. Re-asking the same five at the same seed reproduces it exactly, so
  it is a property of those inputs, not a transient — retrying is wasted. Two things follow:
  google-genai 2.22 does not know the enum (`MALFORMED_RESPONSE is not a valid FinishReason`),
  so read it as a string; and **an empty string at HTTP 200 raises nothing**, so a retry helper
  that only logs exceptions leaves no trace of it anywhere.
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

**OpenRouter's out-of-credit 402 has two wordings, in this order.** First "… would exceed your
available credits given your current in-flight requests. Retry after …" (`in_flight_budget_exhausted`,
`Retry-After: 120`) — the balance covers some calls but not every reservation in flight, since each
reserves `max_tokens` × price. Then "… requires more credits, or fewer max_tokens … can only afford
N" once it cannot cover one. `neg_eval_core.py` and `judge_core.py` retry the first and halt on the
second, which stopped judge job 4482838 cleanly (2026-10-19, no failed rows); `aware_eval_core.py`
matches neither and stopped only at `FAILURE_HALT` (2026-10-12). Before a run whose estimate nears
the balance: `GET https://openrouter.ai/api/v1/credits` (`total_credits`, `total_usage`).
