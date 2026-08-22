# Capping what reasoning costs

A reasoning model bills its internal thinking as output tokens, and on this project's own usage that
line dominates everything else — one day's bill read **$16.99 reasoning against $0.41 completion**,
a factor of 41. The visible answer is nearly free; the thinking is the entire cost.

## The knob, per client

| Client | Parameter | Values | Off entirely? |
|---|---|---|---|
| OpenAI reasoning models | `reasoning_effort` | `none` · `minimal` · `low` · `medium` · `high` · `xhigh` · `max`, **model-dependent** | `none`, where supported |
| Gemini 3.x native | `thinking_level` | `minimal` · `low` · `medium` · `high`. Replaces `thinking_budget`; the two cannot both be sent | no — `minimal` is the floor |
| OpenRouter (any model) | `reasoning: {effort, max_tokens, exclude, enabled}` | `effort` as OpenAI; `max_tokens` numeric | `effort: "none"` |
| Together hybrid models | `reasoning={"enabled": False}` | on/off | yes — this project's shipped fix for Qwen |
| xAI, DeepSeek reasoner | reasoning is the model | — | no. Choose a non-reasoning model instead |

`gpt-4o-mini`, `gemma-*` and `llama-*` are not reasoning models and contribute no reasoning tokens.
A reasoning line on the bill therefore names which model produced it.

## Four ways capping goes wrong

- **`max_output_tokens` includes reasoning.** It does bound the spend, but when thinking exhausts the
  budget the response comes back with `status: "incomplete"` and
  `incomplete_details.reason: "max_output_tokens"` — **input and reasoning are billed and there is no
  visible answer**. A cap set too low converts cost into empty rows, which is the more expensive
  failure because the run has to be repeated. OpenAI's own guidance is to reserve **at least 25,000
  tokens** for reasoning plus output while calibrating.
- **`exclude: true` is not a saving.** OpenRouter still bills excluded reasoning — "reasoning tokens
  are considered output tokens and charged accordingly". It hides the trace; it does not stop the
  work. Use `effort: "none"` to actually stop it.
- **`reasoning_effort` and a token cap can conflict.** Sending `reasoning_effort: "none"` together
  with `max_completion_tokens` is reported to make the API ignore the effort setting, so the cap
  becomes the only control and the model reasons anyway. Send one, verify with the usage numbers, and
  do not assume both applied.
- **A cap changes the measurement.** Less thinking lowers accuracy on exactly the hard items a
  benchmark exists to test, so a capped run and an uncapped run are not comparable and must not share
  a results table. Record the setting beside the score.

## Keeping thinking on but bounded

A level names an intention; **a numeric budget names a ceiling**, and only the ceiling bounds the
bill. Measured on `gemini-3.5-flash-lite`, 2026-08-22, same prompt:

| Setting | Thinking tokens spent |
|---|---|
| no thinking config | 0 |
| `thinking_budget=256` (native) | **127** |
| `thinking_level="high"` | 315 |
| `reasoning={"max_tokens":256}` (OpenRouter) | **258** |
| `reasoning={"effort":"high"}` | 397 |

The budget is honoured on both routes. `thinking_budget` and `thinking_level` are **mutually
exclusive** — sending both is an error, so choosing the budget means giving up the level.

**A cap this size has not cost accuracy here.** On NegotiationToM, 15 real belief items scored 7/15
both with thinking off and with a 512-token budget, while billed output fell from 396 to 15 tokens
per call. That is the measurement to repeat before trusting a budget on a new benchmark, not a
result to assume.

## Record what the thinking said

The expensive half of every response is invisible in the results unless it is stored.

- **Native Gemini**: `include_thoughts=True`, then read the parts and keep those with
  `part.thought` set. **Do not use `resp.text`** — with thoughts on it concatenates the summary with
  the answer, which feeds reasoning prose into `json.loads` and fails every parse.
- **OpenRouter**: the reasoning comes back as readable text in `message.reasoning`, separate from
  `content`, so it cannot contaminate the answer. It is only present if asked for.

Neither changes what is billed. `exclude: true` hides the trace and still charges for it.

## What to set

The standing rule, and the per-model knob and cap for every model this project calls, are in
[model-parameters.md](model-parameters.md). This file is the mechanism behind those choices; that one
is the decision. When a model has no knob at all, the ceiling goes in the prompt — that fallback and
its three caveats are on the same page.

**Measure, do not reason from the parameter name.** Run one small slice at two or three settings and
compare accuracy and cost together. Read the cost from
`usage.completion_tokens_details.reasoning_tokens`, not from the response length — the expensive
tokens are the ones you never see.
