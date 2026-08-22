# Capping what reasoning costs

A reasoning model bills its thinking as output tokens, and on this project's own usage that line
dominates — one day read **$16.99 reasoning against $0.41 completion**, a factor of 41. Confirmed
across the finished NegotiationToM runs: uncapped models averaged 466 and 567 output tokens per
call against 14–15 for the capped ones, on a task whose visible answer is about 15 tokens.

## The knob, per client

Which parameter each model takes, and which models expose none, is the table in
[model-parameters.md](model-parameters.md). `gpt-4o-mini`, `gemma-*` and `llama-*` are not reasoning
models and contribute no reasoning tokens at all, so a reasoning line on the bill names its own
source.

## Four ways capping goes wrong

- **`max_output_tokens` includes reasoning.** It bounds the spend, but when thinking exhausts the
  budget the response returns `status: "incomplete"`, `reason: "max_output_tokens"` — **input and
  reasoning are billed and there is no visible answer**. Too low a cap turns cost into empty rows,
  which is worse because the run repeats. OpenAI advises reserving **25,000 tokens** while
  calibrating.
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

The budget is honoured on both routes, and `thinking_budget` and `thinking_level` are **mutually
exclusive** — choosing one gives up the other.

**A cap this size has not cost accuracy here.** On NegotiationToM, 15 belief items scored 7/15 both
with thinking off and with a 512-token budget, while billed output fell from 396 to 15 per call.
Repeat that measurement on a new benchmark rather than assuming it.

## Two ceilings rather than one

Set both where a run must not overspend: the API level or budget **and** a sentence limit in the
prompt. They fail differently — the API cap is enforced but coarse, the prompt cap precise but only
a request — so neither covers the other. The EmoBench flash-lite runners use `minimal` plus a
two-sentence limit.

**The prompt half changes the measurement.** Keep the unmodified prompt behind a flag so the
comparable run stays possible, and record the condition on each row, not on the run.

## Record what the thinking said

The expensive half of every response is invisible in the results unless it is stored.

- **Native Gemini**: `include_thoughts=True`, then read the parts with `part.thought` set. **Not
  `resp.text`** — with thoughts on it concatenates the summary with the answer, feeding reasoning
  prose into `json.loads` and failing every parse.
- **OpenRouter**: readable text in `message.reasoning`, separate from `content`, so it cannot
  contaminate the answer. Present only if asked for.

Neither changes what is billed; `exclude: true` hides the trace and still charges for it.

## What to set

The standing rule, and the per-model knob and cap for every model this project calls, are in
[model-parameters.md](model-parameters.md). This file is the mechanism behind those choices; that one
is the decision. When a model has no knob at all, the ceiling goes in the prompt — that fallback and
its three caveats are on the same page.

**Measure, do not reason from the parameter name.** Run one small slice at two or three settings and
compare accuracy and cost together. Read the cost from
`usage.completion_tokens_details.reasoning_tokens`, not from the response length — the expensive
tokens are the ones you never see.
