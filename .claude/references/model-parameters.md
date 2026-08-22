# Model parameters — what every runner must set

**Applies to every benchmark, not one.** Keyed by model, because the same providers are called from
NegotiationToM, EmoBench, bbh, mmlu and DocVQA. Read this before writing or changing any runner.

## The standing rule

Set by the user on 2026-08-22:

1. **A reasoning-capable model gets an explicit thinking cap.** Never left to the model's discretion.
2. **A model that does not reason gets an explicit output cap.** `max_tokens` on every call.
3. **Every model is asked to show its reasoning, with the same prompt.** That contract does not vary
   by model, and it is not a cost lever — cutting it trims the cheap side and changes the measurement.
4. **Set the value even when it equals the default.** A default belongs to the provider and can move
   without notice; a pinned value records what the run actually used.
5. **When a limit cannot be set through the API, set it in the prompt.** The table's last column says
   which models that is.

## When the knob does not exist, the prompt is the knob

Several models cannot have reasoning switched off, and the last column of the table names each one.
**They are not exempt from rule 1.** Where the API offers nothing, the ceiling goes into the prompt:

```text
Think briefly. Use at most <N> sentences of reasoning before your final line.
Then end your response with:
"Final Answer: <your concise answer here>"
```

The same applies to output length wherever `max_tokens` is unavailable or unsafe to set tightly.

- **It is a request, not a limit.** Nothing enforces it, so verify from
  `usage.completion_tokens_details.reasoning_tokens`, never from how long the answer looks.
- **Still better than the alternative**, which is a hard `max_tokens` sized to what you *want* to
  pay: that truncates mid-thought into a **billed empty response**, losing the money and the row.
- **It changes the prompt, so it changes the measurement.** A prompt-limited run and an unlimited one
  must not share a results table. Record the wording with the score.

## Per model

| Model | Called by | Reasoning? | Thinking cap | Output cap | Can it be lowered? If not → prompt |
|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | EmoBench (planned) | yes, at the floor | `thinking_level="minimal"` | `max_output_tokens` | **No — `minimal` is the floor**, "as close as possible to a zero budget… but still requires thought signatures". **Prompt ceiling on thinking and on output.** Thinking bills at $2.50/M |
| `gemini-3.5-flash` | not used | yes | `thinking_level`, default `medium` | `max_output_tokens` | **Yes** — `medium` → `minimal`. Measure the step; do not assume it is free |
| `gemini-2.5-flash` | EmoBench, bbh, NegotiationToM, DocVQA | yes | thinking budget | `max_output_tokens` | **Not established** for this generation → treat as no. **Prompt ceiling** until a call proves otherwise |
| `grok-3-mini` | bbh, EmoBench, NegotiationToM | yes | `reasoning_effort` | provider default | **No — reasoning is the model.** **Prompt ceiling on thinking and on output** |
| `deepseek-reasoner` | bbh, EmoBench, NegotiationToM | yes | none exposed | `max_tokens` | **No knob at all.** **Prompt ceiling**, or change model |
| `Qwen/Qwen3.5-9B` | bbh, EmoBench, NegotiationToM | hybrid | `reasoning={"enabled": False}` | `max_tokens` | **Yes** — measured off: 11/12 vs 6/12 completions, 16 vs 517 output tokens |
| `kimi-k2.5` | bbh | not established | verify before assuming | `max_tokens` | **Not established → treat as no.** Prompt ceiling until verified |
| `gpt-4o-mini-2024-07-18` | bbh, DocVQA | no | n/a | `max_tokens` | n/a. `max_tokens` is enforced, so no prompt fallback is needed |
| `google/gemma-4-31B-it` | bbh, EmoBench, NegotiationToM | no as served | n/a | `max_tokens` | n/a **on DeepInfra, which serves it with thinking already off — passing `reasoning_effort` turns it back on.** On Together it is on by default and must be disabled |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | bbh | no | n/a | `max_tokens` | n/a. `max_tokens` is enforced |

"Called by" lists where the model id appears in a runner today; it is not a claim that every one of
those runs is current. `gemini-3.5-flash` is listed only so it is not mistaken for Flash-Lite — it is
a different, dearer model at $1.50/M input and $9.00/M output.

**An unestablished knob counts as no knob.** Where the last column says *not established*, apply the
prompt ceiling and keep it until one call proves the API can do it. The failure this avoids is
assuming a knob exists, setting nothing, and finding out from the bill.

## Choosing the number

**Measure, do not reason from the parameter name.** Run one small slice at two or three settings and
compare accuracy and cost together. How each knob behaves and the four ways capping backfires:
[reasoning-cost.md](reasoning-cost.md) — read it before choosing a value, not after.

**Nothing in this table has been applied to a runner yet.** What each runner currently sets is
recorded on its benchmark's page; this file is what they must be changed to.
