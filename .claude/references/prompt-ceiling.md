# The prompt ceiling — capping a model that has no knob

Split out of [model-parameters.md](model-parameters.md), whose rule 5 sends you here. Rule 1 there is
unconditional: **hidden reasoning is always capped.** This page is what to do when the API offers
nothing to cap it with.

Some models expose no thinking parameter. **They are not exempt from rule 1** — the ceiling goes in
the prompt:

```text
Think briefly. Use at most <N> sentences of reasoning before your final line.
Then end your response with:
"Final Answer: <your concise answer here>"
```

Same for output length where `max_tokens` is unavailable or unsafe to set tightly.

- **A request, not a limit.** Verify from `usage.completion_tokens_details.reasoning_tokens`, never
  from how long the answer looks.
- **It changes the prompt, so it changes the measurement.** Record the wording with the score.

Which models this applies to is the last column of the table in
[model-parameters.md](model-parameters.md) — anything reading *No* or *not established*. **An
unestablished knob counts as no knob:** apply the ceiling and keep it until one call proves the API
can do it.

## When the ceiling does not bind: the forced answer

A request can be ignored, and then the output cap cuts the reply mid-thought and the row has no
answer. **The user's rule, 2026-11-12 and 11-13: a model that reaches its bound without an answer
is made to answer from what it had** ("如果到了那个上界依然没有想出来就直接回答"). A call that ends at
`length` gets one more turn: what it wrote — `reasoning_content`, or the visible text for a model
that deliberates in its answer — goes back as an assistant message, followed by *"Your thinking has
reached its limit. Do not reason any further: give your final answer now, in exactly the format the
task asks for."*, with thinking disabled where a switch exists and a small `max_tokens`.

- **It is part of the config** (`forced_answer`), and every forced row says so and keeps the first
  call's finish and token count — a forced answer is a different measurement from a free one.
- **For rows already on disk, call only those rows** (`--force-cut`), never the run again — the
  user, 2026-11-13. The stored file is copied aside first and the other rows change in `config` only.
- **It does not rescue a repetition loop**: handed its own loop, the model loops again.
- Where it exists and what each row carries: the runner's docstring — `MPGT_Deepseek`,
  `MPGT_Qwen_DeepInfra`, `PLAN_Deepseek`; counts and probes on those benchmarks' pages.
