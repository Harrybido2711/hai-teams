<!-- size-budget: 8000 -->
<!-- One invocation recipe per model in use; it grows when a model is adopted, which is the
     file working. Recipes are measured, so they carry the probe that established them. -->
# How to call each model

Which client, `base_url`, key, model id, and the non-optional parameters. Limits a runner must set:
[model-parameters.md](model-parameters.md). How each client fails:
[provider-gotchas.md](provider-gotchas.md).

Recipes are *measured* — from a runner that has run, or from a probe. A documented shape is a
hypothesis until one call confirms it.

## `google/gemma-4-31B-it` — DeepInfra

```python
OpenAI(api_key=os.getenv("DEEPINFRA_API_KEY"),
       base_url="https://api.deepinfra.com/v1/openai", timeout=300)
# .chat.completions.create(model="google/gemma-4-31B-it", messages=…, temperature=0, max_tokens=…)
```

Same client shape as DeepSeek and GPT; only `base_url` and the key differ.
· `NEG_Gemma_DeepInfra/gemma_di_neg_eval.py:33-36, 103-110, 171`

**Pass no reasoning parameter here.** DeepInfra serves this checkpoint with thinking already off —
measured 15–17 token answers, no `<think>` tags. It *accepts* `reasoning_effort`, and passing it
turns thinking back **on**. Adding a knob helpfully is a regression.

**The same checkpoint on Together is the opposite** — reasoning on by default, off via
`reasoning={"enabled": False}`. Measured A/B, 12 calls an arm: on gave 6/12 completions at 9.3 s and
517 output tokens, off 11/12 at 0.3 s and 16. Capping `max_tokens` is no substitute — every call that
returned stopped on its own. · `NEG_Gemma/gemma_neg_eval.py:21-53`

Across both: **SIGALRM at 120 s is the primary guard**, below the 300 s socket timeout — Together
ignores `timeout=` outright. And **tolerate four `<think>` spellings**, not one.

## `gemini-3.5-flash-lite` — two routes, both probed; **OpenRouter is the settled one**

**Both run end to end.** OpenRouter completed 400/400 in 19:27, 0 empty; the native route 2.76 s an
item against OpenRouter's 2.92 — the same speed once its calls succeed. **Thinking spends zero.**

**The six-item probe under-counted the real prompt by half.** It saw 192/18 tokens and gave
$0.0001032 an item, "about four cents" for 400. Metered over a real run (n=140), EmoBench items are
**379 prompt / 31 output**, so 400 items cost **≈$0.077** at $0.30/M in and $2.50/M out. Probe a
real item, not a toy one.

**Which thinking field exists depends on the installed SDK, and Quest's is not yours** —
[provider-gotchas.md](provider-gotchas.md).

**OpenRouter switches backend mid-run.** It serves this model from Google AI Studio *and* Google
Vertex (US) with failover: over a finished 400-item run `provider` came back `Google AI Studio` 110
times and `Google` 90, on one task alone. Record which answered — the response carries `provider` and
a per-call `usage.cost`. **Both routes take `seed`** and are reproducible with one — rule 6,
[model-parameters.md](model-parameters.md). Untested across a backend switch.

**Route A · Google AI Studio, native SDK.** `client.models.generate_content(model=, contents=,
config=types.GenerateContentConfig(system_instruction=, thinking_config=types.ThinkingConfig(
thinking_budget=128), max_output_tokens=…))` — the shape every runner here uses. **Use
`thinking_budget`, not `thinking_level`: Quest's SDK has no such field.**

- **`minimal` is the floor — there is no off**, and the response carries a `thoughtSignature` either
  way. On a structured task thinking already costs nothing, so a prompt ceiling is only needed where
  it actually fires.
- **Omit `temperature`, `top_p`, `top_k`** — 3.x guidance is to remove, not tune.
- The key is an `AQ.` auth key. Those are *reported* to 401 here, and **this project's key does
  not** — probed, HTTP 200. Check per key; it is not a property of AQ. keys.

**Route B · OpenRouter.** Same client shape as DeepInfra and DeepSeek:

```python
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
client.chat.completions.create(model="google/gemini-3.5-flash-lite", messages=messages,
                               extra_body={"reasoning": {"effort": "minimal"}})
```

`Authorization: Bearer sk-or-v1-…`. `reasoning.effort` maps to the thinking level. 1,048,576 context,
65,536 max output, **$0.30/M in and $2.50/M out**, thinking billed at the output rate. Not
`gemini-3.5-flash`: a dearer model at $1.50/M and $9.00/M, defaulting to `medium`.

## `gpt-5.6-luna` — OpenAI platform

```python
OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=300)   # no base_url override
# .chat.completions.create(model="gpt-5.6-luna", messages=…,
#                          max_completion_tokens=…, reasoning_effort="low", seed=42)
```

· Source: `EMO_GPT_5.6_Luna/gpt56luna_emo_eval.py`

- **`max_tokens` is refused** — it must be `max_completion_tokens`, and the error names the
  replacement. So are `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `logprobs`, and any
  `temperature` but the default. What the model page claims and what it accepts differ on six counts;
  the limits table in [model-parameters.md](model-parameters.md) records the probe.
- **Negotiate the surface at startup rather than hardcoding it.** The runner spends one call finding
  what is accepted, prints it, and writes it onto every row — because a rejected parameter is
  permanent, and discovering it per item costs the run.
- **A refused *value* is not a refused *parameter*.** `reasoning_effort="minimal"` is rejected while
  the parameter itself works; dropping it there runs the benchmark at the default `medium`, uncapped.
- Rate limits on this key, from the response headers: **5,000 RPM, 2,000,000 TPM**. RPD is *not* in
  the headers and is unestablished — it was the constraint that broke DocVQA at 10 shards
  ([quest-cluster.md](quest-cluster.md)).

## `deepseek-reasoner`

```python
OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=7200)
# .chat.completions.create(model=…, messages=…, temperature=0, max_tokens=8192)
```

· `NEG_Deepseek/deepseek_neg_eval.py:16-17, 29-40`

- **The long timeout is not carelessness.** DeepSeek queues; the short timeouts used elsewhere here
  would classify a queued call as a hang.
- **The answer sometimes arrives in `reasoning_content` with `content` empty.** A runner reading only
  `message.content` scores those as failures. Fall back explicitly, and log when it fires.
- **`deepseek-reasoner` is an alias, not a model id.** What it resolves to can change under you;
  record the resolved identity with the results.

No reasoning knob — the levers are in [model-parameters.md](model-parameters.md).
