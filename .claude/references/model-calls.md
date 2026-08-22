# How to call each model

Which client, `base_url`, key, model id, and the parameters that are not optional. What limits a
runner must set is [model-parameters.md](model-parameters.md); how each client fails is
[provider-gotchas.md](provider-gotchas.md).

Recipes for models this project has run are taken from that runner and are *measured*. The rest come
from documentation and say so — a documented shape is a hypothesis until one call confirms it.

## `google/gemma-4-31B-it` — DeepInfra

```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("DEEPINFRA_API_KEY"),
                base_url="https://api.deepinfra.com/v1/openai",
                timeout=300)
client.chat.completions.create(model="google/gemma-4-31B-it",
                               messages=messages, temperature=0, max_tokens=MAX_TOKENS)
```

Same client shape as DeepSeek and GPT; only `base_url` and the key differ.
· Source: `NEG_Gemma_DeepInfra/gemma_di_neg_eval.py:33-36, 103-110, 171`

**Pass no reasoning parameter here.** DeepInfra serves this checkpoint with thinking already off —
measured 15–17 token answers, no `<think>` tags. It *accepts* `reasoning_effort`, and passing it
turns thinking back **on**. The one reasoning-capable model here with no knob to set; adding one
helpfully is a regression.

**The same checkpoint on Together behaves oppositely** — reasoning is on by default and must be
switched off with `reasoning={"enabled": False}`. Measured A/B, 12 calls per arm on production
prompts with `max_tokens` held constant: reasoning on gave 6/12 completions at 9.3 s and 517 output
tokens; off gave 11/12 at 0.3 s and 16 tokens. Capping `max_tokens` does not substitute — every call
that returned stopped on its own. · Source: `NEG_Gemma/gemma_neg_eval.py:21-53`

Two things carry across both: **SIGALRM at 120 s is the primary guard**, deliberately below the
300 s socket timeout so a hang is classified identically on either provider — Together ignores its
`timeout=` argument outright. And **tolerate four `<think>` spellings when parsing**, not one: a
missing opening tag and a pipe-delimited variant both occur alongside the documented shape.

## `gemini-3.5-flash` — two routes, neither used here yet

**Not called by any runner in this repo.** Everything below is from documentation and must be
confirmed by one call before a run depends on it.

**Route A · Google AI Studio, native SDK.** Two API surfaces exist in `google-genai` and the docs for
3.5 use the newer one:

```python
client.interactions.create(model="gemini-3.5-flash", input=prompt,
                           generation_config={"thinking_level": "medium"})
```

Every existing runner here uses the older `client.models.generate_content(model=, contents=,
config=types.GenerateContentConfig(system_instruction=, …))`. **Check which the installed SDK
supports before writing** — do not assume the repo's shape still applies.

- `thinking_level`: `minimal` · `low` · `medium` (default) · `high`.
- **Omit `temperature`, `top_p` and `top_k`** — the 3.x guidance is to remove them, not tune them.
- The key is an `AQ.` auth key, and those are reported to 401 against
  `generativelanguage.googleapis.com` with `ACCESS_TOKEN_TYPE_UNSUPPORTED`. **Probe this route before
  committing to it** — see [provider-gotchas.md](provider-gotchas.md).

**Route B · OpenRouter, OpenAI-compatible.** The same client shape as DeepInfra and DeepSeek:

```python
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
client.chat.completions.create(model="google/gemini-3.5-flash", messages=messages,
                               extra_body={"reasoning": {"effort": "minimal"}})
```

`Authorization: Bearer sk-or-v1-…`. `reasoning.effort` maps to Gemini's thinking level. 1,048,576
context, 65,536 max output, **$1.50/M input and $9.00/M output — and thinking bills at the output
rate**, which makes the thinking cap worth more here than on any other model in this repo.

Served by Google AI Studio *and* Google Vertex (US) with failover, so **the serving path can change
mid-run**. Record which one answered.

## `deepseek-reasoner`

```python
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com", timeout=7200)
client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=8192)
```

· Source: `NEG_Deepseek/deepseek_neg_eval.py:16-17, 29-40`

- **The long timeout is not carelessness.** DeepSeek queues; the short timeouts used elsewhere here
  would classify a queued call as a hang.
- **The answer sometimes arrives in `reasoning_content` with `content` empty.** A runner reading only
  `message.content` scores those as failures. Fall back explicitly, and log when it fires.
- **`deepseek-reasoner` is an alias, not a model id.** What it resolves to can change under you;
  record the resolved identity with the results.

There is no reasoning knob. To spend less, the levers are the prompt ceiling
([model-parameters.md](model-parameters.md)) or a different model.
