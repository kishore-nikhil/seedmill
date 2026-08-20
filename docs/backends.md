# Backends

`model.backend` selects which client the engine talks through. The three are
**not interchangeable by `base_url` alone** — they send different request
bodies to different endpoints.

| `backend:` | Talks to | Key | Constrained decoding via |
|---|---|---|---|
| `ollama` (default) | Ollama's native `/api/chat` | none | `format: <schema>` |
| `openai_compat` | a local llama.cpp server | none | llama.cpp's `schema` extension |
| `openai` | OpenAI, or any OpenAI-compatible endpoint | `OPENAI_API_KEY` | standard `json_schema`, `strict: true` |

## ollama

```yaml
model:
  backend: ollama
  model: gemma4:27b
  base_url: http://localhost:11434
  think: false
```

Uses Ollama's **native** API rather than its OpenAI-compatible one, for two
capabilities the compat endpoint lacks: `think: false`, without which a
reasoning model puts its answer in a `reasoning` field and returns empty
`content` after burning the budget; and `format: <json schema>`, which is
what makes a `vocab:` field a decode-time guarantee.

## openai_compat (llama.cpp)

```yaml
model:
  backend: openai_compat
  model: ggml-org/Qwen3.8-27B-GGUF:Q4_K_M
  base_url: http://127.0.0.1:8080/v1
  think: false          # sends chat_template_kwargs.enable_thinking=false
```

Speaks the OpenAI protocol, but relies on two llama.cpp extensions: a
`schema` key inside a `json_object` response format, and `chat_template_kwargs`
for `think: false`. Both are rejected by hosted APIs — which is why `openai`
is a separate backend rather than a different URL.

## openai (hosted)

```yaml
model:
  backend: openai
  model: <model-id>           # required — no default, so nothing bills by accident
  # base_url:                 # omit for OpenAI; set it for another provider
  # strict: false             # last resort if a provider chokes on the schema
  params:                     # forwarded verbatim to the completions call
    top_p: 0.9
```

The key comes from `OPENAI_API_KEY`, or `model.api_key` in the YAML. A
missing key or missing model id is a config error before the first request,
not a 401 per seed.

Anything exposing an OpenAI-compatible endpoint works here — Gemini, for
one, by pointing `base_url` at its compat endpoint and putting that key in
`OPENAI_API_KEY`.

!!! warning "Verify enums on a new provider"
    Structured-output support varies. Generate a handful of records first and
    confirm the enum fields actually came back constrained rather than
    assuming they did.

### What changes when generation is metered

**Enums still hold, via a schema rewrite.** Hosted `strict: true` requires
`additionalProperties: false` and *every* property listed in `required`, with
optionality expressed as a null branch — the inverse of what the schema
builder emits. `fields:` is translated into that shape automatically, so a
`vocab:` field stays a hard decode-time constraint.

**Cost is reported.** Token usage accumulates per run and appears in the
stats table beside the record counts.

**Rate limits go to the SDK**, whose backoff honours `Retry-After` on 429s.
The engine's own retry loop sleeps a flat second and is for bad *records*,
not throttling. Start at `--workers 1` and raise it once you know your tier's
limit.

### Why `params:` is a passthrough

`reasoning_effort`, `max_completion_tokens` and friends differ per provider
and per model generation. Pinning them as named options in the client would
date it, so whatever you put under `params:` goes to the completions call
verbatim. Nothing is sent by default beyond model, messages, temperature and
the response format.

## Overriding at the command line

```bash
seedmill generate -t task.yaml --backend openai --model <id> --base-url https://…
```

`--backend` is an override, not a translator: it does not rewrite the YAML's
`base_url`, which is shaped for the backend it was written for. Override both
together.

## Reasoning models

gemma4 and Qwen3 think by default. Both **local** backends disable it with
`think: false` — Ollama through its `think` flag, llama.cpp through
`chat_template_kwargs.enable_thinking=false`, which it forwards to the Jinja
template. On llama.cpp the reasoning text goes to `reasoning_content` and
never pollutes the parsed record, so leaving it on only costs latency (~30%
per batch, measured on Qwen3.8-27B).

The hosted `openai` backend has no `think` flag; where a provider exposes an
equivalent, pass it through `params:`.
