# seedmill

**YAML in, training data out.** Config-driven synthetic data generation for
LLM training sets, running against local models (Ollama / llama.cpp) or a
hosted API. One engine, one CLI — *a new use case is a YAML file, not a new
script.*

[![CI](https://github.com/kishore-nikhil/seedmill/actions/workflows/ci.yml/badge.svg)](https://github.com/kishore-nikhil/seedmill/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Why

Most synthetic-data code is a script per dataset: a prompt, a parser, a retry
loop, an ad-hoc consistency check — copy-pasted for the next dataset and
diverging from then on. seedmill makes the use case a YAML file and keeps one
engine behind it. Three things fall out of that:

- **Vocabularies become grammar constraints.** A field declaring `vocab: icons`
  is validated *and* constrained at decode time, so the model physically cannot
  emit an out-of-vocabulary label. No post-hoc cleanup of 134 spellings of
  "coffee cup".
- **Seeds force coverage.** Free-running batches collapse onto the model's
  favourite topics. Iterating a seed axis with per-seed quotas is what gets you
  the tail.
- **Splits are group-aware.** Whole concept groups go to one split, so eval
  measures generalization to unseen concepts rather than memorized phrasings.

Local-first: Ollama and llama.cpp need no key and cost nothing per record.
A hosted `openai` backend is there when you want a stronger teacher model —
same YAML, same constrained decoding (see [Backends](#backends)).

```
config/tasks/<task>.yaml   ← everything about a use case lives here
seedmill/cli.py            ← one CLI for all tasks
seedmill/engine.py         ← one generation loop
```

## Install

Requires Python 3.10+ and a local model server — either
[Ollama](https://ollama.com) or a llama.cpp `llama-server` on its
OpenAI-compatible endpoint.

```bash
git clone https://github.com/kishore-nikhil/seedmill
cd seedmill
pip install -e .

# optional: embedding-based near-duplicate filtering (pulls torch)
pip install -e ".[embeddings]"
```

Paths inside a task YAML (`vocabularies`, `seeds`, `hooks`) resolve against
your working directory first, then against the task YAML's own directory. The
shipped tasks use the latter form, so they work from any directory.

## Quick start

```bash
# needs Ollama running with the task's model pulled (e.g. ollama pull gemma4:27b)
seedmill tasks                                          # list use cases
seedmill generate -t config/tasks/icon_classifier.yaml  # full run (resumable)
seedmill stats    -t config/tasks/icon_classifier.yaml
seedmill export   -t config/tasks/icon_classifier.yaml  # train/val/test in SFT + pairs formats

# smoke test with a smaller model
seedmill generate -t config/tasks/icon_classifier.yaml \
    --model gemma4:12b --limit 2 --per-seed 3 -o output/smoke.jsonl
```

Runnable from a clone without installing, too: `python -m seedmill tasks`.

## A task, abridged

Abridged from `config/tasks/icon_classifier.yaml` — given an interest
phrase, pick the best icon from a fixed set:

```yaml
name: icon_classifier

model:
  backend: ollama
  model: gemma4:27b
  think: false

vocabularies:
  icons:                        # ~280 names; fields using it are decode-constrained
    file: ../icons.yaml
    key: icons

seeds:                          # iterate concepts instead of free-running
  file: ../concepts.yaml
  key: concepts
  group_field: category
  value_field: concept_group
  per_seed: 8
  id_pattern: "{seed}_{counter:03d}"

fields:                         # abridged; the real task has 4 fields
  interest:       {type: str, min_length: 3, max_length: 120}
  positive_icon:  {type: str, vocab: icons}
  hard_negatives: {type: list, items: {vocab: icons}, min_items: 2, max_items: 4}

rules:                          # abridged; 5 rules in the real task
  - unique_items: {list: hard_negatives}
  - not_in: {value: positive_icon, list: hard_negatives}

exports:
  split_by: concept_group       # a group never straddles two splits
  splits: {train: 0.8, val: 0.1, test: 0.1}
```

Producing records like:

```json
{"id": "glassblowing_014", "concept_group": "glassblowing",
 "interest": "making colorful objects from molten glass",
 "positive_icon": "flame", "hard_negatives": ["wine-glass", "palette"]}
```

## Backends

| `backend:` | Talks to | Key | Constrained decoding via |
|---|---|---|---|
| `ollama` (default) | Ollama's native `/api/chat` | none | `format: <schema>` |
| `openai_compat` | a local llama.cpp server | none | llama.cpp's `schema` extension |
| `openai` | OpenAI, or anything with an OpenAI-compatible endpoint | `OPENAI_API_KEY` | standard `json_schema`, `strict: true` |

The two local backends are not interchangeable by `base_url` alone — they
send different request bodies. `openai_compat` is for llama.cpp
specifically: the `schema` key it puts inside a `json_object` response
format, and the `chat_template_kwargs` it sends for `think: false`, are
llama.cpp extensions that a hosted API rejects with a 400. That is why
`openai` is a separate backend rather than a different URL.

```yaml
model:
  backend: openai
  model: <model-id>           # required — no default, so nothing bills by accident
  # base_url:                 # omit for OpenAI; set it for any other provider
  # strict: false             # last resort if a provider chokes on the schema
  params:                     # forwarded verbatim to the completions call
    top_p: 0.9
```

Anything exposing an OpenAI-compatible endpoint works through the same
backend — Gemini, for one, by pointing `base_url` at its compat endpoint
and putting that key in `OPENAI_API_KEY`. Structured-output support varies
between providers, so generate a handful of records first and confirm the
enum fields actually came back constrained rather than assuming they did.

Two differences from a local run are worth knowing before a long one:

- **Enums need a schema rewrite.** Hosted `strict: true` requires
  `additionalProperties: false` and *every* property listed in `required`;
  optionality has to be expressed as a null branch instead. `fields:` is
  translated to that shape automatically (`schema.to_strict_json_schema`),
  so a `vocab:` field stays a hard decode-time constraint here too.
- **Cost is reported, not assumed.** Token usage accumulates per run and
  lands in the stats table next to the record counts. Rate limits are
  handled by the SDK's own backoff, which honours `Retry-After` — start at
  `--workers 1` and raise it once you know the tier's limit.

`params:` is a passthrough rather than a fixed set of options because the
knobs (`reasoning_effort`, `max_completion_tokens`, …) differ per provider
and per model generation; pinning them in the client would date it.

## Anatomy of a task

| Section        | What it does |
|----------------|--------------|
| `model`        | backend (`ollama` \| `openai_compat` \| `openai`), model name, `base_url`, `think: false` for reasoning models (local backends), `params:` passthrough (hosted) — see [Backends](#backends) |
| `vocabularies` | named value sets loaded from files; fields with `vocab: <name>` are validated **and grammar-constrained at decode time** — the model cannot emit an out-of-vocabulary value. Three file shapes: grouped (`{group: [...]}`), flat list, or a mapping (`{slug: codepoint}`) whose **keys** are the vocabulary — so an app's own icon table can be the source of truth |
| `seeds`        | optional iteration axis (e.g. concept groups). The engine loops seeds and generates `per_seed` records each, with patterned ids (`{seed}_{counter:03d}`) and resume support. Without `seeds`, the engine free-runs batches up to `generation.count` |
| `fields`       | the record schema → pydantic validation + the constrained-decoding JSON schema |
| `rules`        | declarative cross-field checks with auto-fix: `not_in`, `contains`, `disjoint`, `unique_items`, `exclusive_switch` (discriminated unions) |
| `prompts`      | `system` / `user` / `retry` templates. `$var` substitution: `$n`, `$seed`, `$group`, `$vocab_<name>`, `$schema_json`, `$examples`, `$description` |
| `generation`   | batch size, retries, temperature, `workers` (concurrent seeds), dedup (hash always; embeddings optional) |
| `exports`      | split ratios and `split_by` group field — group-aware splits so eval measures generalization to *unseen* concepts |
| `hooks`        | optional Python file for logic YAML shouldn't express: `postprocess(record, ctx)` and `EXPORTERS = {"fmt": fn(record, ctx)}` |

Pipeline per batch: constrained decode → pydantic → rules (auto-fix or
reject) → bounds → hooks → dedup → append JSONL. Reruns resume from the
output file.

## Included tasks

Both were built for real datasets, not as demos.

- **icon_classifier** — interest phrase → best icon from a fixed vocabulary
  (`config/icons.yaml`), seeded by concept groups (`config/concepts.yaml`).
  Exports `sft` (chat format for fine-tuning a small LLM) and `pairs`
  (text/label for a classical classifier).

- **pacekeeper_nlp_v3** — natural language → app config for a habit/health
  tracker (tracker / streak / health log / none). Every design decision below
  closed a defect class that made an earlier free-batch version of this
  dataset unusable as classifier training input:

  | Change | Why |
  |---|---|
  | `recommended_icon` grammar-constrained to `config/pacekeeper_icons.yaml` — the app's own picker lists | The earlier version emitted 134 free-text spellings ("coffee cup", "capsule", "yoga_pose"), none of which the Flutter app can render. Icons are also **per entity type** (tracker 16 / streak 12 / health 12); the union is enforced at decode time, the per-entity half in hooks, where `entity_type` is known |
  | `streak_config` carries `start_date` (verbatim user words) + `duration_value`/`duration_unit`, never a calendar date | Every ISO date the generator wrote was invented: a "45 day challenge" spanning 69 days, starts in 2024/2025, three date formats. The app resolves real dates at creation time |
  | Seeded over `config/habit_domains.yaml`, with `none` as its own 14-seed group | The earlier version collapsed onto its favourite topics and produced 2 out-of-scope rows in 918. The `topical_*` seeds are the point: "how do i delete my old gym records" is dense with gym words and must still come back `none` |
  | Hooks reject any label numeral absent from the query | Catches unit conversion — "run 5km" labeled `3.1`, "2L of water" labeled `2000` — at 1.8% of rows with no false positives on the corpus |

  Exports a `bert` format: one text, one label per head, `"n/a"` for heads
  that do not apply to the record's entity type.

  ```bash
  seedmill generate -t config/tasks/pacekeeper_nlp_v3.yaml --workers 4
  seedmill export   -t config/tasks/pacekeeper_nlp_v3.yaml
  python tools/clean_v3.py output/pacekeeper_nlp_v3.jsonl        # report
  python tools/clean_v3.py output/pacekeeper_nlp_v3.jsonl --apply
  ```

  `tools/clean_v3.py` migrates rows written before a schema change and
  drops what it cannot recover, then the next `generate` backfills the gap.
  It runs every row through the task's own `postprocess` hook rather than
  reimplementing the checks, so migrated and generated rows cannot drift.

## Adding a use case

1. Copy an existing task YAML, change fields/prompts/vocab.
2. Only add a hooks file if you need custom export formats or
   post-processing.
3. `seedmill generate -t config/tasks/your_task.yaml`

## Notes

- **Reasoning models**: gemma4 and Qwen3 think by default. Both local
  backends disable it via `think: false` — Ollama through its `think` flag, the
  OpenAI-compat backend through
  `chat_template_kwargs.enable_thinking=false`, which llama.cpp forwards to
  the Jinja template. On llama.cpp the reasoning text goes to
  `reasoning_content` and never pollutes the parsed record, so leaving it
  on only costs latency (~30% per batch, measured on Qwen3.8-27B). The
  hosted `openai` backend has no `think` flag — where a provider exposes
  an equivalent, pass it through `params:`.
- **Throughput on a local llama.cpp server** is decode-bound, not
  prompt-bound: ~14 s per record for Qwen3.8-27B Q4 on an M-series Mac,
  and `--workers 4` buys ~1.4x aggregate, not 4x, because the box is
  memory-bandwidth limited. Budget hours, not minutes, for a 1000+ record
  run — it resumes from the output file, so stopping it is cheap.
- **Prompt order matters for speed**: put per-seed text at the *end* of the
  user prompt. Everything before the first varying byte is reused from
  llama.cpp's slot prefix cache; a "Domain: X" header at the top forces a
  full re-prefill of the vocabulary, schema and examples on every call.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

The test suite is pure-function only — it never contacts a model and never
imports the optional embeddings dependency.

## License

Apache-2.0. See [LICENSE](LICENSE).
