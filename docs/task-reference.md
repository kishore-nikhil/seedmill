# Task reference

Every key a task YAML can contain. `name`, `fields` and `prompts` are
required; everything else has a default.

```yaml
name:           # required — identifies the task and names the output file
description:    # shown by `seedmill tasks`, and available as $description
model:          # backend, model, sampling — see Backends
vocabularies:   # named value sets that become decode-time constraints
seeds:          # optional coverage axis, or one seed per file
fields:         # the record schema
rules:          # cross-field checks, with auto-fix
prompts:        # system / user / retry templates, and images to attach
examples:       # few-shot records rendered into $examples
generation:     # batch size, retries, temperature, workers, dedup
exports:        # splits and formats
hooks:          # path to a Python file with postprocess() / EXPORTERS
```

## name

Required. Also the default output path: `output/<name>.jsonl`, overridable
with `-o`.

## description

Free text. Shown in `seedmill tasks` and substitutable into prompts as
`$description`.

## model

See [Backends](backends.md) for the full treatment.

| Key | Default | Applies to |
|---|---|---|
| `backend` | `ollama` | all |
| `model` | per backend; **required** for `openai` | all |
| `base_url` | per backend | all |
| `think` | `false` (ollama), `true` (openai_compat) | local backends |
| `api_key` | `$OPENAI_API_KEY` | `openai` |
| `strict` | `true` | `openai` |
| `params` | `{}` | `openai` |

Sampling temperature is **not** here — it lives in `generation.temperature`.

## vocabularies

Named value sets. A field declaring `vocab: <name>` is validated against the
set *and* constrained at decode time, so the model cannot emit anything else.

Inline:

```yaml
vocabularies:
  intents: [billing, bug_report, cancellation]
```

Or from a file, with an optional `key` to index into it:

```yaml
vocabularies:
  icons:
    file: ../icons.yaml
    key: icons
```

Three file shapes are accepted:

| Shape | YAML | Behaviour |
|---|---|---|
| grouped | `{group: [value, …]}` | groups are kept and rendered into `$vocab_<name>` |
| flat | `[value, …]` | rendered as a sorted comma list |
| mapping | `{value: <anything>}` | the **keys** are the vocabulary; values ride along unused |

The mapping shape is the useful one for app-owned label spaces: an icon
slug → code point table exported from the app can be the single source of
truth, with no hand-kept second list to drift out of sync.

## seeds

The coverage axis. Without it the engine free-runs batches up to
`generation.count`; with it, it walks every seed and generates `per_seed`
records for each.

```yaml
seeds:
  file: ../concepts.yaml     # or: values: {group: [seed, …]}
  key: concepts
  group_field: category      # record field that stores the group name
  value_field: concept_group # record field that stores the seed value
  per_seed: 8
  id_pattern: "{seed}_{counter:03d}"
```

| Key | Default | Meaning |
|---|---|---|
| `file` / `values` | — | seeds from a YAML file, or inline |
| `key` | — | index into the loaded file |
| `group_field` | `None` | record field the group name is written to |
| `value_field` | `seed` | record field the seed value is written to |
| `per_seed` | `8` | records generated per seed |
| `id_pattern` | `None` | e.g. `{seed}_{counter:03d}`; enables resume |

A flat list instead of a mapping is treated as one group named `default`.

!!! note "How resume actually works"
    A restarted seeded run counts the records already in the output file per
    `value_field`, and generates only the shortfall against `per_seed`. It
    matches on the seed value, not on ids — `id_pattern` only names records
    and continues the counter, so resume works with or without one.

### Seeds from files

`files:` makes one seed per file instead: a glob, or a list of globs. This is
the shape for labelling a folder of images (or anything else on disk) rather
than generating from concepts.

```yaml
seeds:
  files: scans/**/*.jpg      # or a list: [scans/nemo/*.jpg, scans/krazy/*.jpg]
  value_field: page
  group_field: strip
```

| Key | Default | Meaning |
|---|---|---|
| `files` | — | glob or list of globs; `**` matches any depth. Can't be combined with `values` or `file` |
| `per_seed` | `1` | records per file |

The other keys (`value_field`, `group_field`, `id_pattern`) mean what they
mean above.

- **The seed** is the file's path below the glob's fixed part:
  `scans/nemo/1906-08-26.jpg` under `scans/**/*.jpg` is seed
  `nemo/1906-08-26.jpg`. The folders stay in, so two strips' pages of the
  same date stay two seeds, and resume matches on it like any other seed.
- **The group** is the first folder below the fixed part (`nemo`), or the
  fixed part's own folder for files directly in it. Name it in
  `exports.split_by` to keep each folder in one split.
- **`$seed_path`** is the file's absolute path, for `prompts.images` or a
  hook. Hooks also get it as `ctx["variables"]["seed_path"]`.
- **Dedup is off** unless `generation.dedup` is set: two files can rightly
  get the same record.

Globs resolve from the working directory first, then the task YAML's
directory, as other task paths do. A glob that matches nothing is an error
before the run, and so are two files that would make the same seed.

## fields

The record schema. Drives three things at once: pydantic validation, the
constrained-decoding JSON schema, and the `$schema_json` prompt variable.

```yaml
fields:
  interest:
    type: str
    min_length: 3
    max_length: 120
    description: Short free-text phrase a user might type.
  positive_icon:
    type: str
    vocab: icons
  hard_negatives:
    type: list
    items: {vocab: icons}
    min_items: 2
    max_items: 4
  entity:
    type: object
    nullable: true
    fields:
      kind: {type: str, enum: [tracker, streak]}
```

| Key | Meaning |
|---|---|
| `type` | `str`, `int`, `float`, `bool`, `list`, `dict`, `object`. Unrecognised → `str` |
| `enum` | allowed values; enforced by pydantic **and** at decode time |
| `vocab` | resolve a named vocabulary into `enum` |
| `nullable` | field may be null; dropped from `required` |
| `default` | value when absent; also drops the field from `required` |
| `items` | element spec for `type: list` (may itself carry `vocab`/`enum`) |
| `fields` | sub-fields for `type: object` |
| `description` | carried into the schema the model sees |
| `min_length` / `max_length` | string bounds, checked after validation |
| `min_items` / `max_items` | list bounds, checked after validation |

### Nesting

`type: object` takes `fields`, and `type: list` takes `items`, which can
itself be an object. They nest to any depth, and every level gets the same
treatment: pydantic validation, the decode-time schema, `vocab:` resolution
and the bounds checks.

```yaml
fields:
  panels:
    type: list
    max_items: 12
    items:
      type: object
      fields:
        order: {type: int}
        box: {type: list, items: {type: int}, min_items: 4, max_items: 4}
        balloons:
          type: list
          items:
            type: object
            fields:
              kind: {type: str, enum: [speech, thought, caption, sound]}
              text: {type: str, min_length: 1, max_length: 200}
```

A bound below the top level names its path when it fails, as in
`panels[2].balloons[0].text too long (214 chars)`, so the retry prompt points
the model at the right balloon.

!!! info "Why bounds are not in the decode schema"
    `min_length` and friends are checked *after* generation rather than
    expressed as grammar constraints, for backend compatibility. They reject
    a record; enums prevent one.

## rules

Cross-field checks that run after pydantic validation. Each rule is a
single-key dict. Some can auto-fix instead of rejecting.

```yaml
rules:
  - unique_items: {list: hard_negatives}
  - not_in: {value: positive_icon, list: hard_negatives}
  - contains: {list: acceptable_icons, value: positive_icon, autofix: true}
  - disjoint: {a: acceptable_icons, b: hard_negatives, autofix_remove_from: a}
```

| Rule | Spec | Behaviour |
|---|---|---|
| `unique_items` | `{list}` | deduplicates the list in place — always a fix, never a rejection |
| `not_in` | `{value, list}` | rejects if `record[value]` appears in `record[list]` |
| `contains` | `{list, value, autofix}` | `record[list]` must contain `record[value]`; with `autofix`, prepends it |
| `disjoint` | `{a, b, autofix_remove_from}` | two lists must not overlap; with `a`/`b`, drops the overlap from that side |
| `exclusive_switch` | `{switch, cases}` | discriminated union: when `record[switch] == value`, that case field must be non-null and every other case field null |

An unknown rule name is reported as an issue rather than ignored, so a typo
fails loudly.

### exclusive_switch

```yaml
rules:
  - exclusive_switch:
      switch: entity_type
      cases:
        tracker: tracker_config
        streak: streak_config
```

A switch value with no case listed (e.g. `none`) requires **all** case fields
to be null.

## prompts

`system` and `user` are required; `retry` has a built-in default; `images`
is optional.

```yaml
prompts:
  system: |
    You generate training data for …
  user: |
    Generate exactly $n records for "$seed_readable".
  retry: |
    These failed validation:
    $errors
    Regenerate exactly $n valid records.
  images: [$seed_path]    # optional; attached to the user message
```

`$var` substitution, available in `system` and `user`:

| Variable | Value |
|---|---|
| `$n` | records requested in this batch |
| `$name`, `$description` | from the task |
| `$schema_json` | compact sketch of `fields` |
| `$examples` | rendered `examples:` block |
| `$vocab_<name>` | one entry per vocabulary |
| `$seed`, `$group` | seeded mode only |
| `$seed_readable` | `$seed` with underscores as spaces |
| `$seed_path` | file seeds only: the seed file's absolute path |

`retry` receives only `$errors` (the first five, one per line) and `$n`.

### images

A path, or a list of them, attached to the user message for a vision model.
Each entry is a `$var` template rendered per batch, so `$seed_path` attaches
each file seed's own image. A plain path (a reference sheet every call
should see, say) is the same image on every call.

```yaml
prompts:
  images: [$seed_path, ref/lettering-key.png]
```

- **Formats:** `.png`, `.jpg`/`.jpeg`, `.webp`, `.gif`. Anything else is an
  error, before the run for a plain path.
- **Size:** the file is sent as it is, never resized or re-encoded. Prepare
  images at the size the model reads best: Ollama and llama.cpp resize to
  the model's own input, and a hosted API bills by the image's size.
- **Retries** resend the images with the error feedback.
- **The model has to be a vision model.** See [Backends](backends.md#images)
  for what each backend needs.

Paths resolve like other task paths: working directory first, then the task
YAML's directory.

!!! tip "Prompt order matters for local throughput"
    Put per-seed text at the **end** of the user prompt. Everything before
    the first varying byte is reused from llama.cpp's slot prefix cache; a
    `Domain: X` header at the top forces a full re-prefill of the vocabulary,
    schema and examples on every call.

## examples

A list of records rendered into `$examples` for few-shot prompting. Plain
YAML mappings, shaped like the records you want back.

## generation

| Key | Default | Meaning |
|---|---|---|
| `count` | `100` | total records in batch mode (ignored when `seeds:` is set) |
| `batch_size` | `4` | records requested per model call |
| `max_retries` | `2` | error-feedback retries per batch |
| `temperature` | `0.8` | sampling temperature |
| `workers` | `1` | concurrent seeds in seeded mode |
| `dedup` | on; off for file seeds | `false` turns dedup off entirely |
| `dedup.fields` | all fields | fields hashed for exact-duplicate detection |
| `dedup.embeddings` | `false` | near-duplicate filtering by cosine similarity |
| `dedup.threshold` | `0.92` | cosine similarity above which a record is a duplicate |

`dedup.embeddings` is off by default because it pulls `sentence-transformers`
(and therefore torch); install the `embeddings` extra before turning it on.
Hash-based exact dedup runs whenever dedup is on.

!!! note "`workers` on a local model"
    Throughput on a local server is decode-bound, not prompt-bound. On an
    M-series Mac with Qwen3.8-27B Q4, `--workers 4` buys roughly 1.4x
    aggregate rather than 4x — the box is memory-bandwidth limited.

## exports

```yaml
exports:
  split_by: concept_group
  splits: {train: 0.8, val: 0.1, test: 0.1}
  seed: 42
  formats: [raw]
```

| Key | Default | Meaning |
|---|---|---|
| `split_by` | `None` | field whose whole groups go to one split |
| `splits` | `{train: 0.8, val: 0.1, test: 0.1}` | split ratios |
| `seed` | `42` | shuffle seed, so splits are reproducible |
| `formats` | every registered exporter | which formats to write |

`raw` is built in. Anything else comes from the task's hooks file — an
unknown format is an error listing what is available, rather than a silently
missing file.

`split_by` is the group-aware part: naming the seed field there keeps a whole
concept in one split, so eval measures generalization to unseen concepts
rather than memorized phrasings.

## hooks

Path to a Python file, resolved like other task paths. Two entry points, both
optional:

```python
def postprocess(record: dict, ctx: dict) -> dict | None:
    """Return the record, modified — or None to reject it."""

EXPORTERS = {"sft": build_sft}   # name -> (record, ctx) -> dict
```

The `ctx` passed to `postprocess` carries `vocab`, `config`, and — in seeded
mode — `variables`, `seed` and `group`. That last part is the point: a hook
can enforce something the decode-time grammar cannot, like a per-entity
subset of a flattened vocabulary, because a grammar cannot condition on a
field the model has not chosen yet.

Exporters receive a smaller `ctx`: `vocab` and `config` only.
