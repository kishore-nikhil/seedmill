# Cookbook

Recipes for the things people hit second.

## Constrain a field to a vocabulary

The point of the tool. Declare the value set once, reference it from a field:

```yaml
vocabularies:
  icons: {file: ../icons.yaml, key: icons}

fields:
  positive_icon: {type: str, vocab: icons}
```

The vocabulary is injected as an `enum`, which reaches the backend as a
grammar constraint. The model cannot emit anything else — this is a decode
guarantee, not a post-hoc filter.

To check it is actually in force on a new backend, generate a few records and
confirm no rejections come back naming that field.

## Use an app's own table as the vocabulary

If your app already has the label space — an icon slug → code point map, say —
point at it directly rather than keeping a second list:

```yaml
vocabularies:
  icons: {file: ../pacekeeper_icons.yaml, key: icons}
```

A mapping's **keys** become the vocabulary and its values ride along unused,
so the app file stays the single source of truth and cannot drift.

## Enforce a subset the grammar cannot express

A decode-time grammar cannot condition on a field the model has not chosen
yet. If icons are per-entity but `entity_type` is chosen in the same record,
flatten the vocabulary for decoding and narrow it afterwards in a hook:

```python
def postprocess(record, ctx):
    allowed = PER_ENTITY[record["entity_type"]]
    if record["icon"] not in allowed:
        return None          # reject; the batch retries with the error fed back
    return record
```

## Add a custom export format

```python
# config/tasks/mytask_hooks.py
def build_sft(record, ctx):
    return {
        "messages": [
            {"role": "user", "content": record["interest"]},
            {"role": "assistant", "content": record["positive_icon"]},
        ]
    }

EXPORTERS = {"sft": build_sft}
```

```yaml
hooks: mytask_hooks.py
exports:
  formats: [raw, sft]
```

`seedmill export` then writes `sft_train.jsonl`, `sft_val.jsonl`, and so on.

## Keep a concept out of two splits at once

```yaml
exports:
  split_by: concept_group
```

Whole groups go to one split. Without this, near-identical phrasings of the
same concept land on both sides and eval measures memorization.

## Smoke-test a task cheaply

```bash
seedmill generate -t config/tasks/mytask.yaml \
    --model gemma4:12b --limit 2 --per-seed 3 -o output/smoke.jsonl
```

A smaller model, two seeds, three records each, into a throwaway file. Enough
to catch a broken prompt or a vocabulary that does not resolve, without
committing to a full run.

## Speed up a local run

Put per-seed text at the **end** of the user prompt. Everything before the
first varying byte is reused from llama.cpp's slot prefix cache; a
`Domain: X` header at the top re-prefills the vocabulary, schema and examples
on every call.

Then raise `--workers` — but expect sub-linear gains. A local server is
memory-bandwidth limited, so 4 workers buys roughly 1.4x.

## Filter near-duplicates, not just exact ones

```bash
pip install "seedmill[embeddings]"
```

```yaml
generation:
  dedup:
    fields: [interest]
    embeddings: true
    threshold: 0.9
```

Off by default because it pulls torch. Hash-based exact dedup always runs
regardless.

## Point a task at a different model for one run

```bash
seedmill generate -t task.yaml --backend openai --model <id>
```

`--backend` does not rewrite the YAML's `base_url`, which is shaped for the
backend it was written for — override both together if the new backend needs
a different endpoint. See [Backends](backends.md).
