# Getting started

## Install

Requires Python 3.10+ and something to generate against: a local server —
[Ollama](https://ollama.com) or a llama.cpp `llama-server` — or a hosted API
key. See [Backends](backends.md).

```bash
pip install seedmill

# optional: embedding-based near-duplicate filtering (pulls torch)
pip install "seedmill[embeddings]"
```

The wheel ships the engine and the CLI, not task YAMLs — those are yours to
write. To get the worked examples in `config/tasks/` as well, clone instead:

```bash
git clone https://github.com/kishore-nikhil/seedmill
cd seedmill && pip install -e .
```

## Scaffold a task

```bash
seedmill init
```

This writes `config/tasks/example.yaml` — a support-intent classifier with
its vocabulary and seeds inline, so it runs with no other files. Pass a name
to call it something else (`seedmill init support_intents`), `--dir` to put
it elsewhere, `--force` to overwrite.

## Generate

Start small. `--limit` caps how many seeds are visited and `--per-seed`
overrides the quota, so this is a handful of records rather than the full run:

```bash
seedmill generate -t config/tasks/example.yaml --limit 2 --per-seed 2
```

Records land in `output/example.jsonl`, appended as they are accepted. When
it looks right, drop the flags:

```bash
seedmill generate -t config/tasks/example.yaml
```

!!! tip "Runs are resumable"
    Output is appended, and a seeded run skips ids it already has. Stopping
    a long run costs you the batch in flight, nothing more — worth knowing
    before you start a 1000-record job on a local model.

## Inspect and export

```bash
seedmill stats  -t config/tasks/example.yaml   # totals, per-group counts, top enum values
seedmill export -t config/tasks/example.yaml   # train/val/test splits
```

`export` writes one file per format per split into `output/<task>_export/`.
Built in is `raw` (the record as generated); other formats come from the
task's [hooks file](cookbook.md#add-a-custom-export-format).

## Where paths resolve

Paths inside a task YAML (`vocabularies`, `seeds`, `hooks`) resolve against
your **working directory first**, then against the task YAML's own directory.
The shipped tasks use the second form, so they work from anywhere; a
scaffolded task has no external paths at all.

## Next

- [Task reference](task-reference.md) — every key, with defaults.
- [Backends](backends.md) — pointing it at llama.cpp or a hosted model.
