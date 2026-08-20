# seedmill

**YAML in, training data out.** Config-driven synthetic data generation for
LLM training sets, against local models (Ollama / llama.cpp) or a hosted API.
One engine, one CLI — *a new use case is a YAML file, not a new script.*

```bash
pip install seedmill
seedmill init
seedmill generate -t config/tasks/example.yaml
```

## Why it is shaped this way

Most synthetic-data code is a script per dataset: a prompt, a parser, a retry
loop, an ad-hoc consistency check — copy-pasted for the next dataset and
diverging from then on. seedmill makes the use case a YAML file and keeps one
engine behind it. Three things fall out of that:

**Vocabularies become grammar constraints.** A field declaring `vocab: icons`
is validated *and* constrained at decode time, so the model physically cannot
emit an out-of-vocabulary label. No post-hoc cleanup of 134 spellings of
"coffee cup". This holds on every backend — see [Backends](backends.md).

**Seeds force coverage.** Free-running batches collapse onto the model's
favourite topics. Iterating a seed axis with per-seed quotas is what gets you
the tail.

**Splits are group-aware.** Whole concept groups go to one split, so eval
measures generalization to unseen concepts rather than memorized phrasings.

## Where to go next

- [Getting started](getting-started.md) — install to first dataset.
- [Task reference](task-reference.md) — every key a task YAML can contain.
- [Cookbook](cookbook.md) — recipes for the things people hit second.
- [CLI reference](cli.md) — generated from the commands themselves.

## The pipeline

Each batch runs the same way, whatever the task:

```
constrained-decode JSON → pydantic validation → declarative rules (auto-fix)
    → field bounds → hooks.postprocess → dedup → append to JSONL
```

A batch that fails validation is retried with the errors fed back into the
prompt, up to `generation.max_retries`. Output is appended as it goes, so a
run is resumable: stop it and start it again and it picks up from the file.
