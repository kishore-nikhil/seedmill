"""
Guards against the docs drifting from the code.

Neither test checks whether the prose is *good* — only that the reference
pages still mention everything the code actually reads. A key added to
`load_task` or a command added to the CLI has to be written up somewhere,
or these fail.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from seedmill import cli, task_config

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _read(name: str) -> str:
    path = DOCS / name
    if not path.is_file():
        pytest.skip(f"{name} not present (docs are not shipped in the wheel)")
    return path.read_text()


def _top_level_keys() -> set[str]:
    """Every `raw[...]` / `raw.get(...)` key read by load_task."""
    source = Path(task_config.__file__).read_text()
    body = source.split("def load_task")[1]
    return set(re.findall(r'raw(?:\.get)?[(\[]"([a-z_]+)"', body))


def test_every_task_key_is_documented():
    reference = _read("task-reference.md")
    undocumented = sorted(k for k in _top_level_keys() if k not in reference)
    assert not undocumented, (
        f"task-reference.md does not mention: {undocumented}. "
        "A key the loader reads but the reference omits is a key nobody knows about."
    )


def test_the_key_extraction_still_finds_something():
    """If load_task is refactored past the regex, the test above would pass
    vacuously — this is the canary for that."""
    keys = _top_level_keys()
    assert {"name", "fields", "prompts", "seeds"} <= keys


def test_every_cli_command_is_in_the_generated_reference():
    reference = _read("cli.md")
    names = {c.name or c.callback.__name__ for c in cli.app.registered_commands}
    missing = sorted(n for n in names if f"`seedmill {n}`" not in reference)
    assert not missing, (
        f"docs/cli.md is stale, missing: {missing}. Regenerate with "
        "`typer seedmill.cli utils docs --name seedmill --output docs/cli.md`."
    )
