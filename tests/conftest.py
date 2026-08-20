"""
Shared fixtures.

Every fixture writes its inputs into `tmp_path` rather than into a
checked-in data directory: it keeps each test's input visible at the point
of use, and it is what makes the path-resolution tests possible at all.

Fixture filenames are prefixed `zz_` so the cwd-first branch of
`task_config._resolve_path` can never accidentally hit a real file in
whatever directory pytest happens to run from.

No fixture here constructs a model client or touches the network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def load_hooks_module():
    """
    Import a hooks file by path, the same way `task_config._load_hooks`
    does. Hooks files are not importable as packages, so this is the only
    way to test them directly.
    """
    def _load(path: Path):
        spec = importlib.util.spec_from_file_location(f"test_hooks_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return _load


@pytest.fixture
def grouped_vocab_file(tmp_path: Path) -> Path:
    """Grouped shape: {group: [value, ...]} — grouping is kept for prompts."""
    p = tmp_path / "zz_grouped_vocab.yaml"
    p.write_text(yaml.safe_dump({"icons": {"a": ["alpha", "beta"], "b": ["gamma"]}}))
    return p


@pytest.fixture
def flat_vocab_file(tmp_path: Path) -> Path:
    """Flat shape: [value, ...]."""
    p = tmp_path / "zz_flat_vocab.yaml"
    p.write_text(yaml.safe_dump({"icons": ["alpha", "beta", "gamma"]}))
    return p


@pytest.fixture
def mapping_vocab_file(tmp_path: Path) -> Path:
    """Mapping shape: the KEYS are the vocabulary, values ride along unused."""
    p = tmp_path / "zz_mapping_vocab.yaml"
    p.write_text(yaml.safe_dump({"icons": {"alpha": "0xe1", "beta": "0xe2"}}))
    return p


MINI_HOOKS = '''\
def postprocess(record, ctx):
    if record.get("label") == "drop_me":
        return None
    record["touched"] = True
    return record


def build_pairs(record, ctx):
    return {"text": record["text"], "label": record["label"]}


EXPORTERS = {"pairs": build_pairs}
'''


@pytest.fixture
def mini_task(tmp_path: Path) -> Path:
    """
    A complete minimal task YAML exercising every loader branch: a
    vocabulary with a `key:`, inline seeds, an enum field, a `vocab:`
    field, a list with `items: {vocab:}`, a nullable field, a nested
    object, one rule of each kind, exports with `split_by`, and hooks.

    Returns the path to the task YAML.
    """
    (tmp_path / "zz_vocab.yaml").write_text(
        yaml.safe_dump({"icons": {"a": ["alpha", "beta"], "b": ["gamma"]}})
    )
    (tmp_path / "zz_hooks.py").write_text(MINI_HOOKS)

    task = {
        "name": "mini task",
        "description": "fixture task",
        "model": {"backend": "ollama", "model": "nonexistent-model"},
        "vocabularies": {"icons": {"file": "zz_vocab.yaml", "key": "icons"}},
        "seeds": {
            "values": {"g1": ["s1", "s2"], "g2": ["s3"]},
            "group_field": "group",
            "value_field": "seed",
            "per_seed": 4,
            "id_pattern": "{seed}_{counter:03d}",
        },
        "fields": {
            "id": {"type": "str"},
            "text": {"type": "str", "min_length": 3},
            "label": {"type": "str", "enum": ["keep", "drop_me"]},
            "icon": {"type": "str", "vocab": "icons"},
            "tags": {"type": "list", "items": {"vocab": "icons"}, "max_items": 3},
            "note": {"type": "str", "nullable": True},
            "cfg": {
                "type": "object",
                "nullable": True,
                "fields": {"n": {"type": "int"}, "kind": {"type": "str", "enum": ["x", "y"]}},
            },
            "group": {"type": "str"},
        },
        "rules": [
            {"unique_items": {"list": "tags"}},
            {"not_in": {"value": "icon", "list": "tags"}},
        ],
        "prompts": {"system": "sys", "user": "user $n"},
        "generation": {"batch_size": 2, "temperature": 0.5},
        "exports": {
            "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
            "split_by": "group",
            "seed": 7,
            "formats": ["raw", "pairs"],
        },
        "hooks": "zz_hooks.py",
    }
    p = tmp_path / "zz_task.yaml"
    p.write_text(yaml.safe_dump(task, sort_keys=False))
    return p
