"""Task YAML loading: vocab shapes, path resolution, enum injection."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from seedmill.task_config import Seeds, _load_vocab, _resolve_path, load_task

# ── vocab shapes ────────────────────────────────────────────────────────


def test_vocab_inline_list(tmp_path: Path):
    v = _load_vocab("icons", ["alpha", "beta"], tmp_path)
    assert v.flat == {"alpha", "beta"}
    assert v.grouped is None


def test_vocab_grouped_file(grouped_vocab_file: Path):
    v = _load_vocab("icons", {"file": str(grouped_vocab_file), "key": "icons"}, Path("."))
    assert v.flat == {"alpha", "beta", "gamma"}
    assert v.grouped == {"a": ["alpha", "beta"], "b": ["gamma"]}
    # grouped branch keeps the authored order, one group per line
    assert v.prompt_block() == "a: alpha, beta\nb: gamma"


def test_vocab_flat_file(flat_vocab_file: Path):
    v = _load_vocab("icons", {"file": str(flat_vocab_file), "key": "icons"}, Path("."))
    assert v.flat == {"alpha", "beta", "gamma"}
    assert v.grouped is None
    # flat branch sorts, since there is no authored grouping to preserve
    assert v.prompt_block() == "alpha, beta, gamma"


def test_vocab_mapping_file_keys_are_the_vocabulary(mapping_vocab_file: Path):
    """
    The load-bearing case for "the app's own icon table is the source of
    truth": a slug -> codepoint mapping is accepted directly, keys become
    the label space, values ride along unused.
    """
    v = _load_vocab("icons", {"file": str(mapping_vocab_file), "key": "icons"}, Path("."))
    assert v.flat == {"alpha", "beta"}
    assert v.grouped is None


def test_vocab_mapping_with_non_string_keys(tmp_path: Path):
    p = tmp_path / "zz_numeric_keys.yaml"
    p.write_text(yaml.safe_dump({"icons": {1: "a", 2: "b"}}))
    v = _load_vocab("icons", {"file": str(p), "key": "icons"}, tmp_path)
    assert v.flat == {"1", "2"}


def test_vocab_key_selects_the_right_top_level_entry(tmp_path: Path):
    p = tmp_path / "zz_two_keys.yaml"
    p.write_text(yaml.safe_dump({"icons": ["alpha"], "colors": ["red", "blue"]}))
    assert _load_vocab("v", {"file": str(p), "key": "colors"}, tmp_path).flat == {"red", "blue"}


# ── path resolution ─────────────────────────────────────────────────────


def test_resolve_path_absolute_is_untouched(tmp_path: Path):
    target = tmp_path / "zz_abs.yaml"
    target.write_text("x")
    assert _resolve_path(str(target), Path("/nonexistent")) == target


def test_vocab_path_falls_back_to_task_dir(tmp_path_factory, monkeypatch):
    """
    A bare filename in a task YAML must resolve against the task's own
    directory when it does not exist relative to the working directory.

    This is what lets the shipped tasks' `file: ../icons.yaml` work from
    any cwd — and the test that catches a regression in it.
    """
    taskdir = tmp_path_factory.mktemp("taskdir")
    (taskdir / "zz_vocab.yaml").write_text(yaml.safe_dump({"icons": ["alpha", "beta"]}))
    (taskdir / "zz_task.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "vocabularies": {"icons": {"file": "zz_vocab.yaml", "key": "icons"}},
                "fields": {"icon": {"type": "str", "vocab": "icons"}},
                "prompts": {"system": "s"},
            }
        )
    )
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))

    task = load_task(taskdir / "zz_task.yaml")
    assert task.vocabularies["icons"].flat == {"alpha", "beta"}


def test_vocab_path_prefers_cwd(tmp_path_factory, monkeypatch):
    """The mirror image: cwd wins when the file exists in both places."""
    taskdir = tmp_path_factory.mktemp("taskdir2")
    (taskdir / "zz_v.yaml").write_text(yaml.safe_dump({"icons": ["from_taskdir"]}))
    (taskdir / "zz_t.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "vocabularies": {"icons": {"file": "zz_v.yaml", "key": "icons"}},
                "fields": {"icon": {"type": "str", "vocab": "icons"}},
                "prompts": {"system": "s"},
            }
        )
    )
    cwd = tmp_path_factory.mktemp("cwd2")
    (cwd / "zz_v.yaml").write_text(yaml.safe_dump({"icons": ["from_cwd"]}))
    monkeypatch.chdir(cwd)

    task = load_task(taskdir / "zz_t.yaml")
    assert task.vocabularies["icons"].flat == {"from_cwd"}


# ── enum injection ──────────────────────────────────────────────────────


def test_inject_vocab_enums(mini_task: Path):
    task = load_task(mini_task)
    icons = sorted({"alpha", "beta", "gamma"})

    assert task.fields["icon"]["enum"] == icons
    # list items get both a concrete type and the enum
    assert task.fields["tags"]["items"]["type"] == "str"
    assert task.fields["tags"]["items"]["enum"] == icons
    # a field with a hand-written enum is left alone
    assert task.fields["label"]["enum"] == ["keep", "drop_me"]


def test_inject_vocab_enums_recurses_into_nested_objects(tmp_path: Path):
    (tmp_path / "zz_v.yaml").write_text(yaml.safe_dump({"icons": ["a", "b"]}))
    p = tmp_path / "zz_t.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "vocabularies": {"icons": {"file": "zz_v.yaml", "key": "icons"}},
                "fields": {
                    "cfg": {
                        "type": "object",
                        "fields": {"icon": {"type": "str", "vocab": "icons"}},
                    }
                },
                "prompts": {"system": "s"},
            }
        )
    )
    task = load_task(p)
    assert task.fields["cfg"]["fields"]["icon"]["enum"] == ["a", "b"]


# ── load_task contract ──────────────────────────────────────────────────


@pytest.mark.parametrize("missing", ["name", "fields", "prompts"])
def test_load_task_missing_required_key(tmp_path: Path, missing: str):
    raw = {"name": "t", "fields": {"a": {"type": "str"}}, "prompts": {"system": "s"}}
    del raw[missing]
    p = tmp_path / "zz_bad.yaml"
    p.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match=missing):
        load_task(p)


def test_load_task_defaults(tmp_path: Path):
    p = tmp_path / "zz_minimal.yaml"
    p.write_text(
        yaml.safe_dump({"name": "t", "fields": {"a": {"type": "str"}}, "prompts": {"system": "s"}})
    )
    task = load_task(p)

    assert task.seeds is None
    assert task.hooks is None
    assert task.vocabularies == {}
    assert task.rules == []
    assert task.exports == {}
    assert task.examples == []
    assert task.generation == {}
    assert task.model == {}


def test_load_task_loads_hooks_module(mini_task: Path):
    task = load_task(mini_task)
    assert callable(task.hooks.postprocess)
    assert "pairs" in task.hooks.EXPORTERS


# ── seeds ───────────────────────────────────────────────────────────────


def test_seeds_work_list_flattens_groups(mini_task: Path):
    task = load_task(mini_task)
    assert task.seeds.work_list() == [("g1", "s1"), ("g1", "s2"), ("g2", "s3")]
    assert task.seeds.per_seed == 4
    assert task.seeds.value_field == "seed"
    assert task.seeds.id_pattern == "{seed}_{counter:03d}"


def test_seeds_non_dict_values_wrap_into_default_group(tmp_path: Path):
    p = tmp_path / "zz_t.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "fields": {"a": {"type": "str"}},
                "prompts": {"system": "s"},
                "seeds": {"values": ["x", "y"]},
            }
        )
    )
    seeds: Seeds = load_task(p).seeds
    assert seeds.groups == {"default": ["x", "y"]}
    assert seeds.value_field == "seed"  # documented default
