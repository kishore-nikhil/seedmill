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


def test_inject_vocab_enums_reaches_objects_inside_lists(tmp_path: Path):
    """panels[].balloons[].kind: a vocab two lists down still becomes an enum."""
    p = tmp_path / "zz_t.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "vocabularies": {"kinds": ["speech", "thought"]},
                "fields": {
                    "panels": {
                        "type": "list",
                        "items": {
                            "type": "object",
                            "fields": {
                                "balloons": {
                                    "type": "list",
                                    "items": {
                                        "type": "object",
                                        "fields": {"kind": {"type": "str", "vocab": "kinds"}},
                                    },
                                },
                                "tags": {"type": "list", "items": {"vocab": "kinds"}},
                            },
                        },
                    }
                },
                "prompts": {"system": "s", "user": "u"},
            }
        )
    )
    panel = load_task(p).fields["panels"]["items"]["fields"]
    assert panel["balloons"]["items"]["fields"]["kind"]["enum"] == ["speech", "thought"]
    assert panel["tags"]["items"] == {
        "vocab": "kinds", "type": "str", "enum": ["speech", "thought"]
    }


# ── file seeds ──────────────────────────────────────────────────────────


def _file_task(tmp_path: Path, seeds: dict, prompts: dict | None = None) -> Path:
    p = tmp_path / "zz_files_task.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "fields": {"a": {"type": "str"}},
                "prompts": prompts or {"system": "s", "user": "u"},
                "seeds": seeds,
            }
        )
    )
    return p


def _touch(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG")


def test_file_seeds_one_per_file_grouped_by_first_folder(tmp_path: Path):
    """The glob is relative and not under the cwd, so this also exercises
    the fall-back to the task YAML's directory."""
    _touch(
        tmp_path / "zz_scans",
        "nemo/1906-08-26.png",
        "nemo/1906-08-19.png",
        "krazy/1906-08-26.png",
        "cover.png",
        "nemo/notes.txt",
    )
    seeds = load_task(_file_task(tmp_path, {"files": "zz_scans/**/*.png"})).seeds

    # Folders stay in the seed, so the two 1906-08-26 pages don't collide.
    assert seeds.work_list() == [
        ("zz_scans", "cover.png"),
        ("krazy", "krazy/1906-08-26.png"),
        ("nemo", "nemo/1906-08-19.png"),
        ("nemo", "nemo/1906-08-26.png"),
    ]
    page = tmp_path / "zz_scans/nemo/1906-08-26.png"
    assert seeds.paths["nemo/1906-08-26.png"] == page.resolve()
    assert seeds.per_seed == 1  # one record per file, unless asked


def test_file_seeds_from_a_list_of_globs(tmp_path: Path):
    _touch(tmp_path / "zz_a", "x.png")
    _touch(tmp_path / "zz_b", "y.jpg")
    seeds = load_task(
        _file_task(tmp_path, {"files": ["zz_a/*.png", "zz_b/*.jpg"], "per_seed": 2})
    ).seeds
    assert seeds.groups == {"zz_a": ["x.png"], "zz_b": ["y.jpg"]}
    assert seeds.per_seed == 2


def test_file_seeds_two_files_making_one_seed_is_an_error(tmp_path: Path):
    _touch(tmp_path / "zz_a", "x.png")
    _touch(tmp_path / "zz_b", "x.png")
    with pytest.raises(ValueError, match="two files make the seed 'x.png'"):
        load_task(_file_task(tmp_path, {"files": ["zz_a/*.png", "zz_b/*.png"]}))


def test_file_seeds_overlapping_globs_count_a_file_once(tmp_path: Path):
    _touch(tmp_path / "zz_a", "x.png")
    seeds = load_task(_file_task(tmp_path, {"files": ["zz_a/*.png", "zz_a/x.png"]})).seeds
    assert seeds.work_list() == [("zz_a", "x.png")]


def test_file_seeds_matching_nothing_fail_before_the_run(tmp_path: Path):
    with pytest.raises(ValueError, match="matched no files"):
        load_task(_file_task(tmp_path, {"files": "zz_nowhere/*.png"}))


def test_file_seeds_cannot_be_mixed_with_listed_seeds(tmp_path: Path):
    _touch(tmp_path / "zz_a", "x.png")
    with pytest.raises(ValueError, match="can't be combined"):
        load_task(_file_task(tmp_path, {"files": "zz_a/*.png", "values": ["s"]}))


# ── prompts.images ──────────────────────────────────────────────────────


def test_images_come_out_of_prompts_onto_the_task(tmp_path: Path):
    _touch(tmp_path / "zz_a", "x.png")
    task = load_task(
        _file_task(
            tmp_path,
            {"files": "zz_a/*.png"},
            {"system": "s", "user": "u", "images": "$seed_path"},
        )
    )
    assert task.images == ["$seed_path"]
    assert "images" not in task.prompts
    assert task.raw["prompts"]["images"] == "$seed_path"  # hooks still see the YAML as written


def test_seed_path_image_needs_file_seeds(tmp_path: Path):
    p = tmp_path / "zz_t.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "fields": {"a": {"type": "str"}},
                "prompts": {"system": "s", "user": "u", "images": ["$seed_path"]},
                "seeds": {"values": ["x"]},
            }
        )
    )
    with pytest.raises(ValueError, match="only file seeds"):
        load_task(p)


def test_a_plain_image_path_is_checked_at_load(tmp_path: Path):
    _touch(tmp_path / "zz_a", "x.png")
    (tmp_path / "zz_a" / "scan.tiff").write_bytes(b"II*")
    seeds = {"files": "zz_a/*.png"}

    ok = {"system": "s", "user": "u", "images": ["$seed_path", "zz_a/x.png"]}
    assert load_task(_file_task(tmp_path, seeds, ok)).images == ["$seed_path", "zz_a/x.png"]

    missing = {"system": "s", "user": "u", "images": ["zz_a/gone.png"]}
    with pytest.raises(ValueError, match="not found"):
        load_task(_file_task(tmp_path, seeds, missing))

    tiff = {"system": "s", "user": "u", "images": ["zz_a/scan.tiff"]}
    with pytest.raises(ValueError, match="can't send '.tiff'"):
        load_task(_file_task(tmp_path, seeds, tiff))
