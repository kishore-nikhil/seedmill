"""Group-aware splitting and per-format export."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pytest

from seedmill.exporter import export_dataset, split_records
from seedmill.task_config import load_task

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _records(n_groups=10, per_group=2):
    return [
        {"id": f"g{g}_{i}", "group": f"g{g}"}
        for g in range(n_groups)
        for i in range(per_group)
    ]


# ── split_records ───────────────────────────────────────────────────────


def test_group_never_straddles_splits():
    """
    The headline invariant. Everything the tool claims about eval
    measuring generalization to *unseen* groups rather than memorized
    phrasings rests on this holding.
    """
    splits = split_records(_records(), "group", RATIOS)
    seen = {name: {r["group"] for r in recs} for name, recs in splits.items()}
    for a, b in combinations(seen, 2):
        assert not (seen[a] & seen[b]), f"{a} and {b} share groups {seen[a] & seen[b]}"


def test_all_records_are_conserved_exactly_once():
    records = _records()
    splits = split_records(records, "group", RATIOS)
    ids = [r["id"] for recs in splits.values() for r in recs]
    assert sorted(ids) == sorted(r["id"] for r in records)
    assert len(ids) == len(set(ids))


def test_ratio_shape_over_groups():
    splits = split_records(_records(n_groups=10), "group", RATIOS)
    counts = {name: len({r["group"] for r in recs}) for name, recs in splits.items()}
    assert counts == {"train": 8, "val": 1, "test": 1}


def test_deterministic_for_the_same_seed():
    a = split_records(_records(), "group", RATIOS, seed=42)
    b = split_records(_records(), "group", RATIOS, seed=42)
    assert {k: [r["id"] for r in v] for k, v in a.items()} == {
        k: [r["id"] for r in v] for k, v in b.items()
    }


def test_different_seeds_change_the_assignment():
    a = split_records(_records(n_groups=20), "group", RATIOS, seed=1)
    b = split_records(_records(n_groups=20), "group", RATIOS, seed=2)
    assert {r["group"] for r in a["test"]} != {r["group"] for r in b["test"]}


def test_fewer_groups_than_splits_loses_nothing():
    """3 groups at .8/.1/.1 leaves `val` empty — but nothing crashes and
    every record still lands somewhere."""
    records = _records(n_groups=3, per_group=1)
    splits = split_records(records, "group", RATIOS)
    assert len(splits["val"]) == 0
    assert sum(len(v) for v in splits.values()) == len(records)


def test_split_by_none_assigns_per_record():
    records = _records(n_groups=5, per_group=2)
    splits = split_records(records, None, RATIOS)
    assert sum(len(v) for v in splits.values()) == len(records)


def test_every_split_name_is_present_even_when_empty():
    splits = split_records(_records(n_groups=3, per_group=1), "group", RATIOS)
    assert set(splits) == set(RATIOS)


def test_missing_split_by_field_raises_a_named_error():
    records = [{"id": "a", "group": "g1"}, {"id": "b"}]
    with pytest.raises(ValueError, match="group"):
        split_records(records, "group", RATIOS)


# ── export_dataset ──────────────────────────────────────────────────────


@pytest.fixture
def exportable(mini_task: Path, tmp_path: Path):
    task = load_task(mini_task)
    src = tmp_path / "zz_records.jsonl"
    src.write_text(
        "\n".join(
            json.dumps({"id": f"r{i}", "text": f"text {i}", "label": "keep", "group": f"g{i % 5}"})
            for i in range(10)
        )
    )
    return task, src, tmp_path / "out"


def test_raw_format_passes_records_through(exportable):
    task, src, out = exportable
    export_dataset(task, src, out, formats=["raw"])
    rows = [json.loads(x) for x in (out / "raw_train.jsonl").read_text().splitlines()]
    assert all(set(r) == {"id", "text", "label", "group"} for r in rows)


def test_custom_format_comes_from_the_hooks_module(exportable):
    task, src, out = exportable
    export_dataset(task, src, out, formats=["pairs"])
    rows = [json.loads(x) for x in (out / "pairs_train.jsonl").read_text().splitlines()]
    assert rows and all(set(r) == {"text", "label"} for r in rows)


def test_exporter_ctx_carries_vocab_sets_and_the_raw_config(exportable):
    task, src, out = exportable
    seen = {}

    def spy(record, ctx):
        seen.update(ctx)
        return record

    task.hooks.EXPORTERS["spy"] = spy
    try:
        export_dataset(task, src, out, formats=["spy"])
    finally:
        del task.hooks.EXPORTERS["spy"]

    assert seen["vocab"]["icons"] == {"alpha", "beta", "gamma"}
    assert isinstance(seen["vocab"]["icons"], set)
    assert seen["config"]["name"] == "mini task"


def test_unknown_format_raises_and_lists_what_is_available(exportable):
    task, src, out = exportable
    with pytest.raises(ValueError, match="nope"):
        export_dataset(task, src, out, formats=["nope"])


def test_output_filenames_are_format_split(exportable):
    task, src, out = exportable
    export_dataset(task, src, out, formats=["raw", "pairs"])
    written = {p.name for p in out.glob("*.jsonl")}
    assert written == {
        f"{fmt}_{split}.jsonl" for fmt in ("raw", "pairs") for split in ("train", "val", "test")
    }


def test_returned_counts_and_groups_per_split(exportable):
    task, src, out = exportable
    written = export_dataset(task, src, out, formats=["raw"])
    assert sum(v for k, v in written.items() if k.startswith("raw_")) == 10
    assert set(written["groups_per_split"]) == {"train", "val", "test"}


def test_groups_per_split_absent_when_split_by_is_unset(exportable):
    task, src, out = exportable
    task.exports.pop("split_by")
    written = export_dataset(task, src, out, formats=["raw"])
    assert "groups_per_split" not in written


def test_seed_argument_overrides_the_configured_seed(exportable, tmp_path):
    """
    exports.seed is 7 in the fixture; passing seed= must win. Compared by
    split *membership*, since counts can coincide across seeds.
    """
    task, src, _ = exportable

    def members(seed):
        out = tmp_path / f"out_{seed}"
        export_dataset(task, src, out, formats=["raw"], seed=seed)
        return {json.loads(x)["id"] for x in (out / "raw_test.jsonl").read_text().splitlines()}

    assert members(1) != members(999)
    assert members(7) == members(None)  # None falls back to exports.seed == 7


def test_non_ascii_round_trips_as_the_actual_character(exportable):
    task, src, out = exportable
    src.write_text(json.dumps({"id": "r0", "text": "café ☕", "label": "keep", "group": "g0"}))
    export_dataset(task, src, out, formats=["raw"])
    body = "".join(p.read_text() for p in out.glob("raw_*.jsonl"))
    assert "café ☕" in body
    assert "\\u" not in body
