"""
Dataset export: group-aware splits + per-format record builders.

Split config lives in the task YAML (`exports:`); custom formats come from
the task's hooks file via an EXPORTERS dict:

    EXPORTERS = {"sft": build_sft, "pairs": build_pairs}
    def build_sft(record, ctx) -> dict: ...

The built-in "raw" format passes records through unchanged. When
`split_by` names a field (e.g. concept_group), all records sharing that
value land in the same split — so eval measures generalization to unseen
groups, not memorized phrasings.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .task_config import TaskConfig


def split_records(
    records: list[dict[str, Any]],
    split_by: str | None,
    ratios: dict[str, float],
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split records; group-aware when split_by is set."""
    names = list(ratios.keys())

    if split_by:
        for i, r in enumerate(records):
            if split_by not in r:
                raise ValueError(
                    f"record {i} has no '{split_by}' field to split on; "
                    f"exports.split_by names a field every record must carry"
                )
        keys = sorted({r[split_by] for r in records})
    else:
        keys = list(range(len(records)))

    rng = random.Random(seed)
    rng.shuffle(keys)

    assignment: dict[Any, str] = {}
    start = 0
    for i, name in enumerate(names):
        end = len(keys) if i == len(names) - 1 else start + round(len(keys) * ratios[name])
        for k in keys[start:end]:
            assignment[k] = name
        start = end

    out: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for idx, r in enumerate(records):
        key = r[split_by] if split_by else idx
        out[assignment[key]].append(r)
    return out


def export_dataset(
    task: TaskConfig,
    input_path: str | Path,
    output_dir: str | Path,
    *,
    formats: list[str] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in Path(input_path).read_text().strip().split("\n")
        if line.strip()
    ]

    exp_cfg = task.exports
    ratios = exp_cfg.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})
    split_by = exp_cfg.get("split_by")
    split_seed = seed if seed is not None else exp_cfg.get("seed", 42)

    exporters: dict[str, Any] = {"raw": lambda r, ctx: r}
    if task.hooks and hasattr(task.hooks, "EXPORTERS"):
        exporters.update(task.hooks.EXPORTERS)

    if formats is None:
        formats = exp_cfg.get("formats", list(exporters.keys()))
    unknown = [f for f in formats if f not in exporters]
    if unknown:
        raise ValueError(
            f"Unknown export formats {unknown}; available: {sorted(exporters)}"
        )

    ctx = {
        "vocab": {n: v.flat for n, v in task.vocabularies.items()},
        "config": task.raw,
    }

    splits = split_records(records, split_by, ratios, split_seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Any] = {}
    for split_name, split_recs in splits.items():
        for fmt in formats:
            path = out_dir / f"{fmt}_{split_name}.jsonl"
            with open(path, "w") as f:
                for r in split_recs:
                    f.write(
                        json.dumps(exporters[fmt](r, ctx), ensure_ascii=False) + "\n"
                    )
            written[f"{fmt}_{split_name}"] = len(split_recs)

    if split_by:
        written["groups_per_split"] = {
            s: len({r[split_by] for r in recs}) for s, recs in splits.items()
        }
    return written
