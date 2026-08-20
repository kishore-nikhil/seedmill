"""
Task config loader.

A "task" is one synthetic-data use case, fully described by a YAML file:

    name / description
    model:          backend + model + sampling defaults
    vocabularies:   named value sets loaded from files (or inline lists);
                    fields referencing them get enum validation AND
                    enum-constrained decoding
    seeds:          optional iteration axis (e.g. concept groups) — the
                    engine loops over seeds instead of free-running batches
    fields:         output record schema (drives pydantic validation and
                    the constrained-decoding JSON schema)
    rules:          declarative cross-field consistency checks/fixes
    prompts:        system / user / retry templates ($var substitution)
    generation:     count, batch_size, dedup, retries
    exports:        split config; custom formats come from the hooks file
    hooks:          optional python file with postprocess() / EXPORTERS
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


@dataclass
class Vocab:
    name: str
    flat: set[str]
    grouped: dict[str, list[str]] | None  # preserved grouping for prompts

    def prompt_block(self) -> str:
        if self.grouped:
            return "\n".join(
                f"{group}: {', '.join(vals)}" for group, vals in self.grouped.items()
            )
        return ", ".join(sorted(self.flat))


@dataclass
class Seeds:
    groups: dict[str, list[str]]  # {group_name: [seed, ...]}
    group_field: str | None       # record field that stores the group name
    value_field: str              # record field that stores the seed value
    per_seed: int
    id_pattern: str | None        # e.g. "{seed}_{counter:03d}"

    def work_list(self) -> list[tuple[str, str]]:
        return [(g, s) for g, seeds in self.groups.items() for s in seeds]


@dataclass
class TaskConfig:
    path: Path
    raw: dict[str, Any]
    name: str
    description: str
    fields: dict[str, Any]
    vocabularies: dict[str, Vocab] = dc_field(default_factory=dict)
    seeds: Seeds | None = None
    prompts: dict[str, str] = dc_field(default_factory=dict)
    generation: dict[str, Any] = dc_field(default_factory=dict)
    model: dict[str, Any] = dc_field(default_factory=dict)
    rules: list[dict[str, Any]] = dc_field(default_factory=list)
    exports: dict[str, Any] = dc_field(default_factory=dict)
    examples: list[dict[str, Any]] = dc_field(default_factory=list)
    hooks: ModuleType | None = None


def _resolve_path(spec_value: str | Path, base_dir: Path) -> Path:
    """
    Resolve a path written inside a task YAML.

    Working directory first, then the task YAML's own directory. The cwd
    branch is what lets `file: config/icons.yaml` work when you run from
    the repo root; the base_dir fallback is what lets `file: ../icons.yaml`
    work from anywhere, which is how the shipped tasks are written.
    """
    path = Path(spec_value)
    if not path.is_absolute() and not path.exists():
        path = base_dir / path
    return path


def _load_vocab(name: str, spec: Any, base_dir: Path) -> Vocab:
    """
    Vocab spec: inline list, or {file: ..., key: ...} pointing at YAML.

    Three file shapes are accepted:

      grouped   {group: [value, ...]}     — groups are kept for the prompt
      flat      [value, ...]
      mapping   {value: <anything>}       — e.g. an icon slug -> Flutter
                                            codePoint table exported from
                                            the app; the KEYS are the
                                            vocabulary and the values ride
                                            along unused

    The mapping shape means the app's own icon table can be the single
    source of truth for the label space: no hand-kept second list to drift
    out of sync with what the app can actually render.
    """
    if isinstance(spec, list):
        return Vocab(name=name, flat=set(spec), grouped=None)

    path = _resolve_path(spec["file"], base_dir)
    with open(path) as f:
        data = yaml.safe_load(f)
    if key := spec.get("key"):
        data = data[key]

    if isinstance(data, dict):
        if all(isinstance(v, list) for v in data.values()):
            flat = {v for vals in data.values() for v in vals}
            return Vocab(name=name, flat=flat, grouped=data)
        # mapping shape: keys are the vocabulary
        return Vocab(name=name, flat={str(k) for k in data}, grouped=None)
    return Vocab(name=name, flat=set(data), grouped=None)


def _load_seeds(spec: dict[str, Any] | None, base_dir: Path) -> Seeds | None:
    if not spec:
        return None

    if "values" in spec:
        data = spec["values"]
    else:
        path = _resolve_path(spec["file"], base_dir)
        with open(path) as f:
            data = yaml.safe_load(f)
        if key := spec.get("key"):
            data = data[key]

    groups = data if isinstance(data, dict) else {"default": data}
    return Seeds(
        groups=groups,
        group_field=spec.get("group_field"),
        value_field=spec.get("value_field", "seed"),
        per_seed=spec.get("per_seed", 8),
        id_pattern=spec.get("id_pattern"),
    )


def _inject_vocab_enums(
    fields: dict[str, Any], vocabularies: dict[str, Vocab]
) -> None:
    """
    Resolve `vocab: <name>` references into enum constraints, in place.

    Works on plain fields and on list item specs, so both validation and
    the constrained-decoding schema enforce vocabulary membership.
    """
    for fname, spec in fields.items():
        if vocab_name := spec.get("vocab"):
            spec["enum"] = sorted(vocabularies[vocab_name].flat)
        items = spec.get("items")
        if items and (vocab_name := items.get("vocab")):
            items.setdefault("type", "str")
            items["enum"] = sorted(vocabularies[vocab_name].flat)
        if spec.get("type") == "object" and "fields" in spec:
            _inject_vocab_enums(spec["fields"], vocabularies)


def _load_hooks(spec: str | None, base_dir: Path) -> ModuleType | None:
    if not spec:
        return None
    path = _resolve_path(spec, base_dir)
    module_spec = importlib.util.spec_from_file_location(f"hooks_{path.stem}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_task(path: str | Path) -> TaskConfig:
    """
    Load and resolve a task YAML into a TaskConfig.

    Paths inside the config (`vocabularies`, `seeds`, `hooks`) resolve
    against the working directory first, then against the task YAML's own
    directory — see `_resolve_path`.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    for key in ("name", "fields", "prompts"):
        if key not in raw:
            raise ValueError(f"Task config missing required key: '{key}'")

    base_dir = path.parent

    vocabularies = {
        name: _load_vocab(name, spec, base_dir)
        for name, spec in (raw.get("vocabularies") or {}).items()
    }

    fields = raw["fields"]
    _inject_vocab_enums(fields, vocabularies)

    return TaskConfig(
        path=path,
        raw=raw,
        name=raw["name"],
        description=raw.get("description", ""),
        fields=fields,
        vocabularies=vocabularies,
        seeds=_load_seeds(raw.get("seeds"), base_dir),
        prompts=raw["prompts"],
        generation=raw.get("generation") or {},
        model=raw.get("model") or {},
        rules=raw.get("rules") or [],
        exports=raw.get("exports") or {},
        examples=raw.get("examples") or [],
        hooks=_load_hooks(raw.get("hooks"), base_dir),
    )
