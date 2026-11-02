"""
Task config loader.

A "task" is one synthetic-data use case, fully described by a YAML file:

    name / description
    model:          backend + model + sampling defaults
    vocabularies:   named value sets loaded from files (or inline lists);
                    fields referencing them get enum validation AND
                    enum-constrained decoding
    seeds:          optional iteration axis (e.g. concept groups, or one
                    seed per file) — the engine loops over seeds instead
                    of free-running batches
    fields:         output record schema (drives pydantic validation and
                    the constrained-decoding JSON schema)
    rules:          declarative cross-field consistency checks/fixes
    prompts:        system / user / retry templates ($var substitution),
                    plus images attached to the user message
    generation:     count, batch_size, dedup, retries
    exports:        split config; custom formats come from the hooks file
    hooks:          optional python file with postprocess() / EXPORTERS
"""

from __future__ import annotations

import glob
import importlib.util
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from .images import MIME_TYPES


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
    paths: dict[str, Path] = dc_field(default_factory=dict)
    """File seeds only: seed -> the file it names, for `$seed_path`."""

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
    images: list[str] = dc_field(default_factory=list)
    """`prompts.images`: path templates, attached to the user message."""
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

    if "files" in spec:
        if "values" in spec or "file" in spec:
            raise ValueError(
                "seeds: `files` makes one seed per file; it can't be combined "
                "with `values` or `file`, which list the seeds themselves"
            )
        groups, paths = _file_seeds(spec["files"], base_dir)
        return Seeds(
            groups=groups,
            group_field=spec.get("group_field"),
            value_field=spec.get("value_field", "seed"),
            per_seed=spec.get("per_seed", 1),
            id_pattern=spec.get("id_pattern"),
            paths=paths,
        )

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


def _file_seeds(
    patterns: str | list[str], base_dir: Path
) -> tuple[dict[str, list[str]], dict[str, Path]]:
    """
    One seed per file matched by `seeds.files`.

    The seed is the file's path below the glob's fixed part, so
    `scans/**/*.jpg` makes `nemo/1906-08-26.jpg` from
    `scans/nemo/1906-08-26.jpg`. Keeping the folders in the seed is what
    stops two strips' pages of the same date colliding, and resume
    matches on it. The group is the first folder below the fixed part, or
    the fixed part's own folder for files directly in it.
    """
    groups: dict[str, list[str]] = {}
    paths: dict[str, Path] = {}
    for pattern in [patterns] if isinstance(patterns, str) else patterns:
        root, files = _glob_files(pattern, base_dir)
        if not files:
            raise ValueError(
                f"seeds.files: '{pattern}' matched no files, looking from "
                f"{Path.cwd()} and from {base_dir.resolve()}"
            )
        for file in files:
            rel = file.relative_to(root)
            seed = rel.as_posix()
            if seed in paths and paths[seed] != file.resolve():
                raise ValueError(
                    f"seeds.files: two files make the seed '{seed}': "
                    f"{paths[seed]} and {file.resolve()}"
                )
            if seed in paths:
                continue
            paths[seed] = file.resolve()
            group = rel.parts[0] if len(rel.parts) > 1 else (root.resolve().name or "default")
            groups.setdefault(group, []).append(seed)
    return groups, paths


def _glob_files(pattern: str, base_dir: Path) -> tuple[Path, list[Path]]:
    """
    Expand one glob from the working directory, else from the task YAML's
    directory: the order `_resolve_path` uses. Returns the fixed part (the
    folders before the first wildcard) and the files, sorted.
    """
    candidate = Path(pattern)
    candidates = [candidate] if candidate.is_absolute() else [candidate, base_dir / candidate]
    for candidate in candidates:
        parts = candidate.parts
        fixed = next(
            (i for i, part in enumerate(parts) if any(c in part for c in "*?[")), None
        )
        root = candidate.parent if fixed is None else Path(*parts[:fixed]) if fixed else Path()
        files = sorted(
            Path(p) for p in glob.glob(str(candidate), recursive=True) if Path(p).is_file()
        )
        if files:
            return root, files
    return Path(), []


def _load_images(spec: str | list[str] | None, seeds: Seeds | None, base_dir: Path) -> list[str]:
    """
    `prompts.images`: one path or a list. Entries are `$var` templates,
    rendered per batch; a plain path is checked here, before the run.
    """
    if not spec:
        return []
    images = [spec] if isinstance(spec, str) else list(spec)
    for entry in images:
        if "seed_path" in entry and not (seeds and seeds.paths):
            raise ValueError(
                f"prompts.images: '{entry}' uses $seed_path, which only "
                "file seeds (`seeds.files`) set"
            )
        if "$" in entry:
            continue
        path = _resolve_path(entry, base_dir)
        if not path.is_file():
            raise ValueError(f"prompts.images: {entry} not found")
        if path.suffix.lower() not in MIME_TYPES:
            raise ValueError(
                f"prompts.images: can't send '{path.suffix}' files as images. "
                f"Use one of: {', '.join(sorted(MIME_TYPES))}"
            )
    return images


def _inject_vocab_enums(
    fields: dict[str, Any], vocabularies: dict[str, Vocab]
) -> None:
    """
    Resolve `vocab: <name>` references into enum constraints, in place.

    Works on plain fields, list item specs and sub-fields at any depth, so
    both validation and the constrained-decoding schema enforce vocabulary
    membership.
    """
    for spec in fields.values():
        _inject_vocab_enum(spec, vocabularies)


def _inject_vocab_enum(spec: dict[str, Any], vocabularies: dict[str, Vocab]) -> None:
    if vocab_name := spec.get("vocab"):
        spec["enum"] = sorted(vocabularies[vocab_name].flat)
    if items := spec.get("items"):
        if "vocab" in items:
            items.setdefault("type", "str")
        _inject_vocab_enum(items, vocabularies)
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

    Paths inside the config (`vocabularies`, `seeds`, `prompts.images`,
    `hooks`) resolve against the working directory first, then against the
    task YAML's own directory — see `_resolve_path`.
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

    seeds = _load_seeds(raw.get("seeds"), base_dir)
    prompts = dict(raw["prompts"])
    images = _load_images(prompts.pop("images", None), seeds, base_dir)

    return TaskConfig(
        path=path,
        raw=raw,
        name=raw["name"],
        description=raw.get("description", ""),
        fields=fields,
        vocabularies=vocabularies,
        seeds=seeds,
        prompts=prompts,
        images=images,
        generation=raw.get("generation") or {},
        model=raw.get("model") or {},
        rules=raw.get("rules") or [],
        exports=raw.get("exports") or {},
        examples=raw.get("examples") or [],
        hooks=_load_hooks(raw.get("hooks"), base_dir),
    )
