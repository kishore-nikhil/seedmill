"""
Task-driven generation engine.

One loop for every use case. Reads a TaskConfig and runs either:

- seeded mode (task has `seeds`): iterate seed values (e.g. concept
  groups), generate `per_seed` records each, assign patterned ids, and
  stamp the seed/group into each record; or
- batch mode: generate `generation.count` records in batches.

Pipeline per batch: constrained-decode JSON → pydantic validation →
declarative rules (with auto-fix) → field bounds → hooks.postprocess →
dedup → append to JSONL. Reruns resume from the existing output file.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template
from typing import Any, Protocol

from rich.console import Console

from .dedup import DedupFilter
from .rules import apply_rules, check_field_bounds
from .schema import build_json_schema_for_response, build_schema_json
from .task_config import TaskConfig
from .validator import build_pydantic_model, validate_single

console = Console()


class JsonChatClient(Protocol):
    """Any client with chat_json(messages, temperature=, schema=) works."""

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def _render(template: str, variables: dict[str, Any]) -> str:
    """$var substitution — safe with JSON braces in templates."""
    return Template(template).safe_substitute(variables)


def _render_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return ""
    lines = ["Reference examples (generate NEW, different records):\n"]
    for i, ex in enumerate(examples, 1):
        if "query" in ex and "output" in ex:
            lines.append(f"Example {i}:")
            lines.append(f'  Query: "{ex["query"]}"')
            lines.append(f"  Output: {json.dumps(ex['output'], indent=2)}")
        else:
            lines.append(f"Example {i}: {json.dumps(ex, indent=2)}")
        lines.append("")
    return "\n".join(lines)


class TaskEngine:
    def __init__(
        self,
        task: TaskConfig,
        client: JsonChatClient,
        output_path: str | Path,
        *,
        constrained: bool = True,
        workers: int | None = None,
    ):
        self.task = task
        self.client = client
        self.output_path = Path(output_path)

        gen = task.generation
        self.batch_size = gen.get("batch_size", 4)
        self.max_retries = gen.get("max_retries", 2)
        self.temperature = gen.get("temperature", 0.8)
        self.count = gen.get("count", 100)
        # Seeds are independent units of work, so seeded mode can keep more
        # than one of the server's parallel slots busy. Local llama.cpp is
        # memory-bandwidth bound — measured ~1.4x aggregate throughput at 3
        # workers on this box, not 3x — so keep expectations modest.
        self.workers = max(1, int(workers or gen.get("workers", 1)))
        self._lock = threading.Lock()

        self.model = build_pydantic_model({"name": task.name, "fields": task.fields})
        self.response_schema = (
            build_json_schema_for_response(task.fields) if constrained else None
        )

        dedup_cfg = gen.get("dedup") or {}
        self.dedup_fields: list[str] = dedup_cfg.get(
            "fields", list(task.fields.keys())
        )
        self.dedup = DedupFilter(
            similarity_threshold=dedup_cfg.get("threshold", 0.92),
            use_embeddings=dedup_cfg.get("embeddings", False),
            skip_length_check_fields=set(task.fields.keys()),
        )

        # Static prompt variables shared by every call
        self.base_vars: dict[str, Any] = {
            "name": task.name,
            "description": task.description,
            "schema_json": build_schema_json(task.fields),
            "examples": _render_examples(task.examples),
        }
        for vname, vocab in task.vocabularies.items():
            self.base_vars[f"vocab_{vname}"] = vocab.prompt_block()

        self.stats = {
            "accepted": 0,
            "rejected_validation": 0,
            "rejected_rules": 0,
            "rejected_dedup": 0,
            "retries": 0,
            "api_errors": 0,
        }

    # ── shared batch machinery ──────────────────────────────────────

    def _messages(self, variables: dict[str, Any]) -> list[dict[str, str]]:
        merged = {**self.base_vars, **variables}
        return [
            {"role": "system", "content": _render(self.task.prompts["system"], merged)},
            {"role": "user", "content": _render(self.task.prompts["user"], merged)},
        ]

    def _process_raw(
        self, raw: Any, variables: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Validate + rules + bounds + hooks for one raw LLM response."""
        records = raw.get("records", []) if isinstance(raw, dict) else []
        valid: list[dict[str, Any]] = []
        errors: list[str] = []

        for r in records:
            clean, err = validate_single(r, self.model)
            if clean is None:
                errors.append(str(err)[:120])
                self.stats["rejected_validation"] += 1
                continue

            clean, issues = apply_rules(clean, self.task.rules)
            if clean is None:
                errors.extend(issues)
                self.stats["rejected_rules"] += 1
                continue

            issues = check_field_bounds(clean, self.task.fields)
            if issues:
                errors.extend(issues)
                self.stats["rejected_rules"] += 1
                continue

            if self.task.hooks and hasattr(self.task.hooks, "postprocess"):
                clean = self.task.hooks.postprocess(clean, self._hook_ctx(variables))
                if clean is None:
                    self.stats["rejected_rules"] += 1
                    continue

            valid.append(clean)

        return valid, errors

    def _hook_ctx(self, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        # `variables` carries the current seed/group in seeded mode, so hooks
        # can apply per-seed conventions (e.g. the canonical icon for a
        # domain) without the engine knowing what a seed means.
        return {
            "vocab": {n: v.flat for n, v in self.task.vocabularies.items()},
            "config": self.task.raw,
            "variables": variables or {},
            "seed": (variables or {}).get("seed"),
            "group": (variables or {}).get("group"),
        }

    def _dedup_one(self, record: dict[str, Any]) -> bool:
        subset = {f: record.get(f) for f in self.dedup_fields}
        with self._lock:
            accepted, _ = self.dedup.filter_batch([subset], check_quality=False)
        return bool(accepted)

    def _generate_batch(
        self, variables: dict[str, Any], n: int
    ) -> list[dict[str, Any]]:
        """One prompt → validated records, with error-feedback retries."""
        messages = self._messages({**variables, "n": n})

        for attempt in range(self.max_retries + 1):
            try:
                raw = self.client.chat_json(
                    messages,
                    temperature=self.temperature,
                    schema=self.response_schema,
                )
            except Exception as e:
                self.stats["api_errors"] += 1
                console.print(f"  [red]API error: {e}[/red]")
                time.sleep(1)
                continue

            valid, errors = self._process_raw(raw, variables)
            if valid:
                return valid

            if attempt < self.max_retries and errors:
                self.stats["retries"] += 1
                retry = self.task.prompts.get(
                    "retry",
                    "These records failed validation:\n$errors\n"
                    "Regenerate exactly $n valid records. JSON only.",
                )
                messages = messages[:2] + [
                    {
                        "role": "user",
                        "content": _render(
                            retry,
                            {"errors": "\n".join(f"- {e}" for e in errors[:5]), "n": n},
                        ),
                    }
                ]
                console.print(f"  [yellow]retry: {errors[0][:70]}[/yellow]")

        return []

    # ── seeded mode ─────────────────────────────────────────────────

    def _run_seeded(
        self,
        only_seeds: list[str] | None,
        limit: int | None,
    ) -> dict[str, Any]:
        seeds = self.task.seeds
        assert seeds is not None

        # Resume: per-seed counts from existing output
        existing: dict[str, int] = defaultdict(int)
        if self.output_path.exists():
            for line in self.output_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                existing[rec.get(seeds.value_field, "")] += 1
                self._dedup_one(rec)
            console.print(
                f"[cyan]Resuming: {sum(existing.values())} existing records[/cyan]"
            )

        work = seeds.work_list()
        if only_seeds:
            work = [(g, s) for g, s in work if s in only_seeds]
        if limit:
            work = work[:limit]

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        def run_seed(i: int, group: str, seed: str) -> None:
            needed = seeds.per_seed - existing.get(seed, 0)
            if needed <= 0:
                return
            console.print(f"[bold]({i}/{len(work)})[/bold] {group}/{seed} — need {needed}")

            collected: list[dict[str, Any]] = []
            attempts = 0
            max_attempts = (needed // self.batch_size + 2) * (self.max_retries + 1)
            while len(collected) < needed and attempts < max_attempts:
                attempts += 1
                n = min(self.batch_size, needed - len(collected))
                batch = self._generate_batch(
                    {
                        "seed": seed,
                        "seed_readable": seed.replace("_", " "),
                        "group": group,
                    },
                    n,
                )
                for rec in batch:
                    if self._dedup_one(rec):
                        collected.append(rec)
                    else:
                        self.stats["rejected_dedup"] += 1

            # One lock for the id counter and the append together: with
            # workers > 1 the file is shared, and ids must not collide.
            with self._lock:
                counter = existing.get(seed, 0)
                with open(self.output_path, "a") as f:
                    for rec in collected[:needed]:
                        counter += 1
                        full: dict[str, Any] = {}
                        if seeds.id_pattern:
                            full["id"] = seeds.id_pattern.format(
                                seed=seed, counter=counter
                            )
                        full[seeds.value_field] = seed
                        full.update(rec)
                        if seeds.group_field:
                            full[seeds.group_field] = group
                        f.write(json.dumps(full, ensure_ascii=False) + "\n")
                        self.stats["accepted"] += 1
                existing[seed] = counter

            if len(collected) < needed:
                console.print(f"  [yellow]{seed}: only got {len(collected)}/{needed}[/yellow]")

        if self.workers > 1:
            console.print(f"[cyan]{self.workers} workers[/cyan]")
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(
                    lambda item: run_seed(item[0], item[1][0], item[1][1]),
                    enumerate(work, 1),
                ))
        else:
            for i, (group, seed) in enumerate(work, 1):
                run_seed(i, group, seed)

        return self.stats

    # ── batch mode ──────────────────────────────────────────────────

    def _run_batch(self) -> dict[str, Any]:
        accepted = 0
        if self.output_path.exists():
            for line in self.output_path.read_text().strip().split("\n"):
                if line.strip():
                    self._dedup_one(json.loads(line))
                    accepted += 1
            console.print(f"[cyan]Resuming: {accepted} existing records[/cyan]")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        stall = 0
        while accepted < self.count and stall < 10:
            n = min(self.batch_size, self.count - accepted)
            batch = self._generate_batch({}, n)
            wrote = 0
            with open(self.output_path, "a") as f:
                for rec in batch:
                    if self._dedup_one(rec):
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        wrote += 1
                    else:
                        self.stats["rejected_dedup"] += 1
            accepted += wrote
            self.stats["accepted"] += wrote
            stall = 0 if wrote else stall + 1
            console.print(f"  [dim]{accepted}/{self.count}[/dim]")

        return self.stats

    # ── entry point ─────────────────────────────────────────────────

    def run(
        self,
        only_seeds: list[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        if self.task.seeds:
            return self._run_seeded(only_seeds, limit)
        return self._run_batch()
