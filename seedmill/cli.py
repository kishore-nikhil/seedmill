"""
Unified task-driven CLI. One command for every synthetic-data use case;
new use cases are added as YAML files under config/tasks/ (see README).

Run from the repo root — paths inside a task YAML resolve against the
working directory first, then against the task YAML's own directory.

Usage:
    seedmill generate -t config/tasks/icon_classifier.yaml
    seedmill generate -t config/tasks/icon_classifier.yaml --limit 2 --per-seed 3
    seedmill export   -t config/tasks/icon_classifier.yaml
    seedmill stats    -t config/tasks/icon_classifier.yaml
    seedmill tasks

Also runnable from a clone without installing: `python -m seedmill ...`
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .engine import TaskEngine
from .exporter import export_dataset
from .task_config import TaskConfig, load_task

app = typer.Typer(
    name="seedmill",
    help="Config-driven synthetic data generator (local models).",
    add_completion=False,
)
console = Console()


def _default_output(task: TaskConfig) -> str:
    return f"output/{task.name}.jsonl"


def _make_client(
    task: TaskConfig,
    model: str | None,
    base_url: str | None,
    backend_override: str | None = None,
):
    cfg = task.model
    backend = backend_override or cfg.get("backend", "ollama")
    temperature = task.generation.get("temperature", 0.8)

    if backend == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(
            base_url=base_url or cfg.get("base_url", "http://localhost:11434"),
            model=model or cfg.get("model", "gemma4:27b"),
            default_temperature=temperature,
            think=cfg.get("think", False),
        )
    if backend == "openai_compat":
        from .client import LlamaClient

        return LlamaClient(
            base_url=base_url or cfg.get("base_url", "http://localhost:8080/v1"),
            model=model or cfg.get("model", "local-model"),
            default_temperature=temperature,
            think=cfg.get("think", True),
        )
    raise typer.BadParameter(f"Unknown backend: {backend}")


@app.command()
def generate(
    task_path: str = typer.Option(..., "--task", "-t", help="Task YAML path"),
    output: str = typer.Option(None, "--output", "-o"),
    model: str = typer.Option(None, "--model", "-m", help="Override task model"),
    base_url: str = typer.Option(None, "--base-url"),
    backend: str = typer.Option(
        None, "--backend", help="Override backend: ollama | openai_compat (llama.cpp)"
    ),
    per_seed: int = typer.Option(None, "--per-seed", help="Override records per seed"),
    count: int = typer.Option(None, "--count", "-n", help="Override total count (batch mode)"),
    seed_filter: list[str] = typer.Option(
        None, "--seed", help="Only these seed values (repeatable)"
    ),
    limit: int = typer.Option(None, "--limit", help="Only first N seeds (smoke tests)"),
    workers: int = typer.Option(
        None, "--workers", help="Concurrent seeds (seeded mode); server slots permitting"
    ),
    unconstrained: bool = typer.Option(
        False, "--unconstrained", help="Disable schema-constrained decoding"
    ),
) -> None:
    """Generate the raw dataset for a task (resumable)."""
    task = load_task(task_path)
    if per_seed is not None and task.seeds:
        task.seeds.per_seed = per_seed
    if count is not None:
        task.generation["count"] = count

    out = output or _default_output(task)
    client = _make_client(task, model, base_url, backend)

    mode = "seeded" if task.seeds else "batch"
    console.print(Panel.fit(
        f"[bold]{task.name}[/bold] ({mode} mode)\n"
        f"Backend: {task.model.get('backend', 'ollama')}  "
        f"Model: {client.model}\n"
        f"Output: {out}",
        title="Configuration",
    ))

    engine = TaskEngine(
        task, client, out, constrained=not unconstrained, workers=workers
    )
    stats = engine.run(
        only_seeds=list(seed_filter) if seed_filter else None, limit=limit
    )

    table = Table(title="Generation Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in stats.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)


@app.command()
def export(
    task_path: str = typer.Option(..., "--task", "-t"),
    input_file: str = typer.Option(None, "--input", "-i"),
    output_dir: str = typer.Option(None, "--output-dir", "-o"),
    fmt: list[str] = typer.Option(None, "--format", "-f", help="Formats (repeatable)"),
    seed: int = typer.Option(None, "--seed", help="Split seed override"),
) -> None:
    """Export split datasets in the task's configured formats."""
    task = load_task(task_path)
    inp = input_file or _default_output(task)
    out_dir = output_dir or f"output/{task.name}_export"

    written = export_dataset(
        task, inp, out_dir, formats=list(fmt) if fmt else None, seed=seed
    )
    console.print(json.dumps(written, indent=2))
    console.print(f"[green]Exported to {out_dir}/[/green]")


@app.command()
def stats(
    task_path: str = typer.Option(..., "--task", "-t"),
    input_file: str = typer.Option(None, "--input", "-i"),
) -> None:
    """Dataset summary: totals, per-group counts, top enum values."""
    task = load_task(task_path)
    inp = input_file or _default_output(task)
    records = [
        json.loads(line)
        for line in Path(inp).read_text().strip().split("\n")
        if line.strip()
    ]

    summary: dict = {"total": len(records)}
    if task.seeds:
        summary["seeds_covered"] = len(
            {r.get(task.seeds.value_field) for r in records}
        )
        if task.seeds.group_field:
            summary["by_" + task.seeds.group_field] = dict(
                Counter(r.get(task.seeds.group_field) for r in records)
            )
    for fname, spec in task.fields.items():
        if spec.get("enum") and spec.get("type", "str") == "str":
            summary["top_" + fname] = dict(
                Counter(r.get(fname) for r in records).most_common(10)
            )
    console.print(json.dumps(summary, indent=2))


@app.command()
def tasks(
    tasks_dir: str = typer.Option("config/tasks", "--dir"),
) -> None:
    """List available task configs."""
    table = Table(title="Tasks")
    table.add_column("Task", style="cyan")
    table.add_column("Mode")
    table.add_column("Description")
    for path in sorted(Path(tasks_dir).glob("*.yaml")):
        task = load_task(path)
        table.add_row(
            str(path),
            "seeded" if task.seeds else "batch",
            task.description.strip()[:70],
        )
    console.print(table)


if __name__ == "__main__":
    app()
