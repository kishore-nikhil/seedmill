"""
Unified task-driven CLI. One command for every synthetic-data use case;
new use cases are added as YAML files under config/tasks/ (see README).

Run from the repo root — paths inside a task YAML resolve against the
working directory first, then against the task YAML's own directory.

Usage:
    seedmill init                     # scaffold an example task to edit
    seedmill generate -t config/tasks/icon_classifier.yaml
    seedmill generate -t config/tasks/icon_classifier.yaml --limit 2 --per-seed 3
    seedmill export   -t config/tasks/icon_classifier.yaml
    seedmill stats    -t config/tasks/icon_classifier.yaml
    seedmill tasks

Also runnable from a clone without installing: `python -m seedmill ...`
"""

from __future__ import annotations

import json
import os
from collections import Counter
from importlib import resources
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
    help="Config-driven synthetic data generator (local or hosted models).",
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
    if backend == "openai":
        from .openai_client import OpenAIClient

        # No default model: hosted model names date fast and a wrong guess
        # bills against the wrong one. The task YAML has to say.
        name = model or cfg.get("model")
        if not name:
            raise typer.BadParameter("The openai backend needs `model:` in the task YAML")
        try:
            return OpenAIClient(
                model=name,
                base_url=base_url or cfg.get("base_url"),
                api_key=cfg.get("api_key") or os.getenv("OPENAI_API_KEY", ""),
                default_temperature=temperature,
                strict=cfg.get("strict", True),
                params=cfg.get("params") or {},
            )
        except ValueError as e:  # missing key — a config error, not a crash
            raise typer.BadParameter(str(e)) from e
    raise typer.BadParameter(f"Unknown backend: {backend}")


@app.command()
def generate(
    task_path: str = typer.Option(..., "--task", "-t", help="Task YAML path"),
    output: str = typer.Option(None, "--output", "-o"),
    model: str = typer.Option(None, "--model", "-m", help="Override task model"),
    base_url: str = typer.Option(None, "--base-url"),
    backend: str = typer.Option(
        None,
        "--backend",
        help="Override backend: ollama | openai_compat (llama.cpp) | openai (hosted)",
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
    # Metered backends track tokens; local ones have no usage attribute.
    for k, v in getattr(client, "usage", {}).items():
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
    paths = sorted(Path(tasks_dir).glob("*.yaml"))
    if not paths:
        # Task YAMLs are not shipped in the wheel — they are the user's own
        # work, and a pip install starts with none. An empty table looks
        # like a bug, so say what to do instead.
        console.print(
            f"[yellow]No task configs in {tasks_dir}/[/yellow]\n"
            "Run [cyan]seedmill init[/cyan] to scaffold a runnable example, "
            "or point [cyan]--dir[/cyan] at your own tasks."
        )
        raise typer.Exit(0)

    table = Table(title="Tasks")
    table.add_column("Task", style="cyan")
    table.add_column("Mode")
    table.add_column("Description")
    for path in paths:
        task = load_task(path)
        table.add_row(
            str(path),
            "seeded" if task.seeds else "batch",
            task.description.strip()[:70],
        )
    console.print(table)


@app.command()
def init(
    name: str = typer.Argument("example", help="Name for the scaffolded task"),
    tasks_dir: str = typer.Option("config/tasks", "--dir"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file"),
) -> None:
    """Scaffold a runnable example task YAML to edit into your own."""
    dest = Path(tasks_dir) / f"{name}.yaml"
    if dest.exists() and not force:
        raise typer.BadParameter(f"{dest} already exists; pass --force to overwrite")

    template = (
        resources.files("seedmill.templates").joinpath("example_task.yaml").read_text()
    )
    # The template's own `name:` and its example command lines both carry
    # the placeholder name; rewrite them so the file is runnable as written.
    body = template.replace("config/tasks/example.yaml", str(dest)).replace(
        "\nname: example\n", f"\nname: {name}\n"
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(body)

    console.print(
        f"[green]Wrote {dest}[/green]\n"
        "It is self-contained — vocabulary and seeds are inline — so it runs "
        "as soon as a local model is up:\n"
        f"  [cyan]seedmill generate -t {dest} --limit 2 --per-seed 2[/cyan]"
    )


if __name__ == "__main__":
    app()
