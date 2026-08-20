"""
`seedmill init` — the scaffold a pip install starts from.

No model is contacted here either: init only copies a packaged template,
and the assertion that matters is that the copy is a *loadable* task, so
a fresh install's first command cannot land the user on a broken file.
"""

from __future__ import annotations

from importlib import resources

from typer.testing import CliRunner

from seedmill.cli import app
from seedmill.task_config import load_task

runner = CliRunner()


def test_the_template_ships_inside_the_package():
    """A pip install has no config/ — if this is not in the wheel, init
    has nothing to copy and the failure only shows up post-release."""
    template = resources.files("seedmill.templates").joinpath("example_task.yaml")
    assert template.is_file()


def test_init_writes_a_task_that_actually_loads(tmp_path):
    """Loading exercises vocab resolution, seeds and enum injection —
    the same path `seedmill generate` takes before it reaches a model."""
    result = runner.invoke(app, ["init", "--dir", str(tmp_path)])
    assert result.exit_code == 0

    task = load_task(tmp_path / "example.yaml")
    assert task.name == "example"
    assert task.seeds is not None
    # The vocabulary reached the field as a decode-time constraint.
    assert "bug_report" in task.fields["intent"]["enum"]


def test_init_renames_the_task_to_the_given_name(tmp_path):
    """The written file is runnable as-is, so its `name:` and the example
    commands in its header have to match where it landed."""
    result = runner.invoke(app, ["init", "support", "--dir", str(tmp_path)])
    assert result.exit_code == 0

    dest = tmp_path / "support.yaml"
    assert load_task(dest).name == "support"
    assert str(dest) in dest.read_text()
    assert "config/tasks/example.yaml" not in dest.read_text()


def test_init_refuses_to_clobber_an_existing_task(tmp_path):
    runner.invoke(app, ["init", "--dir", str(tmp_path)])
    (tmp_path / "example.yaml").write_text("name: mine\n")

    result = runner.invoke(app, ["init", "--dir", str(tmp_path)])
    assert result.exit_code != 0
    assert (tmp_path / "example.yaml").read_text() == "name: mine\n"


def test_force_overwrites(tmp_path):
    (tmp_path / "example.yaml").write_text("name: mine\n")
    result = runner.invoke(app, ["init", "--dir", str(tmp_path), "--force"])
    assert result.exit_code == 0
    assert load_task(tmp_path / "example.yaml").name == "example"


def test_init_creates_a_missing_tasks_directory(tmp_path):
    result = runner.invoke(app, ["init", "--dir", str(tmp_path / "nested" / "tasks")])
    assert result.exit_code == 0
    assert (tmp_path / "nested" / "tasks" / "example.yaml").is_file()


def test_tasks_points_at_init_when_it_finds_nothing(tmp_path):
    """An empty table reads as a bug on a fresh install."""
    result = runner.invoke(app, ["tasks", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "seedmill init" in result.output
