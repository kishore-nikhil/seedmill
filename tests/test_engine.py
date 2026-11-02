"""
The engine with file seeds and images, against a stub model.

The stub records every message list it is sent and answers with a fixed
record, so these check what reaches the model and what lands in the
output file, not what a model would say.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from seedmill.engine import TaskEngine
from seedmill.task_config import load_task


class StubClient:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def chat_json(self, messages, *, temperature=None, schema=None):
        self.calls.append(messages)
        return {"records": [dict(r) for r in self.records]}


def _scan_task(tmp_path: Path, **overrides) -> Path:
    for name in ("nemo/1906-08-19.png", "nemo/1906-08-26.png", "krazy/1913-10-28.png"):
        path = tmp_path / "zz_scans" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG " + name.encode())
    task = {
        "name": "pages",
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
                                "fields": {"text": {"type": "str", "max_length": 20}},
                            },
                        }
                    },
                },
            }
        },
        "seeds": {"files": "zz_scans/**/*.png", "value_field": "page", "group_field": "strip"},
        "prompts": {"system": "sys", "user": "Read $seed.", "images": ["$seed_path"]},
        **overrides,
    }
    p = tmp_path / "zz_pages.yaml"
    p.write_text(yaml.safe_dump(task, sort_keys=False))
    return p


RECORD = {"panels": [{"balloons": [{"text": "Wake up, Nemo!"}]}]}


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_each_file_seed_sends_its_own_image_and_lands_one_record(tmp_path: Path):
    task = load_task(_scan_task(tmp_path))
    client = StubClient([RECORD])
    out = tmp_path / "out.jsonl"
    TaskEngine(task, client, out).run()

    sent = [(msgs[1]["content"], msgs[1]["images"][0].path) for msgs in client.calls]
    assert sent == [
        ("Read krazy/1913-10-28.png.", (tmp_path / "zz_scans/krazy/1913-10-28.png").resolve()),
        ("Read nemo/1906-08-19.png.", (tmp_path / "zz_scans/nemo/1906-08-19.png").resolve()),
        ("Read nemo/1906-08-26.png.", (tmp_path / "zz_scans/nemo/1906-08-26.png").resolve()),
    ]

    # Identical drafts for three pages are three records: dedup is off for
    # file seeds, since describing two files the same way isn't repetition.
    rows = _lines(out)
    assert [(r["page"], r["strip"]) for r in rows] == [
        ("krazy/1913-10-28.png", "krazy"),
        ("nemo/1906-08-19.png", "nemo"),
        ("nemo/1906-08-26.png", "nemo"),
    ]
    assert all(r["panels"] == RECORD["panels"] for r in rows)


def test_a_rerun_skips_pages_already_done(tmp_path: Path):
    task = load_task(_scan_task(tmp_path))
    out = tmp_path / "out.jsonl"
    TaskEngine(task, StubClient([RECORD]), out).run(limit=1)

    client = StubClient([RECORD])
    TaskEngine(load_task(_scan_task(tmp_path)), client, out).run()
    assert len(client.calls) == 2
    assert len(_lines(out)) == 3


def test_retry_resends_the_image_with_a_nested_bound_error(tmp_path: Path):
    task = load_task(_scan_task(tmp_path))
    too_long = {"panels": [{"balloons": [{"text": "x" * 30}]}]}

    class OnceWrong(StubClient):
        def chat_json(self, messages, **kwargs):
            self.calls.append(messages)
            return {"records": [too_long if len(self.calls) == 1 else RECORD]}

    client = OnceWrong([])
    TaskEngine(task, client, tmp_path / "out.jsonl").run(limit=1)

    retry = client.calls[1]
    assert retry[1]["images"] == client.calls[0][1]["images"]
    assert "panels[0].balloons[0].text too long (30 chars)" in retry[2]["content"]


def test_dedup_can_be_turned_back_on_for_file_seeds(tmp_path: Path):
    generation = {"dedup": {"fields": ["panels"]}, "max_retries": 0}
    task = load_task(_scan_task(tmp_path, generation=generation))
    out = tmp_path / "out.jsonl"
    TaskEngine(task, StubClient([RECORD]), out).run()
    assert len(_lines(out)) == 1


def test_dedup_false_turns_it_off_in_batch_mode(tmp_path: Path):
    p = tmp_path / "zz_batch.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "b",
                "fields": {"a": {"type": "str"}},
                "prompts": {"system": "s", "user": "u $n"},
                "generation": {"count": 3, "batch_size": 1, "dedup": False},
            }
        )
    )
    out = tmp_path / "out.jsonl"
    TaskEngine(load_task(p), StubClient([{"a": "same"}]), out).run()
    assert [r["a"] for r in _lines(out)] == ["same", "same", "same"]


def test_a_task_without_images_sends_plain_text_messages(tmp_path: Path):
    p = tmp_path / "zz_plain.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "p",
                "fields": {"a": {"type": "str"}},
                "prompts": {"system": "s", "user": "u $n"},
                "generation": {"count": 1},
            }
        )
    )
    client = StubClient([{"a": "x"}])
    TaskEngine(load_task(p), client, tmp_path / "out.jsonl").run()
    assert client.calls[0] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u 1"},
    ]
