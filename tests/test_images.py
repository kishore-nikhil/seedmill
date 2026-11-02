"""
Image inputs: reading files, and the wire shape each backend sends.

No network. Each client's transport is swapped for a stub that records
what it was handed, so what is asserted is the request body, which is the
part that differs between Ollama and the OpenAI protocol.
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from seedmill.client import LlamaClient
from seedmill.images import load_image, ollama_messages, openai_messages
from seedmill.ollama_client import OllamaClient
from seedmill.openai_client import OpenAIClient

PNG = b"\x89PNG\r\n\x1a\nnot really a png"


@pytest.fixture
def png(tmp_path: Path) -> Path:
    p = tmp_path / "zz_page.png"
    p.write_bytes(PNG)
    return p


def _messages(png: Path) -> list[dict]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "read this", "images": [load_image(png)]},
    ]


# ── load_image ──────────────────────────────────────────────────────────


def test_load_image_reads_bytes_as_base64(png: Path):
    img = load_image(png)
    assert img.mime == "image/png"
    assert base64.b64decode(img.data) == PNG
    assert img.data_url == "data:image/png;base64," + img.data


def test_load_image_maps_jpeg_suffixes_case_insensitively(tmp_path: Path):
    p = tmp_path / "zz_scan.JPEG"
    p.write_bytes(b"\xff\xd8")
    assert load_image(p).mime == "image/jpeg"


def test_load_image_refuses_formats_apis_dont_take(tmp_path: Path):
    p = tmp_path / "zz_scan.tiff"
    p.write_bytes(b"II*")
    with pytest.raises(ValueError, match="can't send '.tiff'"):
        load_image(p)


def test_image_repr_leaves_out_the_base64(png: Path):
    """An error that prints a message must not print the whole image."""
    img = load_image(png)
    assert img.data not in repr(img)


# ── message conversion ──────────────────────────────────────────────────


def test_text_only_messages_pass_through_both_converters_unchanged():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert ollama_messages(msgs) == msgs
    assert openai_messages(msgs) == msgs


def test_ollama_shape_is_base64_strings_on_the_message(png: Path):
    sys_msg, user = ollama_messages(_messages(png))
    assert sys_msg == {"role": "system", "content": "sys"}
    assert user == {
        "role": "user",
        "content": "read this",
        "images": [base64.b64encode(PNG).decode("ascii")],
    }


def test_openai_shape_is_content_parts_images_first(png: Path):
    _, user = openai_messages(_messages(png))
    assert user["role"] == "user"
    assert "images" not in user
    image, text = user["content"]
    assert image == {"type": "image_url", "image_url": {"url": load_image(png).data_url}}
    assert text == {"type": "text", "text": "read this"}


def test_conversion_does_not_mutate_the_engines_messages(png: Path):
    """The engine reuses the same messages for every retry of a batch."""
    msgs = _messages(png)
    ollama_messages(msgs)
    openai_messages(msgs)
    assert msgs[1]["content"] == "read this"
    assert msgs[1]["images"][0].path == png


# ── what each client sends ──────────────────────────────────────────────


def test_ollama_client_sends_images_natively(png: Path):
    sent = {}

    def post(path, json):
        sent.update(path=path, body=json)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": '{"records": []}'}},
        )

    client = OllamaClient(model="qwen3-vl:8b")
    client._http = SimpleNamespace(post=post)
    client.chat_json(_messages(png), schema={"type": "object"})

    assert sent["path"] == "/api/chat"
    assert sent["body"]["messages"][1]["images"] == [base64.b64encode(PNG).decode("ascii")]


class _StubCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content='{"records": []}', refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _stub(client) -> _StubCompletions:
    stub = _StubCompletions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    return stub


@pytest.mark.parametrize(
    "make",
    [
        lambda: LlamaClient(model="local"),
        lambda: OpenAIClient(model="test-model", api_key="test-key"),
    ],
    ids=["openai_compat", "openai"],
)
def test_openai_protocol_clients_send_image_parts(make, png: Path):
    client = make()
    stub = _stub(client)
    client.chat_json(_messages(png), schema={"type": "object", "properties": {}})

    user = stub.calls[0]["messages"][1]
    assert [part["type"] for part in user["content"]] == ["image_url", "text"]
    assert user["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
