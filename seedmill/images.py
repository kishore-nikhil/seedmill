"""
Image inputs for vision models.

A task attaches images to its user message with `prompts.images`. The
engine reads each file into an `Image` and puts the list on the message
under an `images` key — Ollama's own message shape, minus the encoding.
Each client then writes that into its wire format:

- Ollama's native `/api/chat` wants bare base64 strings in `images`.
- The OpenAI protocol (hosted, and llama.cpp started with `--mmproj`)
  wants the message content as parts, each image as a data URL.

A message without `images` passes through both untouched, so text-only
tasks send exactly what they always did.

Nothing is resized or re-encoded here. That would put Pillow in the core,
and the right size depends on the model; prepare the files first.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


@dataclass(frozen=True)
class Image:
    path: Path
    mime: str
    data: str = field(repr=False)
    """Base64 of the file's bytes. Kept out of the repr so an error message
    that prints a message never dumps a megabyte of base64 to the console."""

    @property
    def data_url(self) -> str:
        return f"data:{self.mime};base64,{self.data}"


def load_image(path: str | Path) -> Image:
    path = Path(path)
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise ValueError(
            f"{path.name}: can't send '{path.suffix}' files as images. "
            f"Use one of: {', '.join(sorted(MIME_TYPES))}"
        )
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return Image(path=path, mime=mime, data=data)


def ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Images as bare base64 strings on the message."""
    return [
        {**m, "images": [img.data for img in m["images"]]} if m.get("images") else m
        for m in messages
    ]


def openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Images as `image_url` content parts, ahead of the text.

    Images go first because that is where Ollama's chat templates put them
    and what the Qwen-VL model cards recommend, so the two protocols show
    the model the same thing.
    """
    out = []
    for m in messages:
        if not m.get("images"):
            out.append(m)
            continue
        parts: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": img.data_url}} for img in m["images"]
        ]
        parts.append({"type": "text", "text": m["content"]})
        out.append({"role": m["role"], "content": parts})
    return out
