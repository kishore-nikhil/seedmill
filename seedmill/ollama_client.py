"""
Native Ollama API client.

Used instead of the OpenAI-compat endpoint for two capabilities it lacks:
- `think: false` — reasoning models (gemma4) otherwise put the answer in a
  `reasoning` field and return empty `content` after burning the budget.
- `format: <json schema>` — grammar-constrained decoding, so enum fields
  (like icon vocabularies) are guaranteed valid at generation time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class OllamaClient:
    """Minimal client for Ollama's native /api/chat endpoint."""

    base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "gemma4:27b"))
    default_temperature: float = 0.7
    think: bool = False
    timeout: float = 600.0

    def __post_init__(self) -> None:
        self._http = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Chat completion returning parsed JSON.

        If `schema` is given, Ollama constrains decoding to that JSON schema;
        otherwise plain JSON mode ("format": "json") is used.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "format": schema if schema is not None else "json",
            "options": {
                "temperature": (
                    temperature if temperature is not None else self.default_temperature
                ),
            },
        }
        resp = self._http.post("/api/chat", json=payload)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()
        if not content:
            raise ValueError("Empty response from Ollama")
        return json.loads(content)
