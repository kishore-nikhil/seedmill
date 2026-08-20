"""
OpenAI-compatible client for local llama.cpp server.

Wraps the openai library pointed at localhost:8080 (or configurable).
Supports both regular chat completions and tool-calling mode.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion


@dataclass
class LlamaClient:
    """Thin wrapper around OpenAI client targeting a local llama-server."""

    base_url: str = field(
        default_factory=lambda: os.getenv("LLAMA_BASE_URL", "http://localhost:8080/v1")
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("LLAMA_API_KEY", "not-needed")
    )
    model: str = field(
        default_factory=lambda: os.getenv("LLAMA_MODEL", "local-model")
    )
    default_temperature: float = 0.7
    default_max_tokens: int = 4096
    think: bool = True
    """Reasoning models (Qwen3, gemma4) think by default. With think=False
    the request carries chat_template_kwargs.enable_thinking=false, which
    llama.cpp passes to the Jinja chat template — for a generation run the
    reasoning is pure latency (measured ~30% slower per batch) and its text
    lands in `reasoning_content`, never in the parsed record."""

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    # ── plain chat completion ───────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> ChatCompletion:
        """Send a chat completion request, optionally requesting JSON mode."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if not self.think:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return self._client.chat.completions.create(**kwargs)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Chat completion with JSON mode; returns parsed dict.

        With `schema`, llama.cpp constrains decoding to that JSON schema
        (grammar-based), matching the Ollama backend's behavior.
        """
        response_format: dict[str, Any] = {"type": "json_object"}
        if schema is not None:
            response_format["schema"] = schema
        resp = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("Empty response from server")

        # llama.cpp sometimes wraps JSON in markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        return json.loads(text)

    # ── tool-calling completion ─────────────────────────────────────────

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict = "auto",
    ) -> ChatCompletion:
        """
        Send a chat completion with tool definitions.

        llama.cpp supports the OpenAI tool-calling format when the model
        has been loaded with a compatible chat template (e.g. Hermes, Qwen).
        """
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
        )

    def extract_tool_calls(
        self, completion: ChatCompletion
    ) -> list[dict[str, Any]]:
        """
        Parse tool calls from a completion response.

        Returns list of dicts: [{"name": ..., "arguments": {...}, "id": ...}]
        """
        message = completion.choices[0].message
        if not message.tool_calls:
            return []

        results = []
        for tc in message.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            results.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                }
            )
        return results
