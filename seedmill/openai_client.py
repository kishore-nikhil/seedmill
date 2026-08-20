"""
Hosted OpenAI-compatible client (OpenAI, Gemini's compat endpoint, …).

Distinct from `client.LlamaClient` — which also speaks the OpenAI protocol —
because a hosted API rejects the two llama.cpp extensions that client relies
on: a `schema` key inside a `json_object` response format, and the
`chat_template_kwargs` body param behind `think: false`. Here the schema goes
through the standard `json_schema` response format instead (see
`schema.to_strict_json_schema` for the shape it has to be in), and nothing
non-standard is sent unless the task asks for it via `params:`.

Two things matter here that do not when the server is on localhost:

- **Rate limits.** `max_retries` is handed to the SDK, which backs off
  exponentially and honours `Retry-After` on 429s. The engine's own retry
  loop sleeps a flat second and is meant for bad *records*, not throttling.
- **Cost.** Every response's token usage accumulates in `self.usage`, so a
  run reports what it spent instead of discovering it on an invoice.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .schema import to_strict_json_schema


@dataclass
class OpenAIClient:
    """Client for hosted OpenAI-compatible chat completions APIs."""

    model: str
    base_url: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None
    )
    """None → the SDK's own default (api.openai.com). Set it for any other
    provider, e.g. Gemini's https://generativelanguage.googleapis.com/v1beta/openai/"""

    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    default_temperature: float = 0.7
    strict: bool = True
    """Constrain decoding to the record schema. Turn off (`strict: false` in
    the task's model block) for a provider whose structured-output support
    chokes on the schema; output then falls back to plain JSON mode and the
    pydantic/rules layer becomes the only thing enforcing enums."""

    params: dict[str, Any] = field(default_factory=dict)
    """Extra keyword arguments forwarded verbatim to the completions call —
    `max_completion_tokens`, `reasoning_effort`, `top_p`, `seed`. Kept as a
    passthrough because these differ per provider and per model generation;
    pinning them here would date the client."""

    max_retries: int = 5
    timeout: float = 600.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError(
                "No API key. Set OPENAI_API_KEY (or api_key in the task's "
                "model block) for the openai backend."
            )
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
        self._usage_lock = threading.Lock()
        self.usage: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0
        }

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Chat completion returning parsed JSON.

        With `schema` and `strict`, decoding is constrained to the record
        schema server-side; otherwise plain JSON mode is requested and the
        schema only reaches the model through the prompt.
        """
        if schema is not None and self.strict:
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "records",
                    "strict": True,
                    "schema": to_strict_json_schema(schema),
                },
            }
        else:
            response_format = {"type": "json_object"}

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=(
                temperature if temperature is not None else self.default_temperature
            ),
            response_format=response_format,
            **self.params,
        )
        self._record_usage(resp)

        message = resp.choices[0].message
        if refusal := getattr(message, "refusal", None):
            raise ValueError(f"Model refused the request: {refusal}")

        text = (message.content or "").strip()
        if not text:
            raise ValueError("Empty response from server")
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        return json.loads(text)

    def _record_usage(self, resp: Any) -> None:
        """Accumulate token usage; `workers > 1` shares one client."""
        usage = getattr(resp, "usage", None)
        with self._usage_lock:
            self.usage["requests"] += 1
            if usage is None:
                return
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self.usage[key] += getattr(usage, key, 0) or 0
