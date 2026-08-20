"""
Hosted OpenAI backend: request shaping, usage accounting, error surfacing.

No network — a stub stands in for the SDK client, so what is asserted is
the payload we *send*. That is the part that differs from the llama.cpp
client and the part a hosted API rejects when it is wrong.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from seedmill.openai_client import OpenAIClient
from seedmill.schema import build_json_schema_for_response


class _StubCompletions:
    """Records the kwargs it was called with; replays canned responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _response(content, *, usage=(3, 5, 8), refusal=None):
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=usage[2]
        ),
    )


def _client(responses, **kwargs):
    client = OpenAIClient(model="test-model", api_key="test-key", **kwargs)
    stub = _StubCompletions(responses)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    return client, stub


SCHEMA = build_json_schema_for_response({"a": {"type": "str", "enum": ["x", "y"]}})
MESSAGES = [{"role": "user", "content": "go"}]


def test_missing_api_key_fails_at_construction():
    """Better a config error up front than one 401 per seed."""
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIClient(model="test-model", api_key="")


def test_schema_is_sent_as_a_strict_json_schema_response_format():
    """The llama.cpp shape ({"type": "json_object", "schema": ...}) is a
    400 here; the standard envelope is what has to go out."""
    client, stub = _client([_response('{"records": []}')])
    client.chat_json(MESSAGES, schema=SCHEMA)

    fmt = stub.calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    record = fmt["json_schema"]["schema"]["properties"]["records"]["items"]
    assert record["additionalProperties"] is False
    assert record["properties"]["a"]["enum"] == ["x", "y"]


def test_strict_false_falls_back_to_plain_json_mode():
    client, stub = _client([_response('{"records": []}')], strict=False)
    client.chat_json(MESSAGES, schema=SCHEMA)
    assert stub.calls[0]["response_format"] == {"type": "json_object"}


def test_no_llama_cpp_extensions_are_sent():
    """`think`/chat_template_kwargs is the other thing a hosted API 400s on."""
    client, stub = _client([_response('{"records": []}')])
    client.chat_json(MESSAGES, schema=SCHEMA)
    assert "extra_body" not in stub.calls[0]
    assert "max_tokens" not in stub.calls[0]


def test_params_are_forwarded_verbatim():
    client, stub = _client(
        [_response('{"records": []}')], params={"top_p": 0.9, "max_completion_tokens": 512}
    )
    client.chat_json(MESSAGES, schema=SCHEMA)
    assert stub.calls[0]["top_p"] == 0.9
    assert stub.calls[0]["max_completion_tokens"] == 512


def test_per_call_temperature_overrides_the_default():
    client, stub = _client([_response('{"records": []}')], default_temperature=0.7)
    client.chat_json(MESSAGES, temperature=0.2)
    assert stub.calls[0]["temperature"] == 0.2


def test_usage_accumulates_across_calls():
    """Cost visibility: a metered run reports what it spent."""
    client, _ = _client([_response("{}"), _response("{}")])
    client.chat_json(MESSAGES)
    client.chat_json(MESSAGES)
    assert client.usage == {
        "prompt_tokens": 6, "completion_tokens": 10, "total_tokens": 16, "requests": 2
    }


def test_refusal_is_raised_rather_than_parsed_as_json():
    client, _ = _client([_response(None, refusal="no")])
    with pytest.raises(ValueError, match="refused"):
        client.chat_json(MESSAGES)


def test_empty_content_raises():
    client, _ = _client([_response("   ")])
    with pytest.raises(ValueError, match="Empty response"):
        client.chat_json(MESSAGES)


def test_markdown_fences_are_stripped():
    client, _ = _client([_response('```json\n{"records": [1]}\n```')])
    assert client.chat_json(MESSAGES) == {"records": [1]}
