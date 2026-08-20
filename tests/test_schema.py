"""Schema derivation: the prompt sketch and the constrained-decode schema."""

from __future__ import annotations

import json

import pytest

from seedmill.schema import (
    build_json_schema_for_response,
    build_schema_json,
    to_strict_json_schema,
)


def _record_schema(fields):
    """Unwrap the {records: [...]} envelope down to the record properties."""
    return build_json_schema_for_response(fields)["properties"]["records"]["items"]


# ── build_json_schema_for_response ──────────────────────────────────────


def test_top_level_envelope_is_a_records_array():
    out = build_json_schema_for_response({"a": {"type": "str"}})
    assert out["type"] == "object"
    assert out["required"] == ["records"]
    assert out["properties"]["records"]["type"] == "array"
    assert out["properties"]["records"]["items"]["type"] == "object"


@pytest.mark.parametrize(
    "declared,expected",
    [
        ("str", "string"), ("string", "string"),
        ("int", "integer"), ("integer", "integer"),
        ("float", "number"), ("number", "number"),
        ("bool", "boolean"), ("boolean", "boolean"),
        ("list", "array"), ("dict", "object"),
        ("something_unrecognised", "string"),  # documented default
    ],
)
def test_type_mapping(declared, expected):
    items = _record_schema({"f": {"type": declared}})
    assert items["properties"]["f"]["type"] == expected


def test_enum_is_injected_into_the_property():
    """
    This is the assertion that a config vocabulary actually reaches the
    backend as a grammar constraint — the whole premise of the tool.
    """
    items = _record_schema({"icon": {"type": "str", "enum": ["a", "b"]}})
    assert items["properties"]["icon"]["enum"] == ["a", "b"]


def test_nullable_wraps_in_anyof_and_drops_from_required():
    items = _record_schema({"note": {"type": "str", "nullable": True}, "id": {"type": "str"}})
    assert items["properties"]["note"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert items["required"] == ["id"]


def test_field_with_a_default_is_not_required():
    items = _record_schema({"n": {"type": "int", "default": 5}})
    assert items["required"] == []


def test_nested_object_builds_sub_properties_and_sub_required():
    items = _record_schema(
        {
            "cfg": {
                "type": "object",
                "fields": {"n": {"type": "int"}, "note": {"type": "str", "nullable": True}},
            }
        }
    )
    cfg = items["properties"]["cfg"]
    assert cfg["type"] == "object"
    assert set(cfg["properties"]) == {"n", "note"}
    assert cfg["required"] == ["n"]  # the nullable subfield is excluded


def test_nullable_nested_object_is_wrapped_too():
    items = _record_schema(
        {"cfg": {"type": "object", "nullable": True, "fields": {"n": {"type": "int"}}}}
    )
    assert "anyOf" in items["properties"]["cfg"]


def test_list_items_carry_their_own_type_and_enum():
    items = _record_schema({"tags": {"type": "list", "items": {"type": "str", "enum": ["a", "b"]}}})
    assert items["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string", "enum": ["a", "b"]},
    }


def test_description_passes_through():
    items = _record_schema({"f": {"type": "str", "description": "hello"}})
    assert items["properties"]["f"]["description"] == "hello"


# ── build_schema_json (the prompt-facing sketch) ────────────────────────


def test_schema_json_is_valid_json():
    assert isinstance(json.loads(build_schema_json({"a": {"type": "str"}})), dict)


def test_schema_json_renders_enums_and_nullability_readably():
    out = json.loads(
        build_schema_json(
            {
                "label": {"type": "str", "enum": ["x", "y"]},
                "note": {"type": "str", "nullable": True},
                "n": {"type": "int"},
            }
        )
    )
    assert out["label"] == "Enum(x, y)"
    assert out["note"] == "str | null"
    assert out["n"] == "int"


def test_schema_json_renders_nested_objects():
    out = json.loads(
        build_schema_json(
            {
                "cfg": {
                    "type": "object",
                    "nullable": True,
                    "fields": {"kind": {"type": "str", "enum": ["a"]}, "n": {"type": "int"}},
                }
            }
        )
    )
    assert out["cfg"].startswith("object | null: ")
    inner = json.loads(out["cfg"].split(": ", 1)[1])
    assert inner == {"kind": "Enum(a)", "n": "int"}


# ── to_strict_json_schema ───────────────────────────────────────────────


def _strict_record(fields):
    """Unwrap the strict-rewritten schema down to the record properties."""
    strict = to_strict_json_schema(build_json_schema_for_response(fields))
    return strict["properties"]["records"]["items"]


def test_strict_marks_every_property_required():
    """The rule hosted APIs enforce: no property may be omitted."""
    items = _strict_record(
        {"id": {"type": "str"}, "note": {"type": "str", "nullable": True}}
    )
    assert set(items["required"]) == {"id", "note"}


def test_strict_forbids_additional_properties_at_every_object():
    items = _strict_record(
        {"id": {"type": "str"}, "obj": {"type": "object", "fields": {"x": {"type": "str"}}}}
    )
    assert items["additionalProperties"] is False
    assert items["properties"]["obj"]["additionalProperties"] is False


def test_strict_makes_a_defaulted_field_nullable():
    """Optionality has to move from `required` into the type, or the
    model has no legal way to leave a defaulted field out."""
    items = _strict_record({"n": {"type": "int", "default": 5}})
    assert items["properties"]["n"] == {
        "anyOf": [{"type": "integer"}, {"type": "null"}]
    }


def test_strict_does_not_double_wrap_an_already_nullable_field():
    items = _strict_record({"note": {"type": "str", "nullable": True}})
    assert items["properties"]["note"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }


def test_strict_preserves_enums_through_the_rewrite():
    """The vocabulary constraint must survive, or the backend stops
    enforcing the thing the tool exists to enforce."""
    items = _strict_record({"icon": {"type": "str", "enum": ["a", "b"]}})
    assert items["properties"]["icon"]["enum"] == ["a", "b"]


def test_strict_recurses_into_array_items_and_nested_objects():
    items = _strict_record(
        {
            "tags": {"type": "list", "items": {"type": "str", "enum": ["a"]}},
            "obj": {
                "type": "object",
                "fields": {"x": {"type": "str"}, "y": {"type": "str", "nullable": True}},
            },
        }
    )
    assert items["properties"]["tags"]["items"]["enum"] == ["a"]
    assert set(items["properties"]["obj"]["required"]) == {"x", "y"}


def test_strict_leaves_the_original_schema_untouched():
    """The same schema object is reused across a run; rewriting in place
    would leak strict-mode shape into the local backends."""
    original = build_json_schema_for_response({"n": {"type": "int", "default": 5}})
    before = json.dumps(original, sort_keys=True)
    to_strict_json_schema(original)
    assert json.dumps(original, sort_keys=True) == before
