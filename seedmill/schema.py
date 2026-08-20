"""
Schema derivation from a task's `fields:` block.

Two shapes come out of the same declaration:

  * `build_schema_json` — a compact, human-readable sketch used for the
    `$schema_json` prompt variable, so the model sees the record shape in
    the prompt.
  * `build_json_schema_for_response` — a real JSON Schema handed to the
    backend for grammar-constrained decoding, so the model *cannot* emit a
    value outside a field's enum.

Both handle nested objects and nullable fields. Prompts themselves live in
the task YAML and are rendered by `engine._render`, not here.
"""

from __future__ import annotations

import json
from typing import Any


def build_schema_json(fields: dict[str, Any]) -> str:
    """Build a compact JSON representation of the schema for the prompt."""
    schema: dict[str, Any] = {}

    for name, spec in fields.items():
        if spec.get("type") == "object" and "fields" in spec:
            nullable = " | null" if spec.get("nullable") else ""
            sub = {}
            for fname, fspec in spec["fields"].items():
                type_str = fspec.get("type", "str")
                if enum := fspec.get("enum"):
                    sub[fname] = f"Enum({', '.join(str(v) for v in enum)})"
                else:
                    sub[fname] = type_str
            schema[name] = f"object{nullable}: {json.dumps(sub)}"
        elif enum := spec.get("enum"):
            schema[name] = f"Enum({', '.join(str(v) for v in enum)})"
        else:
            nullable = " | null" if spec.get("nullable") else ""
            schema[name] = f"{spec.get('type', 'str')}{nullable}"

    return json.dumps(schema, indent=2)



def build_json_schema_for_response(
    fields: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a JSON Schema representation of the expected record structure.
    Useful for structured output / grammar-constrained generation.
    Handles nested objects and nullable types.
    """
    def _field_to_json_schema(spec: dict[str, Any]) -> dict[str, Any]:
        prop: dict[str, Any] = {}

        if spec.get("type") == "object" and "fields" in spec:
            # Nested object
            sub_props = {}
            sub_required = []
            for fname, fspec in spec["fields"].items():
                sub_props[fname] = _field_to_json_schema(fspec)
                if not fspec.get("nullable"):
                    sub_required.append(fname)
            prop = {
                "type": "object",
                "properties": sub_props,
                "required": sub_required,
            }
        elif spec.get("type") == "list" and "items" in spec:
            # Typed array items (e.g. vocab-enum strings)
            prop = {
                "type": "array",
                "items": _field_to_json_schema(spec["items"]),
            }
        else:
            type_map = {
                "str": "string", "string": "string",
                "int": "integer", "integer": "integer",
                "float": "number", "number": "number",
                "bool": "boolean", "boolean": "boolean",
                "list": "array", "dict": "object",
            }
            prop["type"] = type_map.get(spec.get("type", "str"), "string")

        if desc := spec.get("description"):
            prop["description"] = desc
        if enum := spec.get("enum"):
            prop["enum"] = enum

        # Nullable → anyOf with null
        if spec.get("nullable"):
            return {"anyOf": [prop, {"type": "null"}]}

        return prop

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, spec in fields.items():
        properties[name] = _field_to_json_schema(spec)
        if not spec.get("nullable") and spec.get("default") is None:
            required.append(name)

    return {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        },
        "required": ["records"],
    }
