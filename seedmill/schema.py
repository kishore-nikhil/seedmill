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


def to_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Rewrite a schema from `build_json_schema_for_response` into the subset
    hosted structured-output APIs accept under `strict: true`.

    Those APIs require, at every object node, that `additionalProperties`
    is false and that *every* declared property appears in `required` —
    optionality is expressed by admitting null, not by omission. Our
    builder does the opposite: nullable and defaulted fields are simply
    left out of `required`.

    So for each object: union every non-required property's type with
    null, then mark all of them required. A field already wrapped in
    `anyOf: [..., {"type": "null"}]` (the `nullable: true` case) is left
    alone; only defaulted-but-not-nullable fields gain a null branch.

    The rewrite is value-preserving for a local backend too — a record
    that satisfies the strict schema satisfies the original.
    """
    def _walk(node: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(node, dict):
            return node

        out = dict(node)

        if "anyOf" in out:
            out["anyOf"] = [_walk(b) for b in out["anyOf"]]
            return out

        if out.get("type") == "array" and "items" in out:
            out["items"] = _walk(out["items"])
            return out

        if out.get("type") == "object" and "properties" in out:
            props = {name: _walk(spec) for name, spec in out["properties"].items()}
            required = set(out.get("required", []))
            for name, spec in props.items():
                if name not in required:
                    props[name] = _nullable(spec)
            out["properties"] = props
            out["required"] = list(props)
            out["additionalProperties"] = False

        return out

    def _nullable(spec: dict[str, Any]) -> dict[str, Any]:
        """Admit null, without double-wrapping an already-nullable branch."""
        if "anyOf" in spec:
            if any(b.get("type") == "null" for b in spec["anyOf"]):
                return spec
            return {"anyOf": [*spec["anyOf"], {"type": "null"}]}
        return {"anyOf": [spec, {"type": "null"}]}

    return _walk(schema)
