"""
Pydantic model validation for generated records.

Dynamically builds Pydantic models from YAML schema definitions,
then validates LLM output against them. Supports nested objects,
nullable conditional fields, and enum constraints.

Validation is single-record; the retry loop lives in `engine._generate_batch`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ValidationError, create_model

# ── type mapping ────────────────────────────────────────────────────────

_TYPE_MAP: dict[str, type] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "list": list,
    "dict": dict,
}


def _resolve_type(spec: dict[str, Any], field_name: str = "") -> type:
    """
    Resolve a field spec to a Python type.

    If an enum is defined, creates a dynamic Enum type so Pydantic
    enforces the allowed values.
    """
    type_str = spec.get("type", "str")

    # Nested object → build a sub-model recursively
    if type_str == "object" and "fields" in spec:
        return _build_nested_model(field_name, spec["fields"])

    # Typed list items (e.g. list of vocab-enum strings)
    if type_str == "list" and "items" in spec:
        item_type = _resolve_type(spec["items"], field_name=f"{field_name}_item")
        return list[item_type]  # type: ignore[valid-type]

    base = _TYPE_MAP.get(type_str, str)

    if enum_vals := spec.get("enum"):
        enum_name = f"_{field_name}_enum" if field_name else f"_{type_str}_enum"
        enum_members = {str(v): v for v in enum_vals}
        return Enum(enum_name, enum_members)  # type: ignore[return-value]

    return base


def _build_nested_model(name: str, fields: dict[str, Any]) -> type[BaseModel]:
    """Recursively build a Pydantic model for a nested object."""
    field_definitions: dict[str, Any] = {}

    for fname, fspec in fields.items():
        ftype = _resolve_type(fspec, field_name=f"{name}_{fname}")
        is_nullable = fspec.get("nullable", False)

        if is_nullable:
            field_definitions[fname] = (Optional[ftype], None)
        elif fspec.get("default") is not None:
            field_definitions[fname] = (ftype, fspec["default"])
        else:
            field_definitions[fname] = (ftype, ...)

    model_name = name.replace(" ", "_").title().replace("_", "") + "Config"
    return create_model(model_name, **field_definitions)


def build_pydantic_model(schema: dict[str, Any]) -> type[BaseModel]:
    """
    Dynamically create a Pydantic model from a YAML schema definition.

    Supports:
    - Flat fields (str, int, float, bool, list, dict)
    - Enum-constrained fields
    - Nested object fields with their own sub-fields
    - Nullable fields (Optional types, default None)

    Args:
        schema: Parsed YAML with 'name' and 'fields' keys.

    Returns:
        A Pydantic BaseModel subclass with the defined fields.
    """
    field_definitions: dict[str, Any] = {}

    for name, spec in schema["fields"].items():
        field_type = _resolve_type(spec, field_name=name)
        is_nullable = spec.get("nullable", False)

        if is_nullable:
            # Optional field, defaults to None
            field_definitions[name] = (Optional[field_type], None)
        elif spec.get("default") is not None:
            field_definitions[name] = (field_type, spec["default"])
        else:
            # Required field
            field_definitions[name] = (field_type, ...)

    model_name = schema.get("name", "DynamicRecord").replace(" ", "_").title()
    return create_model(model_name, **field_definitions)


def _serialize_record(instance: BaseModel) -> dict[str, Any]:
    """
    Serialize a Pydantic model instance to a dict,
    converting enums to their .value and sub-models to dicts.
    """
    result = {}
    for field_name, field_value in instance:
        if field_value is None:
            result[field_name] = None
        elif isinstance(field_value, BaseModel):
            result[field_name] = _serialize_record(field_value)
        elif isinstance(field_value, Enum):
            result[field_name] = field_value.value
        elif isinstance(field_value, list):
            result[field_name] = [
                _serialize_record(item) if isinstance(item, BaseModel)
                else (item.value if isinstance(item, Enum) else item)
                for item in field_value
            ]
        else:
            result[field_name] = field_value
    return result


def validate_single(
    record: dict[str, Any],
    model: type[BaseModel],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a single record. Returns (data, None) or (None, error)."""
    try:
        instance = model.model_validate(record)
        return _serialize_record(instance), None
    except ValidationError as e:
        return None, str(e)
