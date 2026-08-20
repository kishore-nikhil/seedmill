"""
Declarative cross-field consistency rules.

Replaces per-task hardcoded validators (e.g. a hardcoded per-entity
consistency check). Each rule is a small dict in the task YAML;
rules run after pydantic validation and may auto-fix instead of rejecting.

Supported rules:

  - unique_items: {list: <field>}
      Deduplicate a list field in place (always an auto-fix).

  - not_in: {value: <field>, list: <field>}
      Reject if record[value] appears in record[list].

  - contains: {list: <field>, value: <field>, autofix: bool}
      record[list] must contain record[value]. With autofix, prepend it.

  - disjoint: {a: <field>, b: <field>, autofix_remove_from: "a"|"b"|null}
      The two list fields must not overlap. With autofix, drop the
      overlapping items from the named side.

  - exclusive_switch: {switch: <field>, cases: {<value>: <field>, ...}}
      Discriminated-union check: when record[switch] == value, that case
      field must be non-null and every other case field null. A switch
      value with no case (e.g. "none") requires all case fields null.
"""

from __future__ import annotations

from typing import Any


def apply_rules(
    record: dict[str, Any],
    rules: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """
    Run all rules against a record.

    Returns (record, issues). record is None if any non-fixable rule
    failed; otherwise it's the (possibly auto-fixed) record.
    """
    issues: list[str] = []

    for rule in rules:
        kind, spec = next(iter(rule.items()))

        if kind == "unique_items":
            f = spec["list"]
            if isinstance(record.get(f), list):
                record[f] = list(dict.fromkeys(record[f]))

        elif kind == "not_in":
            value = record.get(spec["value"])
            listed = record.get(spec["list"]) or []
            if value in listed:
                issues.append(f"{spec['value']}='{value}' appears in {spec['list']}")

        elif kind == "contains":
            f_list, f_value = spec["list"], spec["value"]
            value = record.get(f_value)
            listed = record.get(f_list) or []
            if value not in listed:
                if spec.get("autofix"):
                    record[f_list] = [value] + listed
                else:
                    issues.append(f"{f_list} does not contain {f_value}='{value}'")

        elif kind == "disjoint":
            fa, fb = spec["a"], spec["b"]
            overlap = set(record.get(fa) or []) & set(record.get(fb) or [])
            if overlap:
                side = spec.get("autofix_remove_from")
                if side in ("a", "b"):
                    f = fa if side == "a" else fb
                    record[f] = [v for v in record[f] if v not in overlap]
                else:
                    issues.append(f"{fa} and {fb} overlap: {sorted(overlap)}")

        elif kind == "exclusive_switch":
            switch_val = record.get(spec["switch"])
            cases: dict[str, str] = spec["cases"]
            expected = cases.get(switch_val)
            if expected is not None and record.get(expected) is None:
                issues.append(
                    f"{spec['switch']}='{switch_val}' but {expected} is null"
                )
            for case_val, case_field in cases.items():
                if case_field != expected and record.get(case_field) is not None:
                    issues.append(
                        f"{spec['switch']}='{switch_val}' but {case_field} is not null"
                    )

        else:
            issues.append(f"unknown rule type: {kind}")

    return (None, issues) if issues else (record, issues)


def check_field_bounds(
    record: dict[str, Any],
    fields: dict[str, Any],
) -> list[str]:
    """
    Post-validation length/size checks from field specs
    (min_length/max_length for strings, min_items/max_items for lists).
    Kept out of the constrained-decoding schema for backend compatibility.
    """
    issues = []
    for fname, spec in fields.items():
        value = record.get(fname)
        if isinstance(value, str):
            n = len(value.strip())
            if "min_length" in spec and n < spec["min_length"]:
                issues.append(f"{fname} too short ({n} chars)")
            if "max_length" in spec and n > spec["max_length"]:
                issues.append(f"{fname} too long ({n} chars)")
        elif isinstance(value, list):
            if "min_items" in spec and len(value) < spec["min_items"]:
                issues.append(f"{fname} has {len(value)} items (min {spec['min_items']})")
            if "max_items" in spec and len(value) > spec["max_items"]:
                issues.append(f"{fname} has {len(value)} items (max {spec['max_items']})")
    return issues
