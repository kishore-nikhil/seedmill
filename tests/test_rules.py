"""Declarative cross-field rules and post-validation field bounds."""

from __future__ import annotations

import pytest

from seedmill.rules import apply_rules, check_field_bounds

# ── unique_items ────────────────────────────────────────────────────────


def test_unique_items_dedupes_preserving_first_seen_order():
    rule = [{"unique_items": {"list": "tags"}}]
    rec, issues = apply_rules({"tags": ["b", "a", "b", "c", "a"]}, rule)
    assert rec["tags"] == ["b", "a", "c"]
    assert issues == []


@pytest.mark.parametrize("value", ["not a list", None, 42])
def test_unique_items_ignores_non_lists(value):
    """Never rejects — it is always an auto-fix, and a no-op off-type."""
    rec, issues = apply_rules({"tags": value}, [{"unique_items": {"list": "tags"}}])
    assert rec is not None and issues == []
    assert rec["tags"] == value


def test_unique_items_ignores_missing_field():
    rec, issues = apply_rules({}, [{"unique_items": {"list": "tags"}}])
    assert rec == {} and issues == []


# ── not_in ──────────────────────────────────────────────────────────────


def test_not_in_rejects_when_value_is_in_list():
    rule = [{"not_in": {"value": "icon", "list": "tags"}}]
    rec, issues = apply_rules({"icon": "gym", "tags": ["gym", "run"]}, rule)
    assert rec is None
    assert "icon" in issues[0] and "tags" in issues[0]


def test_not_in_passes_when_absent():
    rule = [{"not_in": {"value": "icon", "list": "tags"}}]
    rec, issues = apply_rules({"icon": "gym", "tags": ["run"]}, rule)
    assert rec is not None and issues == []


def test_not_in_has_no_autofix_path():
    """
    Unlike `contains` and `disjoint`, `not_in` rejects unconditionally —
    an `autofix` key in the spec is ignored, not honoured.
    """
    rule = [{"not_in": {"value": "icon", "list": "tags", "autofix": True}}]
    rec, issues = apply_rules({"icon": "gym", "tags": ["gym"]}, rule)
    assert rec is None and len(issues) == 1


def test_not_in_treats_null_list_as_empty():
    rule = [{"not_in": {"value": "icon", "list": "tags"}}]
    assert apply_rules({"icon": "gym", "tags": None}, rule)[0] is not None
    assert apply_rules({"icon": "gym"}, rule)[0] is not None


# ── contains ────────────────────────────────────────────────────────────


def test_contains_autofix_prepends_the_value():
    rule = [{"contains": {"list": "tags", "value": "icon", "autofix": True}}]
    rec, issues = apply_rules({"icon": "gym", "tags": ["run"]}, rule)
    assert rec["tags"] == ["gym", "run"]
    assert issues == []


def test_contains_without_autofix_rejects():
    rule = [{"contains": {"list": "tags", "value": "icon"}}]
    rec, issues = apply_rules({"icon": "gym", "tags": ["run"]}, rule)
    assert rec is None and len(issues) == 1


def test_contains_no_mutation_when_already_present():
    rule = [{"contains": {"list": "tags", "value": "icon", "autofix": True}}]
    rec, _ = apply_rules({"icon": "gym", "tags": ["gym", "run"]}, rule)
    assert rec["tags"] == ["gym", "run"]


def test_contains_autofix_prepends_none_when_value_is_null():
    """
    Current behaviour, pinned deliberately: a null value is prepended as
    None rather than skipped. If a guard is ever added, this test should
    change as a conscious decision, not silently.
    """
    rule = [{"contains": {"list": "tags", "value": "icon", "autofix": True}}]
    rec, _ = apply_rules({"icon": None, "tags": ["run"]}, rule)
    assert rec["tags"] == [None, "run"]


# ── disjoint ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "side,expect_a,expect_b",
    [("a", ["x"], ["y", "z"]), ("b", ["x", "y"], ["z"])],
)
def test_disjoint_autofix_removes_from_named_side(side, expect_a, expect_b):
    rule = [{"disjoint": {"a": "fa", "b": "fb", "autofix_remove_from": side}}]
    rec, issues = apply_rules({"fa": ["x", "y"], "fb": ["y", "z"]}, rule)
    assert rec["fa"] == expect_a
    assert rec["fb"] == expect_b
    assert issues == []


def test_disjoint_rejects_without_autofix_and_names_the_overlap():
    rule = [{"disjoint": {"a": "fa", "b": "fb"}}]
    rec, issues = apply_rules({"fa": ["x", "y"], "fb": ["y", "z"]}, rule)
    assert rec is None
    assert "['y']" in issues[0]


def test_disjoint_no_overlap_is_a_no_op():
    rule = [{"disjoint": {"a": "fa", "b": "fb", "autofix_remove_from": "a"}}]
    rec, issues = apply_rules({"fa": ["x"], "fb": ["y"]}, rule)
    assert rec["fa"] == ["x"] and issues == []


# ── exclusive_switch ────────────────────────────────────────────────────

SWITCH = [
    {
        "exclusive_switch": {
            "switch": "entity_type",
            "cases": {"tracker": "tracker_config", "streak": "streak_config"},
        }
    }
]


def test_exclusive_switch_matching_case_passes():
    rec, issues = apply_rules(
        {"entity_type": "tracker", "tracker_config": {"n": 1}, "streak_config": None}, SWITCH
    )
    assert rec is not None and issues == []


def test_exclusive_switch_rejects_null_expected_case():
    rec, issues = apply_rules(
        {"entity_type": "tracker", "tracker_config": None, "streak_config": None}, SWITCH
    )
    assert rec is None
    assert "tracker_config is null" in issues[0]


def test_exclusive_switch_rejects_non_null_other_case():
    rec, issues = apply_rules(
        {"entity_type": "tracker", "tracker_config": {"n": 1}, "streak_config": {"n": 2}}, SWITCH
    )
    assert rec is None
    assert "streak_config is not null" in issues[0]


def test_exclusive_switch_accumulates_both_failures():
    rec, issues = apply_rules(
        {"entity_type": "tracker", "tracker_config": None, "streak_config": {"n": 2}}, SWITCH
    )
    assert rec is None
    assert len(issues) == 2


def test_exclusive_switch_value_with_no_case_requires_all_null():
    """'none' has no entry in `cases`, so every case field must be null."""
    ok, issues = apply_rules(
        {"entity_type": "none", "tracker_config": None, "streak_config": None}, SWITCH
    )
    assert ok is not None and issues == []

    bad, issues = apply_rules(
        {"entity_type": "none", "tracker_config": {"n": 1}, "streak_config": None}, SWITCH
    )
    assert bad is None and len(issues) == 1


# ── cross-cutting ───────────────────────────────────────────────────────


def test_unknown_rule_type_is_an_issue_not_a_crash():
    rec, issues = apply_rules({"a": 1}, [{"bogus": {}}])
    assert rec is None
    assert issues == ["unknown rule type: bogus"]


def test_empty_rules_list_returns_the_record_unchanged():
    record = {"a": 1}
    rec, issues = apply_rules(record, [])
    assert rec is record and issues == []


def test_autofix_mutation_persists_through_a_later_rejection():
    """
    `apply_rules` returns None, but the caller's dict has already been
    mutated by the earlier auto-fix. Harmless today because the engine
    discards rejected records — pinned so a future refactor that starts
    reusing them fails loudly rather than silently shipping fixed-up
    rejects.
    """
    record = {"icon": "gym", "tags": ["run"], "banned": ["run"]}
    rules = [
        {"contains": {"list": "tags", "value": "icon", "autofix": True}},
        {"not_in": {"value": "icon", "list": "tags"}},
    ]
    rec, issues = apply_rules(record, rules)

    assert rec is None and issues
    assert record["tags"] == ["gym", "run"]  # mutated in place regardless


# ── check_field_bounds ──────────────────────────────────────────────────


def test_str_bounds_measure_the_stripped_value():
    """The .strip() is load-bearing: padding must not satisfy min_length."""
    fields = {"text": {"type": "str", "min_length": 3}}
    assert check_field_bounds({"text": "   ab   "}, fields) == ["text too short (2 chars)"]
    assert check_field_bounds({"text": "abc"}, fields) == []


def test_str_max_length():
    assert check_field_bounds({"t": "abcdef"}, {"t": {"max_length": 3}}) == ["t too long (6 chars)"]


def test_list_item_bounds():
    fields = {"tags": {"min_items": 2, "max_items": 3}}
    assert check_field_bounds({"tags": ["a"]}, fields)
    assert check_field_bounds({"tags": ["a", "b", "c", "d"]}, fields)
    assert check_field_bounds({"tags": ["a", "b"]}, fields) == []


def test_missing_field_is_not_an_issue():
    assert check_field_bounds({}, {"text": {"min_length": 5}}) == []


def test_length_spec_on_an_int_value_is_ignored():
    """Neither isinstance branch matches, so the spec simply does nothing."""
    assert check_field_bounds({"n": 1}, {"n": {"min_length": 5}}) == []


def test_multiple_violations_accumulate():
    issues = check_field_bounds(
        {"a": "x", "b": ["1", "2", "3"]},
        {"a": {"min_length": 5}, "b": {"max_items": 2}},
    )
    assert len(issues) == 2
