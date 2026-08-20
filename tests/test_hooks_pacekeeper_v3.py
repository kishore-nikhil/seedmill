"""
The pacekeeper_nlp_v3 hooks.

Loaded from the real file via `load_hooks_module`, so these also assert
the module imports cleanly — it reads pacekeeper_icons.yaml at import
time, which is the fragile part of the hooks contract.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

HOOKS_REL = Path("config/tasks/pacekeeper_nlp_v3_hooks.py")


@pytest.fixture(scope="module")
def hooks(repo_root, load_hooks_module):
    return load_hooks_module(repo_root / HOOKS_REL)


# ── _query_numbers ──────────────────────────────────────────────────────


def test_query_numbers_digits_and_k_suffix(hooks):
    nums = hooks._query_numbers("run 5k every tuesday and thursday")
    assert 5.0 in nums          # the literal digit
    assert 5000.0 in nums       # the k-suffix reading
    assert 2.0 in nums          # two distinct weekday names


def test_query_numbers_range_midpoints(hooks):
    assert 3.5 in hooks._query_numbers("3-4 times a day")
    assert 7.5 in hooks._query_numbers("sleep 7 to 8 hours")


def test_query_numbers_spelled_out_words(hooks):
    nums = hooks._query_numbers("a couple of glasses")
    assert 1.0 in nums and 2.0 in nums


def test_query_numbers_strips_thousands_separators(hooks):
    assert 1500.0 in hooks._query_numbers("1,500 steps")


def test_query_numbers_empty_when_no_number_stated(hooks):
    assert hooks._query_numbers("drink more water") == set()


# ── _states_forward_horizon ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "query,expected",
    [
        ("10 day streak starting from yesterday", True),
        ("for the next 3 months", True),      # only last/past are excluded
        ("started 3 weeks ago", False),       # elapsed
        ("i work out 5 days a week", False),  # a rate
        ("for the past 3 months", False),     # elapsed
        ("i journal every day", False),       # no digit at all
    ],
)
def test_states_forward_horizon(hooks, query, expected):
    assert hooks._states_forward_horizon(query) is expected


# ── _numeral_grounded ───────────────────────────────────────────────────


def test_numeral_grounded_catches_unit_conversion(hooks):
    """
    The failure this whole mechanism exists for: "run 5km" labeled 3.1
    (miles) is a number in the label that appears nowhere in the text.
    """
    assert hooks._numeral_grounded(3.1, "run 5km a day", set()) is False
    assert hooks._numeral_grounded(5.0, "run 5km a day", set()) is True


def test_numeral_grounded_passes_when_nothing_is_stated(hooks):
    assert hooks._numeral_grounded(None, "anything at all", set()) is True
    assert hooks._numeral_grounded(7.0, "drink water daily", set()) is True


def test_numeral_grounded_allows_the_implicit_one(hooks):
    assert hooks._numeral_grounded(1.0, "log my headaches whenever they hit 3 times", set())


def test_numeral_grounded_accepts_an_extra_ground(hooks):
    """Slot count: "at 9am and 9pm" states 9, but 2 doses is right."""
    assert hooks._numeral_grounded(2.0, "take my pills at 9am and 9pm", {2.0}) is True


# ── _resolve_icon ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "slug,entity,expected",
    [
        ("gym", "tracker", "gym"),            # already in the picker
        ("GYM  ", "tracker", "gym"),          # strip + lower
        ("reading", "streak", "book"),        # cross-entity swap
        ("medication", "tracker", "custom"),  # swap to the tracker default
        ("running", "health", "gym"),         # the health picker's "Therapy" glyph
        ("nonsense-slug", "health", "medication"),  # health default
        (None, "streak", "streak"),           # streak default
    ],
)
def test_resolve_icon(hooks, slug, entity, expected):
    assert hooks._resolve_icon(slug, entity) == expected


def test_resolve_icon_always_lands_inside_the_entity_picker(hooks):
    """
    The invariant the function exists for: whatever goes in, what comes
    out is renderable by that entity's picker.
    """
    for entity, allowed in hooks._ICONS_BY_ENTITY.items():
        for slug in [None, "", "totally-made-up", "gym", "book", "medication"]:
            assert hooks._resolve_icon(slug, entity) in allowed


# ── postprocess ─────────────────────────────────────────────────────────


def _rec(**kw):
    base = {
        "id": "x",
        "name": "Test entity",
        "query": "",
        "entity_type": "tracker",
        "recommended_icon": "gym",
        "recommended_color": "blue",
    }
    base.update(kw)
    return base


def test_none_entity_gets_no_icon(hooks):
    """
    An out-of-scope query creates no entity, so a free icon would teach
    the icon head to fire on exactly the inputs the entity head rejects.
    """
    out = hooks.postprocess(
        _rec(
            entity_type="none",
            query="how do i delete my old gym records",
            recommended_icon="gym",
        ),
        {},
    )
    assert out is not None
    assert out["recommended_icon"] is None


def test_percentage_target_on_a_0_1_scale_is_rescaled(hooks):
    out = hooks.postprocess(
        _rec(
            query="hit 85% adherence",
            tracker_config={"goal_type": "percentage", "target_value": 0.85},
        ),
        {},
    )
    assert out["tracker_config"]["target_value"] == 85.0
    assert out["tracker_config"]["unit"] == "percent"


def test_percentage_target_out_of_range_is_rejected(hooks):
    assert hooks.postprocess(
        _rec(query="hit 150", tracker_config={"goal_type": "percentage", "target_value": 150}), {}
    ) is None


def test_at_least_phrasing_flips_exact_count_to_minimum_average(hooks):
    out = hooks.postprocess(
        _rec(query="run at least 3 miles a day",
             tracker_config={"goal_type": "exactCount", "target_value": 3}),
        {},
    )
    assert out["tracker_config"]["goal_type"] == "minimumAverage"


def test_tracker_target_not_in_the_query_is_rejected(hooks):
    assert hooks.postprocess(
        _rec(
            query="run 5km daily",
            tracker_config={"goal_type": "exactCount", "target_value": 3.1},
        ),
        {},
    ) is None


def test_streak_calendar_date_is_rejected(hooks):
    assert hooks.postprocess(
        _rec(entity_type="streak", query="start a streak",
             streak_config={"streak_type": "countUp", "start_date": "2026-03-12"}),
        {},
    ) is None


def test_streak_accepts_verbatim_user_phrasing_as_start_date(hooks):
    out = hooks.postprocess(
        _rec(entity_type="streak", query="no sugar since march",
             streak_config={"streak_type": "countUp", "start_date": "march 12th"}),
        {},
    )
    assert out is not None
    assert out["streak_config"]["start_date"] == "march 12th"


def test_countdown_without_a_duration_is_rejected(hooks):
    assert hooks.postprocess(
        _rec(entity_type="streak", query="30 day challenge",
             streak_config={"streak_type": "countDown", "start_date": "today",
                            "duration_value": None, "duration_unit": None}),
        {},
    ) is None


def test_duration_value_and_unit_must_be_set_together(hooks):
    assert hooks.postprocess(
        _rec(entity_type="streak", query="30 day challenge",
             streak_config={"streak_type": "countDown", "start_date": "today",
                            "duration_value": 30, "duration_unit": None}),
        {},
    ) is None


def test_buildup_with_a_past_start_is_rejected(hooks):
    assert hooks.postprocess(
        _rec(entity_type="streak", query="no smoking",
             streak_config={"streak_type": "buildUp", "start_date": "last monday"}),
        {},
    ) is None


@pytest.mark.parametrize(
    "daily_target,reason",
    [
        (0.43, "below 1 — the '3 per week -> 0.43 per day' bug"),
        (2.5, "not a whole count — the range midpoint applied in the wrong place"),
        (15000, "a quantity that wandered into the count field"),
    ],
)
def test_health_daily_target_rejections(hooks, daily_target, reason):
    assert hooks.postprocess(
        _rec(entity_type="health", query="log it 3-4 times",
             health_config={"health_category": "symptom", "daily_target": daily_target,
                            "schedule_slots": []}),
        {},
    ) is None, reason


def test_medication_dose_count_is_autofixed_to_the_named_slots(hooks):
    """Named times are the trustworthy half of the pair; the count is fixed."""
    out = hooks.postprocess(
        _rec(entity_type="health", query="take my pills at 9am and 9pm",
             health_config={
                 "health_category": "medication",
                 "daily_target": 3,
                 "schedule_slots": [
                     {"hour": 9, "minute": 0, "label": "Morning"},
                     {"hour": 21, "minute": 0, "label": "Night"},
                 ],
             }),
        {},
    )
    assert out is not None
    assert out["health_config"]["daily_target"] == 2.0


@pytest.mark.parametrize(
    "slot",
    [
        {"hour": 25, "minute": 0, "label": "Morning"},
        {"hour": 9, "minute": 99, "label": "Morning"},
        {"hour": 9, "minute": 0, "label": "Lunchtime"},
    ],
)
def test_invalid_schedule_slots_are_rejected(hooks, slot):
    assert hooks.postprocess(
        _rec(entity_type="health", query="log it once a day",
             health_config={"health_category": "symptom", "daily_target": 1,
                            "schedule_slots": [slot]}),
        {},
    ) is None


def test_medication_query_overrides_a_mismatched_icon(hooks):
    out = hooks.postprocess(
        _rec(entity_type="health", query="take my pills at 8am", recommended_icon="gym",
             health_config={"health_category": "medication", "daily_target": 1,
                            "schedule_slots": [{"hour": 8, "minute": 0, "label": "Morning"}]}),
        {},
    )
    assert out["recommended_icon"] == "medication"


def test_reject_logs_its_reason(hooks, caplog):
    """The reason strings are built at ~20 call sites; they must go somewhere."""
    with caplog.at_level(logging.DEBUG, logger=hooks.__name__):
        hooks.postprocess(
            _rec(query="hit 150", tracker_config={"goal_type": "percentage", "target_value": 150}),
            {},
        )
    assert any("percentage target_value" in r.getMessage() for r in caplog.records)


# ── exporters ───────────────────────────────────────────────────────────


def test_build_bert_fills_inapplicable_heads_with_na(hooks):
    out = hooks.build_bert(
        _rec(entity_type="tracker", query="run 3 miles",
             tracker_config={"goal_type": "exactCount", "target_value": 3, "unit": "miles"}),
        {},
    )
    labels = out["labels"]
    assert labels["entity_type"] == "tracker"
    # every head is present; streak/health heads are "n/a", never missing
    assert all(v is not None for v in labels.values())
    assert hooks._NA in labels.values()


def test_build_bert_maps_a_null_icon_to_na(hooks):
    out = hooks.build_bert(_rec(entity_type="none", recommended_icon=None), {})
    assert out["labels"]["icon"] == hooks._NA


def test_build_pairs_has_an_exact_key_set(hooks):
    out = hooks.build_pairs(_rec(query="run daily"), {})
    assert set(out) == {"id", "text", "entity_type", "icon"}
