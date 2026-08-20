"""
Hooks for pacekeeper_nlp_v3.

Two jobs:

1. postprocess() — enforce the machine-checkable half of the labeling
   conventions (inherited from v2, where they were derived from the label
   contradictions found in v1's data), plus v3's icon-consistency rules.
   Anything safely normalizable is fixed in place; anything needing
   judgment is rejected so the engine's retry loop regenerates it.

2. EXPORTERS — a "bert" format that flattens each record into the
   multi-head classification target an on-device BERT actually trains on:
   one text input, one label per head, with an explicit "n/a" class for
   heads that do not apply to the record's entity type. Numeric targets
   are kept separate: they are extraction/regression, not classification.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

_log = logging.getLogger(__name__)

# The app's picker lists, aliases and per-entity defaults. Loaded here
# rather than taken from ctx["vocab"] because the engine flattens the
# vocabulary to one set for decoding, and the whole point of this file is
# the part that cannot be flattened: which icons belong to WHICH entity.
_ICONS_PATH = Path(__file__).resolve().parent.parent / "pacekeeper_icons.yaml"
_ICON_CFG: dict[str, Any] = yaml.safe_load(_ICONS_PATH.read_text())
_ICONS_BY_ENTITY: dict[str, set[str]] = {
    entity: set(slugs) for entity, slugs in _ICON_CFG["icons"].items()
}
_ICON_ALIASES: dict[str, str] = {
    str(k).strip().lower(): v for k, v in (_ICON_CFG.get("aliases") or {}).items()
}
_ICON_DEFAULTS: dict[str, str] = _ICON_CFG.get("defaults") or {}

_SLOT_LABELS = {"Morning", "Afternoon", "Evening", "Night", "Bedtime"}
_NOW_TOKENS = {"now", "today", "tomorrow"}
_DURATION_UNITS = {"days", "weeks", "months"}

# A calendar date in start_date, in any of the formats the generator
# reached for before the schema change: ISO, slashed, or a month name
# carrying a year. "march 12th" is fine — that is the user's phrasing and
# the app resolves it. "march 12th 2026" is not: the year is invented.
_CALENDAR_DATE = re.compile(
    r"""
      \b\d{4}-\d{1,2}-\d{1,2}\b          # 2026-03-01
    | \b\d{1,2}/\d{1,2}/\d{2,4}\b        # 3/1/26
    | \b(19|20)\d{2}\b                   # a bare year anywhere
    """,
    re.X,
)

_WORD_NUMBERS = {
    "a": 1.0, "an": 1.0, "one": 1.0, "once": 1.0, "two": 2.0, "twice": 2.0,
    "three": 3.0, "thrice": 3.0, "four": 4.0, "five": 5.0, "six": 6.0,
    "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0, "eleven": 11.0,
    "twelve": 12.0, "dozen": 12.0, "fifteen": 15.0, "twenty": 20.0,
    "thirty": 30.0, "sixty": 60.0, "hundred": 100.0, "half": 0.5,
    "couple": 2.0, "few": 3.0,
}

# Icon consistency. The generator picks freely within the 28-slug
# vocabulary, which drifts on the request types that dominate the dataset —
# v1 spelled the gym icon four different ways. For these, the query text
# decides the label, not the model's mood. First match wins, so order
# matters: specific patterns first ("blood sugar" before "sugar").
#
# Values are canonical slugs; _resolve_icon() maps them onto whichever
# picker list the record's entity_type actually has.
_ICON_BY_QUERY: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(inject\w*|insulin|syringe|jab|vaccin\w*)\b", re.I), "injection"),
    (re.compile(r"\b(blood pressure|bp reading|glucose|blood sugar|a1c|heart rate|pulse|vitals)\b", re.I), "vitals"),
    (re.compile(r"\b(fever|temperature|thermometer)\b", re.I), "temperature"),
    (re.compile(r"\b(smok\w*|cigarette|vap\w*|nicotine)\b", re.I), "quit-smoking"),
    (re.compile(r"\b(pill|pills|tablet|meds?|medication|dose|doses|prescription|inhaler|vitamin|supplement|antibiotic)\w*\b", re.I), "medication"),
    (re.compile(r"\b(headache|migraine|pain|ache|aches|flare|symptom|cramps?|nausea)\w*\b", re.I), "symptom"),
    (re.compile(r"\b(mood|anxiety|anxious|depress\w*|stress|panic|therapy session)\b", re.I), "mood"),
    (re.compile(r"\b(gym|weights?|weightlift\w*|lifting|bench press|squats?|workout|physical therapy)\b", re.I), "gym"),
    (re.compile(r"\b(run|runs|running|jog\w*|5k|10k|marathon|steps?|walk\w*)\b", re.I), "running"),
    # Sports the app has no icon for. fitness_center is its generic
    # exercise glyph, so these belong on "gym" rather than falling through
    # to "custom" — cycling and swimming alone were 15% of the corpus.
    (re.compile(r"\b(cycl\w*|bike|biking|swim\w*|hik\w*|climb\w*|rowing|pilates|stretch\w*|cardio|tennis|basketball|soccer|football|boxing|martial arts|karate|judo|dance|dancing|skat\w*)\b", re.I), "gym"),
    (re.compile(r"\b(water|hydrat\w*|litres?|liters?|glasses of)\b", re.I), "water"),
    (re.compile(r"\b(sleep|bedtime|asleep|in bed|nap)\w*\b", re.I), "sleep"),
    (re.compile(r"\b(meditat\w*|mindful\w*|yoga|breathing exercise|zen)\b", re.I), "meditation"),
    (re.compile(r"\b(read|reading|books?|pages|novel)\b", re.I), "reading"),
    (re.compile(r"\b(code|coding|program\w*|leetcode|commit)\b", re.I), "coding"),
    (re.compile(r"\b(study|studying|class|classes|revision|homework|exam|lecture|language)\w*\b", re.I), "study"),
    (re.compile(r"\b(eat\w*|meal|meals|food|calorie|protein|sugar|snack|diet|veg\w*|junk)\b", re.I), "diet"),
    # NOT sing\w* — that matches "every single day", which appears in half
    # the streak queries in the corpus and silently labelled them "music"
    (re.compile(r"\b(guitar|piano|sing|singing|vocals?|music|instrument|drums?)\b", re.I), "music"),
    (re.compile(r"\b(draw\w*|paint\w*|sketch\w*|photo\w*|art)\b", re.I), "art"),
    (re.compile(r"\b(chores?|clean\w*|laundry|dishes|shopping|groceries|declutter\w*)\b", re.I), "errands"),
    (re.compile(r"\b(volunteer\w*|charity|donat\w*)\b", re.I), "volunteer"),
    (re.compile(r"\b(work|office|job|meeting|standup|inbox|email)\b", re.I), "office"),
    (re.compile(r"\b(friends?|family|call\w*|date night|social)\b", re.I), "social"),
    (re.compile(r"\b(skincare|self.?care|spa|relax\w*)\b", re.I), "wellness"),
    (re.compile(r"\b(bathroom|toilet|pee|urinat\w*)\b", re.I), "bathroom"),
]

# The few slugs with a genuine counterpart in another entity's picker.
# Everything else that lands outside its entity's list falls back to that
# entity's default rather than to a guess.
_CROSS_ENTITY: dict[tuple[str, str], str] = {
    ("reading", "streak"): "book",
    ("book", "tracker"): "reading",
    ("book", "health"): "wellness",
    ("quit-smoking", "tracker"): "custom",
    ("quit-smoking", "health"): "wellness",
    ("streak", "tracker"): "custom",
    ("launch", "tracker"): "custom",
    ("office", "streak"): "coding",
    ("study", "streak"): "book",
    ("art", "streak"): "music",
    ("mood", "tracker"): "custom",
    ("symptom", "tracker"): "custom",
    ("vitals", "tracker"): "custom",
    ("medication", "tracker"): "custom",
    ("wellness", "tracker"): "meditation",
    ("wellness", "streak"): "meditation",
    # the health picker has no running glyph; its fitness_center icon is
    # the one it labels "Therapy"
    ("running", "health"): "gym",
    ("reading", "health"): "wellness",
    ("coding", "health"): "wellness",
}


def _resolve_icon(slug: str | None, entity: str) -> str:
    """
    Map any icon name onto a slug in this entity's picker list.

    The app shows a different picker per entity type and marks the current
    icon selected by comparing code points, so an entity carrying an icon
    from another entity's list shows nothing selected when the user opens
    it to edit. Grammar-constrained decoding can only enforce the union of
    all three lists — this is where the per-entity half is enforced.
    """
    allowed = _ICONS_BY_ENTITY.get(entity, set())
    key = (slug or "").strip().lower()
    key = _ICON_ALIASES.get(key, key)

    if key in allowed:
        return key
    if swap := _CROSS_ENTITY.get((key, entity)):
        return swap if swap in allowed else _ICON_DEFAULTS.get(entity, "custom")
    return _ICON_DEFAULTS.get(entity, "custom")


def _query_numbers(query: str) -> set[float]:
    """
    Every number a reader could reasonably take from the query.

    Deliberately generous — this set is used to *reject* records whose
    numeral appears nowhere in the query, so anything it misses is a false
    rejection. It covers digits, k-suffixes ("10k" is both 10 and 10000
    depending on whether it is steps or kilometres), range midpoints
    ("3-4 times" -> 3.5, the labeling convention), and spelled-out numbers.
    """
    nums: set[float] = set()

    raw = re.findall(r"\d+(?:[.,]\d+)?", query)
    ordered: list[float] = []
    for token in raw:
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        nums.add(value)
        ordered.append(value)

    for value, suffix in re.findall(r"(\d+(?:\.\d+)?)\s*([kK])\b", query):
        nums.add(float(value) * 1000)

    # range midpoints: "3-4 times", "7 to 8 hours"
    for a, b in re.findall(r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)", query):
        nums.add((float(a) + float(b)) / 2)

    for word in re.findall(r"[a-z]+", query.lower()):
        if word in _WORD_NUMBERS:
            nums.add(_WORD_NUMBERS[word])

    # Counting named days is inference, not conversion: "i run 5k every
    # tuesday and thursday" is legitimately 2 sessions a week even though
    # the query contains no 2. Same shape as counting named times of day
    # for medication slots.
    weekdays = len(set(re.findall(
        r"\b(mon|tues?|wednes|thurs?|fri|satur|sun)(?:day)?s?\b", query.lower()
    )))
    if weekdays:
        nums.add(float(weekdays))

    return nums


_HORIZON = re.compile(r"\b\d+(?:\.\d+)?[-\s]?(?:day|week|month)s?\b", re.I)


def _states_forward_horizon(query: str) -> bool:
    """
    True when the query names a length the streak runs *forward* for.

    "10 day streak starting from yesterday" states a horizon; "started 3
    weeks ago" states elapsed time and "5 days a week" states a frequency.
    Only the first is a countDown. Deliberately conservative — a miss lets
    a row through, a false positive throws away a good one.
    """
    q = query.lower()
    for m in _HORIZON.finditer(q):
        pre, post = q[max(0, m.start() - 24):m.start()], q[m.end():m.end() + 20]
        # elapsed ("3 weeks ago", "3 months since"), or a rate ("2 days every week")
        if re.search(r"\bago\b|\bsince\b|\bevery\b|\ba week\b|\ba month\b", post):
            continue
        # elapsed ("for the past 3 months", "been 3 weeks") or a rate ("per 3 days").
        # "for the next 3 months" is a horizon, so only last/past are excluded here.
        if re.search(
            r"\bsince\b|\blast\b|\bpast\b|\bbeen\b|\bfor the (last|past)\b"
            r"|\bevery\b|\beach\b|\bper\b",
            pre,
        ):
            continue
        return True
    return False


def _numeral_grounded(value: float | None, query: str, extra: set[float]) -> bool:
    """
    True when `value` is traceable to the query.

    The unit-conversion class of error — "run 5km" labeled 3.1 — is exactly
    a number that appears in the label but nowhere in the text. If the query
    states no number at all, there is nothing to contradict, so it passes.
    """
    if value is None:
        return True
    stated = _query_numbers(query)
    if not stated:
        # nothing to contradict; the model had to pick a default
        return True
    # "every friday night" -> 1 a week is the other everyday inference, and
    # a conversion landing exactly on 1.0 is not the failure mode
    return any(abs(float(value) - n) < 0.011 for n in stated | extra | {1.0})


def _reject(reason: str) -> None:
    """
    Reject this record. The engine counts hook rejections under
    `rejected_rules`; the reason is emitted at DEBUG so a run can be
    diagnosed without changing the return contract.
    """
    _log.debug("rejected: %s", reason)
    return None


def postprocess(record: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    tc = record.get("tracker_config")
    sc = record.get("streak_config")
    hc = record.get("health_config")
    query = record.get("query") or ""

    # ── tracker ────────────────────────────────────────────────────────
    if tc:
        unit = (tc.get("unit") or "").strip()
        if unit in ("%", "pct", "Pct", "PERCENT", "Percent"):
            tc["unit"] = "percent"

        if tc.get("goal_type") == "percentage":
            tv = tc.get("target_value") or 0.0
            # 0-1 scale slipped through: 0.85 -> 85. A genuine sub-1%
            # target does not exist in this app.
            if 0.0 < tv <= 1.0:
                tc["target_value"] = round(tv * 100.0, 2)
            elif not (1.0 < tv <= 100.0):
                return _reject("percentage target_value outside (0, 100]")
            tc["unit"] = "percent"

        if tc.get("target_value") is not None and tc["target_value"] <= 0:
            return _reject("non-positive target_value")

        # unit conversion: the label's number must exist in the user's text
        if not _numeral_grounded(tc.get("target_value"), query, set()):
            return _reject(
                f"target_value {tc.get('target_value')} appears nowhere in the query"
            )

        # "at least" / "minimum" / a range in the query is minimumAverage by
        # convention; v1's single largest source of contradictory labels
        if re.search(r"\bat least\b|\bminimum\b|\bor more\b|\bno less than\b", query, re.I):
            if tc.get("goal_type") == "exactCount":
                tc["goal_type"] = "minimumAverage"

    # ── streak ─────────────────────────────────────────────────────────
    if sc:
        st = sc.get("streak_type")
        start = (sc.get("start_date") or "").strip().lower()
        dvalue = sc.get("duration_value")
        dunit = sc.get("duration_unit")

        # belt-and-suspenders for the schema change: the field is gone, but
        # a leftover date anywhere in the block means the model is still
        # inventing calendars
        if "target_date" in sc:
            return _reject("target_date is not part of the schema any more")
        if _CALENDAR_DATE.search(start):
            return _reject(f"start_date '{start}' is a calendar date")

        if dunit is not None and dunit not in _DURATION_UNITS:
            return _reject(f"duration_unit '{dunit}' outside enum")
        if (dvalue is None) != (dunit is None):
            return _reject("duration_value and duration_unit must be set together")
        if dvalue is not None and dvalue <= 0:
            return _reject("non-positive duration_value")

        if st == "countDown" and dvalue is None:
            return _reject("countDown without a duration")
        if st in ("buildUp", "countUp") and dvalue is not None:
            return _reject(f"{st} with a duration set")
        # the length has to be the user's, not the model's
        if not _numeral_grounded(dvalue, query, set()):
            return _reject(f"duration_value {dvalue} appears nowhere in the query")
        # buildUp starts now by definition; a named past start means the
        # chain already exists, which is countUp
        if st == "buildUp" and start not in _NOW_TOKENS:
            return _reject("buildUp with a non-now start_date")

        # a stated forward length is a countDown, full stop. The convention
        # held 187/191 times; the strays were "30 day no spending challenge,
        # started last monday" labeled countUp — a horizon read as elapsed
        # time because the start was in the past. Both facts are true; only
        # countDown can express them together.
        if st != "countDown" and _states_forward_horizon(query):
            return _reject(f"{st} but the query states a forward horizon")

    # ── health ─────────────────────────────────────────────────────────
    if hc:
        dt = hc.get("daily_target")
        slots = hc.get("schedule_slots") or []

        # the v1 "3 per week -> 0.43 per day" bug: daily_target is a literal
        # per-day count or amount, never a fraction
        if dt is None or dt < 1.0:
            return _reject("daily_target below 1")

        # daily_target counts occurrences, so it is a whole number. The
        # range-midpoint convention is right for tracker targets and wrong
        # here: "cramps 2-3 times a day" came back 2.5, which is the v1
        # fractional-target bug wearing a different hat.
        if float(dt) % 1:
            return _reject(f"daily_target {dt} is not a whole count")

        # daily_target counts times per day, so anything above roughly
        # hourly is a quantity that wandered into the count field —
        # "log my steps, around 15k" came back as daily_target 15000
        if dt > 24:
            return _reject(f"daily_target {dt} is a quantity, not a per-day count")

        # ...and the same error in units: "20 mins of walking daily" as
        # daily_target 20. A number the query immediately follows with a
        # time unit is a duration, which makes the record a tracker.
        if re.search(
            rf"\b{re.escape(str(int(dt)) if float(dt).is_integer() else str(dt))}\s*"
            r"(min|mins|minute|minutes|hr|hrs|hour|hours)\b",
            query,
            re.I,
        ):
            return _reject(f"daily_target {dt} is a duration in the query, not a count")

        for slot in slots:
            if not isinstance(slot, dict):
                return _reject("schedule_slot is not an object")
            hour, minute = slot.get("hour"), slot.get("minute")
            if not (isinstance(hour, int) and 0 <= hour <= 23):
                return _reject("slot hour out of range")
            if not (isinstance(minute, int) and 0 <= minute <= 59):
                return _reject("slot minute out of range")
            if slot.get("label") not in _SLOT_LABELS:
                return _reject("slot label outside canonical set")

        # dose-count/slot agreement only binds when times were named AND the
        # target is a count; quantity targets (7.5 hours, 2000 ml) may carry
        # a reminder slot without matching it numerically
        if (
            hc.get("health_category") == "medication"
            and slots
            and float(dt) != float(len(slots))
        ):
            # trust the named times; the count is the error half of the pair
            hc["daily_target"] = float(len(slots))

        # Same unit-conversion check as the tracker side. Two extra grounds
        # beyond the query's own numbers: the slot count ("at 9am and 9pm"
        # states 9, not 2, but 2 doses is right), and 1.0, the convention
        # for as-needed logging regardless of what the query counts.
        if not _numeral_grounded(
            hc.get("daily_target"), query, {float(len(slots)), 1.0}
        ):
            return _reject(
                f"daily_target {hc.get('daily_target')} appears nowhere in the query"
            )

    # ── icon ───────────────────────────────────────────────────────────
    entity = record.get("entity_type")

    # An out-of-scope query creates no entity, so there is no icon. Left
    # free, these pick up the topic's icon ("how do i delete my old gym
    # records?" -> gym), which teaches the icon head to fire on exactly the
    # inputs the entity head is meant to reject.
    if entity == "none":
        record["recommended_icon"] = None
        return record

    icon = record.get("recommended_icon")
    for pattern, canonical in _ICON_BY_QUERY:
        if pattern.search(query):
            icon = canonical
            break

    # a medication log showing a barbell is a mislabel, not a style choice
    if hc and hc.get("health_category") == "medication" and icon not in (
        "medication", "injection"
    ):
        icon = "medication"

    record["recommended_icon"] = _resolve_icon(icon, entity)
    return record


# ── exports ───────────────────────────────────────────────────────────

_NA = "n/a"


def build_bert(record: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Flat multi-head target for an on-device BERT.

    Every head is a closed label set including "n/a", so heads that do not
    apply to the record's entity type still have a defined target and the
    model learns when NOT to predict them.
    """
    tc = record.get("tracker_config") or {}
    sc = record.get("streak_config") or {}
    hc = record.get("health_config") or {}

    return {
        "id": record.get("id"),
        "text": record["query"],
        "labels": {
            "entity_type": record["entity_type"],
            # null icon (entity_type none) becomes the same "n/a" class the
            # other conditional heads use, so every head is defined on
            # every row
            "icon": record.get("recommended_icon") or _NA,
            "color": record["recommended_color"],
            "period": tc.get("period", _NA),
            "goal_type": tc.get("goal_type", _NA),
            "average_period": tc.get("average_period", _NA),
            "schedule_preference": tc.get("schedule_preference", _NA),
            "notification_mode": tc.get("notification_mode", _NA),
            "streak_type": sc.get("streak_type", _NA),
            "duration_unit": sc.get("duration_unit") or _NA,
            "health_category": hc.get("health_category", _NA),
        },
        # numeric targets: extraction/regression heads, not classification.
        # start_date stays out of here on purpose — it is a verbatim span of
        # the query ("next monday"), so it belongs to a span-extraction head
        # or to the app's own date parser, not to a label.
        "numeric": {
            "target_value": tc.get("target_value"),
            "daily_target": hc.get("daily_target"),
            "duration_value": sc.get("duration_value"),
            "slot_count": len(hc.get("schedule_slots") or []) if hc else None,
        },
        "start_date_text": sc.get("start_date"),
        "name": record["name"],
        "domain": record.get("domain"),
        "domain_group": record.get("domain_group"),
    }


def build_pairs(record: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Single-head (text, label) pairs — one file per head is easier to
    sanity-check than a nested dict when debugging class imbalance."""
    return {
        "id": record.get("id"),
        "text": record["query"],
        "entity_type": record["entity_type"],
        "icon": record.get("recommended_icon") or _NA,
    }


EXPORTERS = {"bert": build_bert, "pairs": build_pairs}
