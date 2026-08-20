"""
One-off cleaner for rows generated before the v3 streak schema change.

The first v3 run wrote streak_config with ISO calendar dates
(start_date "2026-03-12", target_date "2026-10-25"). The schema now carries
the user's own words plus a duration, because every date the generator
wrote was invented: spans that contradicted the query ("45 day challenge"
across 69 days), start dates in 2024 and 2025, three date formats.

This script rewrites what can be recovered from the query text and drops
what cannot, so the existing rows and the resumed run share one schema.

    streak rows   → start_date recovered as a verbatim phrase from the
                    query; duration recovered from an "N days/weeks/months"
                    mention. Dropped when the phrase or a required duration
                    cannot be recovered, or when the query itself names an
                    absolute year (nothing in the schema can hold it).
    tracker/health→ dropped when the target numeral appears nowhere in the
                    query (the unit-conversion class: "run 5km" → 3.1).
    all rows      → ids renumbered contiguously per domain, so the resumed
                    run's per-seed counter cannot collide with a surviving
                    id.

Usage:
    python tools/clean_v3.py output/pacekeeper_nlp_v3.jsonl            # report only
    python tools/clean_v3.py output/pacekeeper_nlp_v3.jsonl --apply
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_HOOKS_PATH = Path(__file__).resolve().parent.parent / "config/tasks/pacekeeper_nlp_v3_hooks.py"
_spec = importlib.util.spec_from_file_location("v3_hooks", _HOOKS_PATH)
hooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hooks)

_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_WEEKDAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"

# Ordered: the most specific phrasing wins, so "next monday" beats "monday"
# and "3 weeks ago" beats a stray weekday later in the sentence.
#
# Every pattern that could match an END marker instead of a start is
# anchored to a starting word ("on the 15th" is a start, "until the 15th"
# is not; "starts monday" is a start, "until sunday" is not).
_START_PATTERNS = [
    re.compile(r"\b(right now|now|today|tonight|tomorrow|yesterday|this morning)\b", re.I),
    re.compile(r"\b(\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago)\b", re.I),
    re.compile(rf"\b(next\s+(?:{_WEEKDAYS}|week|month))\b", re.I),
    re.compile(rf"\b(last\s+(?:{_WEEKDAYS}|week|month|year))\b", re.I),
    re.compile(rf"\b((?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?)\b", re.I),
    re.compile(rf"\b((?:{_MONTHS})\s+(?:first|second|third|fourth|fifth))\b", re.I),
    re.compile(rf"\bin\s+((?:{_MONTHS}))\b", re.I),
    re.compile(rf"\b(?:since|from|starting|started)\s+((?:{_MONTHS})\w*)\b", re.I),
    re.compile(
        rf"\b(?:starts?|starting|started|begins?|beginning|from|on)\s+"
        rf"((?:this\s+|next\s+)?(?:{_WEEKDAYS}))\b",
        re.I,
    ),
    re.compile(rf"\b(this\s+(?:{_WEEKDAYS}))\b", re.I),
    re.compile(
        r"\b(?:on|from|starting|started|since)\s+(the\s+\d{1,2}(?:st|nd|rd|th))\b", re.I
    ),
    re.compile(r"\b(?:since|from)\s+(my \w+|the \w+)\b", re.I),
]

_DURATION = re.compile(
    r"\b(\d+(?:\.\d+)?)[\s-]*(day|days|week|weeks|month|months)\b", re.I
)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


_END_MARKER = re.compile(
    r"\b(ends?|ending|until|till|til|by|through|thru|finish\w*)\s*(on|in)?\s*$", re.I
)


def recover_start(query: str, streak_type: str) -> str | None:
    """
    The user's own words for when the streak starts, or None.

    A date in the query is not automatically the start: "45 day cycling
    challenge ends on october 1st" names the finish. Any match whose
    immediate left context is an end-marker is skipped, otherwise the
    cleaner would write the deadline into start_date and silently invert
    the record.
    """
    for pattern in _START_PATTERNS:
        for match in pattern.finditer(query):
            preceding = query[max(0, match.start() - 18):match.start()]
            if _END_MARKER.search(preceding):
                continue
            phrase = match.group(1).strip().lower()
            if _YEAR.search(phrase):
                return None
            return phrase
    # A forward-looking streak with no stated start begins now, by
    # definition — that holds for a challenge you are setting up ("im
    # starting a 21 day mobility challenge") as much as for an open chain.
    # Only countUp genuinely needs a past start it cannot invent.
    if streak_type in ("buildUp", "countDown"):
        return "now"
    return None


def recover_duration(query: str) -> tuple[float, str] | tuple[None, None]:
    """An 'N days/weeks/months' length stated in the query."""
    if match := _DURATION.search(query):
        value = float(match.group(1))
        unit = match.group(2).lower()
        return value, unit if unit.endswith("s") else unit + "s"
    return None, None


def clean_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Returns (migrated_row_or_None, reason)."""
    query = row.get("query") or ""
    sc = row.get("streak_config")
    tc = row.get("tracker_config")
    hc = row.get("health_config")

    if tc and not hooks._numeral_grounded(tc.get("target_value"), query, set()):
        return None, "tracker target_value not in query (unit conversion)"

    if hc:
        slots = hc.get("schedule_slots") or []
        if not hooks._numeral_grounded(
            hc.get("daily_target"), query, {float(len(slots)), 1.0}
        ):
            return None, "health daily_target not in query (unit conversion)"

    if sc:
        streak_type = sc.get("streak_type")
        if _YEAR.search(query):
            return None, "query names an absolute year the schema cannot hold"

        start = recover_start(query, streak_type)
        if start is None:
            return None, "start_date not recoverable from query"

        value, unit = recover_duration(query)
        if streak_type == "countDown" and value is None:
            return None, "countDown with no length stated in the query"
        if streak_type in ("buildUp", "countUp"):
            # an open-ended chain carries no horizon even if the query
            # happens to mention some other number of days
            value, unit = None, None
        if streak_type == "buildUp" and start not in hooks._NOW_TOKENS:
            return None, "buildUp with a non-now start"

        row["streak_config"] = {
            "streak_type": streak_type,
            "start_date": start,
            "duration_value": value,
            "duration_unit": unit,
        }

    # Final authority is the task's own postprocess hook — the same code
    # every freshly generated row goes through. That covers the icon layer
    # (query-text inference, then per-entity picker resolution, null for
    # none) and re-applies every convention check, so migrated rows and
    # generated rows cannot drift apart. Rows generated before the app's
    # vocabulary existed carry Font Awesome names (dumbbell, glass-water);
    # inference from the query recovers a real slug for most of them, and
    # the alias table catches the rest.
    migrated = hooks.postprocess(row, {})
    if migrated is None:
        return None, "rejected by task hooks"

    return migrated, "kept"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--apply", action="store_true", help="rewrite the file in place")
    args = ap.parse_args()

    path = Path(args.path)
    rows = [
        json.loads(line)
        for line in path.read_text().strip().split("\n")
        if line.strip()
    ]

    kept: list[dict[str, Any]] = []
    reasons: Counter = Counter()
    for row in rows:
        cleaned, reason = clean_row(dict(row))
        reasons[reason] += 1
        if cleaned is not None:
            kept.append(cleaned)

    # contiguous ids per domain, so the engine's resume counter lines up
    counters: defaultdict[str, int] = defaultdict(int)
    for row in kept:
        domain = row.get("domain")
        if domain and "id" in row:
            counters[domain] += 1
            row["id"] = f"{domain}_{counters[domain]:03d}"

    print(f"{len(rows)} rows in, {len(kept)} kept, {len(rows) - len(kept)} dropped")
    for reason, n in reasons.most_common():
        if reason != "kept":
            print(f"  {n:4d}  {reason}")
    print("\nper-domain counts the resumed run will backfill from:")
    short = {d: c for d, c in sorted(counters.items()) if c < 12}
    print(f"  {len(short)} of {len(counters)} domains below the per_seed target of 12")

    if not args.apply:
        print("\n(report only — pass --apply to rewrite)")
        return

    backup = path.with_suffix(path.suffix + ".pre-clean")
    shutil.copy2(path, backup)
    with open(path, "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nrewrote {path}  (backup: {backup})")


if __name__ == "__main__":
    main()
