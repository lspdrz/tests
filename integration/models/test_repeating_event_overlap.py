"""Regression: a repeating event still running when the listed dates began was left out.

Fix `ad01bca2b` (#31606, issue #31605): `GET /api/v1/calendars/events` expanded a repeating
event only into occurrences that start inside the requested range, and looked for them from one
day before it. An occurrence that began before the range and was still running at its start was
dropped, while the same event without a repeat was listed: a weekly event from 23:00 to 01:00
was missing from the next day. The expansion now looks back by the event's length as well and
keeps every occurrence that ends after the range starts.

Discriminates: passes on dev a5bc78300; with ad01bca2b reverted the occurrences running across
the range start are missing. The nearby tests pass on both.
"""

from __future__ import annotations

import datetime as dt

import pytest

from harness.calendar_api import (
    DAY_NS,
    HOUR_NS,
    create_event,
    default_calendar_id,
    events_between,
    set_timezone,
    to_ns,
)

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

# A Tuesday at 23:00 UTC; the range under test starts at the midnight after a later occurrence.
SERIES_START = dt.datetime(2026, 11, 3, 23, 0, tzinfo=dt.timezone.utc)
SERIES_START_NS = to_ns(SERIES_START)


@pytest.fixture
def calendar(make_user):
    """A fresh account reading its calendar in UTC, with its own calendar id."""
    owner = make_user()
    client = owner.client()
    set_timezone(client, "UTC")
    yield client, default_calendar_id(client)
    client.close()


def _repeating_event(client, calendar_id: str, length_ns: int, rrule: str) -> str:
    created = create_event(
        client,
        calendar_id,
        start_at=SERIES_START_NS,
        end_at=SERIES_START_NS + length_ns,
        rrule=rrule,
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _listed_starts(client, event_id: str, start_ns: int, end_ns: int) -> list[dt.datetime]:
    listed = events_between(client, start_ns, end_ns)
    return [
        dt.datetime.fromtimestamp(event["start_at"] // 1_000_000_000, dt.timezone.utc)
        for event in listed
        if event["id"] == event_id
    ]


# ---------------------------------------------------------------- narrow


def test_a_weekly_overnight_event_shows_on_the_morning_after(calendar):
    client, calendar_id = calendar
    event_id = _repeating_event(client, calendar_id, 2 * HOUR_NS, "RRULE:FREQ=WEEKLY")
    second_occurrence = SERIES_START + dt.timedelta(weeks=1)
    morning_after = to_ns(second_occurrence) + HOUR_NS

    starts = _listed_starts(client, event_id, morning_after, morning_after + DAY_NS)

    assert starts == [second_occurrence], (
        f"the weekly 23:00 to 01:00 event running at the start of the next day was listed as "
        f"{starts} (#31605)"
    )


# ---------------------------------------------------------------- broad


@pytest.mark.parametrize(
    ("rrule", "length_days", "into_occurrence_days"),
    [
        ("RRULE:FREQ=WEEKLY", 3, 2),
        ("RRULE:FREQ=MONTHLY", 5, 4),
        ("RRULE:FREQ=YEARLY", 10, 9),
    ],
    ids=["three-day-weekly", "five-day-monthly", "ten-day-yearly"],
)
def test_a_long_occurrence_that_began_days_before_the_range_is_listed(
    calendar, rrule, length_days, into_occurrence_days
):
    client, calendar_id = calendar
    event_id = _repeating_event(client, calendar_id, length_days * DAY_NS, rrule)
    range_start = SERIES_START_NS + into_occurrence_days * DAY_NS

    starts = _listed_starts(client, event_id, range_start, range_start + DAY_NS)

    assert starts == [SERIES_START], (
        f"an occurrence of {length_days} days that began {into_occurrence_days} days before the "
        f"range was listed as {starts} (#31605)"
    )


def test_a_daily_overnight_event_lists_both_the_running_and_the_starting_occurrence(calendar):
    client, calendar_id = calendar
    event_id = _repeating_event(client, calendar_id, 3 * HOUR_NS, "RRULE:FREQ=DAILY")
    next_midnight = SERIES_START_NS + HOUR_NS

    starts = _listed_starts(client, event_id, next_midnight, next_midnight + DAY_NS)

    assert starts == [SERIES_START, SERIES_START + dt.timedelta(days=1)]


# ---------------------------------------------------------------- nearby


def test_an_occurrence_that_ended_as_the_range_began_is_left_out(calendar):
    client, calendar_id = calendar
    event_id = _repeating_event(client, calendar_id, 2 * HOUR_NS, "RRULE:FREQ=WEEKLY")
    occurrence_end = SERIES_START_NS + 2 * HOUR_NS

    assert _listed_starts(client, event_id, occurrence_end, occurrence_end + DAY_NS) == []


def test_an_occurrence_starting_inside_the_range_is_still_listed_once(calendar):
    client, calendar_id = calendar
    event_id = _repeating_event(client, calendar_id, 2 * HOUR_NS, "RRULE:FREQ=WEEKLY")
    day_before = SERIES_START_NS + 7 * DAY_NS - 23 * HOUR_NS

    starts = _listed_starts(client, event_id, day_before, day_before + DAY_NS)

    assert starts == [SERIES_START + dt.timedelta(weeks=1)]
