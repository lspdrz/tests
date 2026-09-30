"""Regression: the Scheduled Tasks calendar showed more runs than an automation's run count.

Fix `909a2075d` (#31604, issue #31600): `GET /api/v1/calendars/events` lists an active
automation's upcoming runs by expanding its rule from the automation's next run. A rule that
ends after a number of runs (`COUNT`, which needs a `DTSTART`) counts them from that DTSTART
when it runs, so once a run had passed the calendar started the count again at the next run and
showed runs that never happen. The calendar now expands such a rule from its DTSTART. The same
fix counts the 5000-entry display limit inside the requested range only, so a frequent schedule
no longer spends the limit on the day before it.

An automation whose DTSTART lies in the past is what the calendar sees after runs have passed.
The automations here are active, as only active ones are listed, with their next run an hour
or more away.

Discriminates: passes on dev a5bc78300; with 909a2075d reverted the calendar lists four runs of a
four-run schedule after two of them have passed. With only its display-limit change reverted, a
minutely schedule anchored two days back lists 3560 runs in a range holding more than 5000. The
nearby test passes on both.
"""

from __future__ import annotations

import datetime as dt

import httpx
import pytest

from harness.calendar_api import DAY_NS, SECOND_NS, events_between, set_timezone, to_ns
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

DISPLAY_LIMIT = 5000


@pytest.fixture
def scheduler(make_user, preserve, admin):
    """`scheduler(rule)` creates an active automation as a fresh user in UTC; yields its lister."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["automations"] = True
        granted = client.post("/api/v1/users/default/permissions", json=permissions)
    assert granted.status_code == 200, granted.text
    client = make_user().client()
    set_timezone(client, "UTC")
    created: list[str] = []

    def create(rule: str) -> str:
        response = client.post(
            "/api/v1/automations/create",
            json={
                "name": "Report",
                "is_active": True,
                "data": {"prompt": "write the report", "model_id": MOCK_MODEL_ID, "rrule": rule},
            },
        )
        assert response.status_code == 200, response.text
        created.append(response.json()["id"])
        return response.json()["id"]

    yield client, create
    for automation_id in created:
        client.delete(f"/api/v1/automations/{automation_id}/delete")
    client.close()


def _dtstart(moment: dt.datetime) -> str:
    return moment.strftime("DTSTART:%Y%m%dT%H%M%S")


def _anchor_two_days_back() -> dt.datetime:
    """Two days ago plus an hour, so two daily runs have passed and the next is an hour away."""
    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    return now - dt.timedelta(days=2) + dt.timedelta(hours=1)


def _listed_runs(client: httpx.Client, automation_id: str, days: int) -> list[dt.datetime]:
    now_ns = to_ns(dt.datetime.now(dt.timezone.utc))
    listed = events_between(client, now_ns, now_ns + days * DAY_NS)
    return [
        dt.datetime.fromtimestamp(event["start_at"] // SECOND_NS, dt.timezone.utc)
        for event in listed
        if event["id"] == f"auto_{automation_id}"
    ]


# ---------------------------------------------------------------- narrow


def test_a_four_run_schedule_shows_only_the_runs_left_after_two_have_passed(scheduler):
    client, create = scheduler
    anchor = _anchor_two_days_back()
    automation_id = create(f"{_dtstart(anchor)}\nRRULE:FREQ=DAILY;COUNT=4")

    runs = _listed_runs(client, automation_id, 10)

    assert runs == [anchor + dt.timedelta(days=2), anchor + dt.timedelta(days=3)], (
        f"a schedule of four daily runs, two of them past, showed {len(runs)} upcoming runs: "
        f"{[str(run) for run in runs]} (#31600)"
    )


# ---------------------------------------------------------------- broad


def test_a_count_schedule_starting_in_the_future_shows_all_its_runs(scheduler):
    client, create = scheduler
    anchor = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    anchor += dt.timedelta(days=1)
    automation_id = create(f"{_dtstart(anchor)}\nRRULE:FREQ=DAILY;COUNT=3")

    runs = _listed_runs(client, automation_id, 10)

    assert runs == [anchor + dt.timedelta(days=offset) for offset in range(3)]


def test_a_minutely_count_schedule_fills_the_display_limit_inside_the_range(scheduler):
    client, create = scheduler
    anchor = _anchor_two_days_back()
    automation_id = create(f"{_dtstart(anchor)}\nRRULE:FREQ=MINUTELY;COUNT=100000")

    runs = _listed_runs(client, automation_id, 5)

    assert len(runs) == DISPLAY_LIMIT, (
        f"a minutely schedule over five days listed {len(runs)} runs; the day before the range "
        f"counted against the {DISPLAY_LIMIT}-entry limit"
    )


# ---------------------------------------------------------------- nearby


def test_a_schedule_without_a_count_still_runs_on_from_the_next_run(scheduler):
    client, create = scheduler
    anchor = _anchor_two_days_back()
    automation_id = create(f"{_dtstart(anchor)}\nRRULE:FREQ=DAILY")

    runs = _listed_runs(client, automation_id, 5)

    assert runs[:3] == [anchor + dt.timedelta(days=offset) for offset in (2, 3, 4)]
    assert len(runs) == 5
