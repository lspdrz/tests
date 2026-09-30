"""Regression: the Scheduled Tasks calendar showed more runs than an automation's run count.

Fix `909a2075d` (#31604, issue #31600): the calendar page shows an active automation's upcoming
runs, which the server expanded from the automation's next run. A rule that ends after a number
of runs (`COUNT`, with a `DTSTART`) counts them from that DTSTART when it runs, so once runs had
passed the month view showed the full count again from the next run: after two of four daily
runs, four more. The server now expands such a rule from its DTSTART.

The automation's DTSTART lies two days back, as after two runs, and its next run an hour ahead.

Twin of integration/models/test_scheduled_task_run_count.py.

Discriminates: passes on dev a5bc78300 with its build; with 909a2075d reverted in the backend the
month view shows four upcoming runs where two are left.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.calendar_api import set_timezone
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

IN_UTC = {"timezone_id": "UTC", "locale": "en-US"}


@pytest.fixture
def automation_author(make_user, preserve, admin):
    """A fresh account in UTC allowed to create automations; its automations are deleted after."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["automations"] = True
        granted = client.post("/api/v1/users/default/permissions", json=permissions)
    assert granted.status_code == 200, granted.text
    author = make_user()
    with author.client() as client:
        set_timezone(client, "UTC")
    created: list[str] = []
    yield author, created
    with author.client() as client:
        for automation_id in created:
            client.delete(f"/api/v1/automations/{automation_id}/delete")


def _run_chips(page: Page, name: str):
    return page.get_by_role("button", name=name).filter(has_not=page.get_by_role("button"))


def test_a_four_run_schedule_shows_the_two_runs_left(page_for, automation_author):
    author, created = automation_author
    name = f"Report {uuid.uuid4().hex[:6]}"
    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    anchor = now - dt.timedelta(days=2) + dt.timedelta(hours=1)
    rule = f"{anchor.strftime('DTSTART:%Y%m%dT%H%M%S')}\nRRULE:FREQ=DAILY;COUNT=4"
    with author.client() as client:
        response = client.post(
            "/api/v1/automations/create",
            json={
                "name": name,
                "is_active": True,
                "data": {"prompt": "write the report", "model_id": MOCK_MODEL_ID, "rrule": rule},
            },
        )
    assert response.status_code == 200, response.text
    created.append(response.json()["id"])

    page = page_for(author, **IN_UTC)
    with page.expect_response(lambda response: "/calendars/events?" in response.url):
        page.goto("/calendar")
    expect(_run_chips(page, name).first).to_be_visible()

    shown = _run_chips(page, name).count()
    assert shown == 2, (
        f"a schedule of four daily runs, two of them past, shows {shown} upcoming runs (#31600)"
    )
