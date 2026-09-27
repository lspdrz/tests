"""The analytics chart for the last 24 hours ignored the selected group, #30977.

Fix commit `9836367ab` (open-webui/open-webui#30978). The admin analytics page asks
`GET /api/v1/analytics/daily` for its chart, with `granularity=hourly` for the 24-hour range and
the selected `group_id`. The daily counts applied the group but the hourly ones dropped it, so
that chart counted every user's messages while the rest of the page was filtered to the group.

Discriminates: passes on dev efe63bd34; with 9836367ab reverted the hourly count filtered to the
group also holds the outsider's reply; the daily and unfiltered counts pass on both.
"""

from __future__ import annotations

import time
import uuid
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.access import make_group
from harness.actors import Actor
from harness.chat import ask
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PUBLIC_READ = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]


@pytest.fixture
def counted_model(admin) -> Iterator[str]:
    """A preset only this test chats with, so its counts are the test's own."""
    model_id = f"analytics-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": {},
                "params": {},
                "access_grants": PUBLIC_READ,
            },
        )
        assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": True}).raise_for_status()
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def group_and_chats(admin, make_user, upstream, counted_model) -> str:
    """Two replies for a group member and one for someone outside the group."""
    member, outsider = make_user(), make_user()
    group_id = make_group(admin, [member])
    for actor, prompt in ((member, "first"), (member, "second"), (outsider, "third")):
        upstream.queue(reply.text(f"reply to {prompt}", match=reply.answering(prompt)))
        with actor.client() as client:
            ask(client, prompt, model=counted_model)
    return group_id


def _replies_counted(admin: Actor, model_id: str, **params) -> int:
    now = int(time.time())
    window = {"start_date": now - 24 * 3600, "end_date": now + 3600}
    with admin.client() as client:
        response = client.get("/api/v1/analytics/daily", params={**window, **params})
    assert response.status_code == 200, response.text
    return sum(entry["models"].get(model_id, 0) for entry in response.json()["data"])


def test_the_hourly_chart_counts_only_the_selected_group(admin, group_and_chats, counted_model):
    counted = _replies_counted(admin, counted_model, granularity="hourly", group_id=group_and_chats)
    assert counted == 2, "the 24-hour chart counted users outside the selected group (#30977)"


# ---------------------------------------------------------------- nearby


def test_the_daily_chart_counts_only_the_selected_group(admin, group_and_chats, counted_model):
    counted = _replies_counted(admin, counted_model, granularity="daily", group_id=group_and_chats)
    assert counted == 2


@pytest.mark.parametrize("granularity", ["hourly", "daily"])
def test_without_a_group_every_reply_is_counted(admin, group_and_chats, counted_model, granularity):
    assert _replies_counted(admin, counted_model, granularity=granularity) == 3
