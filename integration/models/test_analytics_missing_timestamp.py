"""Regression: one message without a timestamp broke the analytics chart for All time.

Fix f25708484 (PR #31888, issue #27316). A chat stored through the API with a null message
`timestamp` wrote the first such message with no time and failed on the rest, so the replies after
it never reached analytics. A message row left without a time then failed
`GET /api/v1/analytics/daily` without a date range (the dashboard's All time) with a 500, as every
row's time is turned into a date. Such messages now get the time they were saved, and a row already
stored without a time counts on today.

Discriminates: passes on dev b859124f9, fails with f25708484 reverted (All time answers 500 once a
stored message has no time, and a reply saved without a timestamp is counted nowhere).
"""

from __future__ import annotations

import time
import uuid
from datetime import date
from typing import Iterator

import pytest

from harness.actors import Actor
from harness.backends import write_rows
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PUBLIC_READ = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]
DAY = 24 * 3600


@pytest.fixture
def counted_model(admin) -> Iterator[str]:
    """A preset only this test's chats name, so its counts are the test's own."""
    model_id = f"untimed-{uuid.uuid4().hex[:8]}"
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
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def owner(make_user) -> Iterator[Actor]:
    """A fresh account whose chats are deleted afterwards, untimed rows included."""
    account = make_user()
    yield account
    with account.client() as client:
        client.delete("/api/v1/chats/")


def seed_answered(owner: Actor, model_id: str, timestamp: int | None) -> str:
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "a question", "timestamp": timestamp},
                {"role": "assistant", "content": "an answer", "timestamp": timestamp},
            ],
            model=model_id,
        )
    return chat_id


def daily_counts(admin: Actor, model_id: str, **window) -> dict[str, int]:
    with admin.client() as client:
        response = client.get("/api/v1/analytics/daily", params=window)
    assert response.status_code == 200, f"the daily chart failed to load: {response.text}"
    return {
        entry["date"]: entry["models"][model_id]
        for entry in response.json()["data"]
        if model_id in entry["models"]
    }


def last_day() -> dict[str, int]:
    now = int(time.time())
    return {"start_date": now - DAY, "end_date": now + 3600}


def test_all_time_loads_with_a_message_saved_without_a_timestamp(admin, owner, counted_model):
    seed_answered(owner, counted_model, timestamp=None)

    assert daily_counts(admin, counted_model) == {date.today().isoformat(): 1}


def test_a_message_saved_without_a_timestamp_counts_in_the_last_day(admin, owner, counted_model):
    seed_answered(owner, counted_model, timestamp=None)

    assert sum(daily_counts(admin, counted_model, **last_day()).values()) == 1


def test_all_time_loads_with_a_message_already_stored_without_a_time(
    instance, admin, owner, counted_model
):
    chat_id = seed_answered(owner, counted_model, timestamp=int(time.time()))
    write_rows(
        instance,
        "UPDATE chat_message SET created_at = NULL WHERE chat_id = :chat_id",
        [{"chat_id": chat_id}],
    )

    assert daily_counts(admin, counted_model) == {date.today().isoformat(): 1}


# ---------------------------------------------------------------- nearby


def test_a_timestamped_message_still_counts_on_its_own_day(admin, owner, counted_model):
    three_days_ago = int(time.time()) - 3 * DAY
    seed_answered(owner, counted_model, timestamp=three_days_ago)

    expected_day = date.fromtimestamp(three_days_ago).isoformat()
    assert daily_counts(admin, counted_model) == {expected_day: 1}
    assert daily_counts(admin, counted_model, **last_day()) == {}
