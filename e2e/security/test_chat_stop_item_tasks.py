"""Regression: Stop must stop every model's reply when several models answer at once.

open-webui 0.11.4 fix for #29816 (PR #29844, `0313ea023` and `e35b907f7`): without Redis,
stopping a chat walked the live list of its replies while each stopped reply removed itself from
that list, so with three models answering one message the Stop button left one of them
streaming to its end. The fix walks a copy and stops every reply in turn.

Three presets on the scripted model answer one message slowly; after Stop none of the replies
may grow any further.

Discriminates: passes on dev ef67cc3fa; with both commits reverted (the live list and the early
return) one reply keeps streaming after Stop and its task is still running.
"""

from __future__ import annotations

import re
import time
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.inflight import SLOW_PIECES
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import replies, send, stop_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "all three of you, tell me a long story"


@pytest.fixture
def three_models(admin):
    """Three presets on the scripted model, readable by every account."""
    model_ids = [f"storyteller-{uuid.uuid4().hex[:8]}" for _ in range(3)]
    with admin.client() as client:
        for model_id in model_ids:
            form = {
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": {},
                "params": {},
                "access_grants": [EVERYONE_READS],
            }
            saved = client.post("/api/v1/models/create", json=form)
            assert saved.status_code == 200, saved.text
    yield model_ids
    with admin.client() as client:
        for model_id in model_ids:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def _running_tasks(client, chat_id: str) -> list[str]:
    return client.get(f"/api/tasks/chat/{chat_id}").json()["task_ids"]


def _reply_texts(page: Page) -> list[str]:
    return replies(page).all_inner_texts()


def test_stop_stops_every_models_reply(page_for, make_user, three_models, upstream):
    person = make_user()
    with person.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"models": three_models}}
        )
    saved.raise_for_status()
    for _ in three_models:
        upstream.queue(reply.text(SLOW_PIECES, chunk_delay=0.3, match=reply.answering(PROMPT)))
    page = page_for(person)

    send(page, PROMPT)
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))
    chat_id = page.url.rsplit("/", 1)[1]
    expect(replies(page)).to_have_count(3)
    for index in range(3):
        expect(replies(page).nth(index)).to_contain_text(SLOW_PIECES[1].strip())
    with person.client() as client:
        assert len(_running_tasks(client, chat_id)) == 3

        stop_button(page).click()

        # The replies stream for six seconds, so a skipped one is still running at two.
        deadline = time.monotonic() + 2
        while _running_tasks(client, chat_id) and time.monotonic() < deadline:
            time.sleep(0.1)
        still_running = _running_tasks(client, chat_id)
    shown_after_stop = _reply_texts(page)
    time.sleep(1.5)  # a stopped reply stays as it was

    assert _reply_texts(page) == shown_after_stop, (
        "a reply kept growing on screen after Stop (#29816)"
    )
    assert still_running == [], "Stop left a model's reply streaming: a reply was skipped (#29816)"
