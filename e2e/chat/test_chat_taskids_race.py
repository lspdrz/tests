"""Regression: a chat reload finishing late forgot the generation of a message sent meanwhile.

Issue open-webui/open-webui#25217, fixed by 2856def6c. Reloading a chat asks the server which
generations still run in it and then records that answer. A message sent while the answer was
on its way registered its own generation, and the late answer, taken before that message
existed, replaced it with nothing: the page treated the chat as idle while the reply still
streamed, and offered to delete the messages under it. The fix drops an answer that arrives
after the page's own record changed.

The page reloads the chat after `/compact`. The test holds the page's running-generations
request until a new message streams, then lets through the answer the server gave before the
message was sent. The message's Delete action must stay hidden until its reply is done.

Discriminates: passes on the ef67cc3fa build; on a build with the `if (taskIds !==
activeTaskIds) return` guard removed the Delete action shows while the reply streams.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, Route, expect

from harness import upstream as reply
from utils.chat_ui import conversation, expect_reply, send, stop_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
PIECES = [f"piece {number} " for number in range(12)]


@pytest.fixture
def compaction_on(admin, preserve) -> None:
    """`/compact` offered, with a threshold no turn here reaches by itself."""
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        saved = client.post(
            CHAT_CONFIG[1],
            json={
                **current,
                "ENABLE_CONTEXT_COMPACTION": True,
                "CONTEXT_COMPACTION_TOKEN_THRESHOLD": 1_000_000,
                "CONTEXT_COMPACTION_TOKEN_CAP": 1_000_000,
            },
        )
    assert saved.status_code == 200, saved.text


def user_message(page: Page, text: str) -> Locator:
    return conversation(page).locator(".chat-user").filter(has_text=text)


def hold_task_lookups(page: Page) -> list[Route]:
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/tasks/chat/**", hold)
    return held


def wait_until_held(page: Page, held: list[Route], timeout_ms: int = 15_000) -> None:
    for _ in range(timeout_ms // 100):
        if held:
            return
        page.wait_for_timeout(100)
    raise AssertionError("the page never asked which generations run after /compact")


def test_a_late_reload_answer_keeps_the_new_generation_tracked(
    page_for, make_user, upstream, compaction_on
):
    account = make_user(role="admin")
    page = page_for(account)
    first, second = (f"{label} {uuid.uuid4().hex[:6]}" for label in ("first", "second"))
    upstream.queue(
        reply.text("first answer", match=reply.answering(first)),
        reply.text(PIECES, chunk_delay=0.5, match=reply.answering(second)),
    )
    send(page, first)
    expect_reply(page, "first answer")
    expect(stop_button(page)).to_have_count(0)
    chat_id = page.url.rstrip("/").split("/")[-1]

    held = hold_task_lookups(page)
    send(page, "/compact")
    wait_until_held(page, held)
    with account.client() as client:
        stale = client.get(f"/api/tasks/chat/{chat_id}").json()

    send(page, second)
    expect_reply(page, PIECES[1])
    for route in held:
        route.fulfill(json=stale)
    page.unroute("**/api/tasks/chat/**")
    page.wait_for_timeout(1_000)

    expect(stop_button(page)).to_be_visible()
    expect(user_message(page, second).get_by_role("button", name="Delete")).to_have_count(0)

    expect_reply(page, PIECES[-1].strip())
    expect(stop_button(page)).to_have_count(0)
    expect(user_message(page, second).get_by_role("button", name="Delete")).to_have_count(1)
