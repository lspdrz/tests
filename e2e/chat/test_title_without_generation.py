"""Regression: a new chat titled with its first message kept "New Chat" until a reload.

Two fixes, both needed:

- Fix commit `fccd75568` (PR open-webui/open-webui#31355, issue open-webui/open-webui#31348). The
  live `chat:title` event carried the assistant message, still empty at that point, where it now
  carries the saved title.
- Issue open-webui/open-webui#31492, PR open-webui/open-webui#31495. The title is sent while the
  chat is being created, often before the page learns the new chat's id, and the page dropped
  events for a chat id it did not know yet. The page now keeps that title and applies it once the
  id arrives, also when the first reply is stopped or fails, or the task model answers with
  nothing usable.

The tests hold the chat-creation response for half a second, so the title always arrives first
and the race is decided the same way on every run. Twin of
integration/chat/test_title_event_without_generation.py.

Discriminates: fails on dev 00a245b9f in every test (the tab keeps "Open WebUI"), passes with
#31495 applied; a single stored title in place of one per chat fails the background chats test.
"""

from __future__ import annotations

import re
import time

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.chat import send_message
from utils.chat_ui import expect_reply, send, stop_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "Plan a weekend in Graz"
ANSWER = ["Start at ", "the Schlossberg."]
TITLE_TIMEOUT_MS = 10_000
OTHER_BEFORE = "Pack for a hike in Tyrol"
OTHER_AFTER = "Find a cafe in Linz"


def hold_chat_creation(page) -> None:
    def hold_response(route):
        response = route.fetch()
        time.sleep(0.5)
        route.fulfill(response=response)

    page.route("**/api/chat/completions", hold_response)


def expect_prompt_title(page) -> None:
    expect(page).to_have_title(re.compile(re.escape(PROMPT)), timeout=TITLE_TIMEOUT_MS)


def saved_title(client, chat_id: str, expected: str) -> str:
    deadline = time.monotonic() + 5
    title = ""
    while time.monotonic() < deadline:
        title = client.get(f"/api/v1/chats/{chat_id}").json()["title"]
        if title == expected:
            break
        time.sleep(0.1)
    return title


@pytest.fixture
def title_generation_on(admin, preserve) -> None:
    preserve("tasks")
    with admin.client() as client:
        current = client.get("/api/v1/tasks/config").json()
        client.post(
            "/api/v1/tasks/config/update", json={**current, "ENABLE_TITLE_GENERATION": True}
        ).raise_for_status()


def test_a_new_chat_is_titled_with_its_first_message_without_a_reload(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    upstream.queue(reply.text(ANSWER, chunk_delay=1.0, match=reply.answering(PROMPT)))
    hold_chat_creation(page)

    send(page, PROMPT)

    expect_reply(page, "Start at the Schlossberg.")
    expect_prompt_title(page)


def test_the_title_shows_after_the_first_reply_is_stopped(page_for, make_user, upstream):
    page = page_for(make_user())
    upstream.queue(reply.text(["a "] * 60, chunk_delay=0.2, match=reply.answering(PROMPT)))
    hold_chat_creation(page)

    send(page, PROMPT)
    expect(stop_button(page)).to_be_visible(timeout=TITLE_TIMEOUT_MS)
    page.wait_for_timeout(1500)
    stop_button(page).click()

    expect_prompt_title(page)


def test_the_title_shows_when_the_first_reply_fails(page_for, make_user, upstream):
    page = page_for(make_user())
    upstream.queue(reply.error(500, "boom", match=reply.answering(PROMPT)))
    hold_chat_creation(page)

    send(page, PROMPT)

    expect_prompt_title(page)


def test_the_title_shows_when_the_task_model_answers_nothing_usable(
    page_for, make_user, upstream, title_generation_on
):
    page = page_for(make_user())
    upstream.queue(
        reply.text("no json here", match=lambda body: not body.get("stream")),
        reply.text(ANSWER, chunk_delay=1.0, match=reply.answering(PROMPT)),
    )
    hold_chat_creation(page)

    send(page, PROMPT)

    expect_reply(page, "Start at the Schlossberg.")
    expect_prompt_title(page)


def test_other_chats_titled_in_the_background_keep_their_titles(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    upstream.queue(
        reply.text("Boots.", match=reply.answering(OTHER_BEFORE)),
        reply.text("Try the old town.", match=reply.answering(OTHER_AFTER)),
        reply.text(ANSWER, chunk_delay=1.0, match=reply.answering(PROMPT)),
    )
    others = {}

    def start_other(client, prompt):
        turn = send_message(client, prompt, background_tasks={"title_generation": True})
        others[prompt] = turn.chat_id
        assert saved_title(client, turn.chat_id, prompt) == prompt
        time.sleep(0.3)  # let its title event reach the page

    def hold_with_other_chats(route):
        with account.client() as client:
            start_other(client, OTHER_BEFORE)
            response = route.fetch()
            start_other(client, OTHER_AFTER)
        route.fulfill(response=response)

    page.route("**/api/chat/completions", hold_with_other_chats)
    send(page, PROMPT)
    expect_reply(page, "Start at the Schlossberg.")
    expect_prompt_title(page)
    new_chat_id = page.url.rsplit("/", 1)[-1]
    assert new_chat_id not in others.values()
    page.wait_for_timeout(1500)
    expect_prompt_title(page)

    with account.client() as client:
        for prompt, chat_id in others.items():
            assert client.get(f"/api/v1/chats/{chat_id}").json()["title"] == prompt
            expect(page.locator(f'a[href="/c/{chat_id}"]')).to_contain_text(prompt)
        assert client.get(f"/api/v1/chats/{new_chat_id}").json()["title"] == PROMPT
    expect(page.locator(f'a[href="/c/{new_chat_id}"]')).to_contain_text(PROMPT)
