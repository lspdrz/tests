"""A reply saved as a copy carried the original's rating, open-webui/open-webui#30961.

Fix commit `693f666c6` (PR open-webui/open-webui#30962). Editing a reply and choosing Save As
Copy spread the original message into the copy, its `annotation` and `feedbackId` included. The
copy showed the original's thumbs, and rating it updated the original's feedback row, so the
original's rating was overwritten. The copy now starts unrated.

Discriminates: passes on the efe63bd34 build, fails on it with `693f666c6` reverted (the copy
holds the original's feedback id and rating it rewrites the original's feedback).
"""

from __future__ import annotations

import json
import time

import pytest
from playwright.sync_api import Page

from harness import upstream as reply
from utils.chat_ui import conversation, expect_reply, last_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "suggest a name for a cat"
ANSWER = "Call it Pepper."
COPY = "Call it Biscuit."


def is_feedback_write(response) -> bool:
    return response.request.method == "POST" and "/evaluations/feedback" in response.url


def rate(page: Page, verdict: str):
    last_reply(page).hover()
    with page.expect_response(is_feedback_write) as written:
        conversation(page).get_by_role("button", name=verdict).last.click()
    return written.value


def save_as_copy(page: Page, text: str) -> None:
    last_reply(page).hover()
    conversation(page).get_by_role("button", name="Edit").last.click()
    editor = conversation(page).locator("textarea").last
    editor.fill(text)
    conversation(page).get_by_role("button", name="Save As Copy").click()
    expect_reply(page, text)


def reads(message: dict, text: str) -> bool:
    # a reply may be stored as `output` items with an empty `content`
    return message.get("content") == text or text in json.dumps(message.get("output") or [])


def stored_message(client, chat_id: str, content: str) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
        found = [message for message in messages.values() if reads(message, content)]
        if found:
            return found[0]
        time.sleep(0.2)
    raise AssertionError(f"no stored message reads {content!r}")


def feedback_ratings(client) -> dict[str, int]:
    listed = client.get("/api/v1/evaluations/feedbacks/user").json()["items"]
    return {item["meta"]["message_id"]: item["data"]["rating"] for item in listed}


@pytest.fixture
def rated_reply(make_user, page_for, upstream):
    """A chat whose reply the account rated thumbs up."""
    account = make_user()
    page = page_for(account)
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)
    created = rate(page, "Good Response")
    assert created.url.endswith("/evaluations/feedback"), created.url
    return account, page


def test_a_copy_of_a_rated_reply_starts_unrated(rated_reply):
    account, page = rated_reply
    save_as_copy(page, COPY)
    chat_id = page.url.rsplit("/", 1)[-1]

    with account.client() as client:
        copy = stored_message(client, chat_id, COPY)
        original = stored_message(client, chat_id, ANSWER)

    assert original.get("feedbackId")
    assert not copy.get("feedbackId"), "the copy holds the original's feedback id"
    assert not (copy.get("annotation") or {}).get("rating"), "the copy holds the original's rating"


def test_rating_the_copy_leaves_the_originals_feedback_alone(rated_reply):
    account, page = rated_reply
    save_as_copy(page, COPY)
    chat_id = page.url.rsplit("/", 1)[-1]

    written = rate(page, "Bad Response")

    assert written.url.endswith("/evaluations/feedback"), "rating the copy rewrote a feedback row"
    with account.client() as client:
        original = stored_message(client, chat_id, ANSWER)
        copy = stored_message(client, chat_id, COPY)
        ratings = feedback_ratings(client)
    assert ratings == {original["id"]: 1, copy["id"]: -1}
