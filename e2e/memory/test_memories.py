"""Journey: a memory is added, edited and deleted under Settings > Personalization.

A fresh account adds a memory through the Actions menu and sees it listed, edits it and sees the
new text, and a chat then sends the model the edited memory as context. Deleting it empties the
list, and it stays gone after the settings are opened again. Switching Memory off in the same tab
keeps a stored memory out of the next chat, and stays off after a reload, while another account's
memory still reaches the model.

Discriminates: passes on dev ac00d40e3; in a backend copy, with
`POST /api/v1/memories/{id}/update` answering without storing the new text the edited memory
never shows, and with `DELETE /api/v1/memories/{id}` answering true without deleting the memory
is listed again once the settings are reopened; in a frontend copy, a chat sending the memory
feature whatever the account's Memory switch says turns the switch test red.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FIRST = "I keep bees on the roof."
EDITED = "I keep bees and two hens on the roof."
MEMORY_FIELD = "Add a preference, fact, or instruction about you"


def _personalization(page: Page) -> Locator:
    page.goto("/?settings=personalization")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("heading", name="Personalization")).to_be_visible()
    return settings


def test_a_memory_is_added_edited_used_and_deleted(page_for, make_user, upstream):
    page = page_for(make_user())
    settings = _personalization(page)
    settings.get_by_role("button", name="Actions").first.click()
    page.get_by_role("menu").get_by_role("button", name="Add Memory").click()
    adding = page.get_by_role("dialog").filter(has_text="Add Memory")
    adding.get_by_role("textbox", name=MEMORY_FIELD).fill(FIRST)
    adding.get_by_role("button", name="Add", exact=True).click()
    expect(settings.get_by_text(FIRST)).to_be_visible()

    settings.get_by_role("button", name="Edit").click()
    editing = page.get_by_role("dialog").filter(has_text="Edit Memory")
    expect(editing.get_by_role("textbox", name=MEMORY_FIELD)).to_have_value(FIRST)
    editing.get_by_role("textbox", name=MEMORY_FIELD).fill(EDITED)
    editing.get_by_role("button", name="Update").click()
    expect(settings.get_by_text(EDITED)).to_be_visible()

    settings = _personalization(page)
    expect(settings.get_by_text(EDITED)).to_be_visible()
    expect(settings.get_by_text(FIRST)).to_have_count(0)

    question = "what do I keep on the roof?"
    upstream.queue(reply.text("Bees and two hens.", match=reply.answering(question)))
    page.goto("/")
    send(page, question)
    expect_reply(page, "Bees and two hens.")
    sent = json.dumps(next(filter(reply.answering(question), upstream.chat_requests())))
    assert EDITED in sent, "the chat was not sent the edited memory"

    settings = _personalization(page)
    settings.get_by_role("button", name="Remove").click()
    page.get_by_role("dialog", name="Delete Memory?").get_by_role("button", name="Confirm").click()
    expect(settings.get_by_text(EDITED)).to_have_count(0)

    settings = _personalization(page)
    expect(settings.get_by_text("Memories accessible by LLMs will be shown here.")).to_be_visible()
    expect(settings.get_by_text(EDITED)).to_have_count(0)


def _memory_switch(page: Page) -> Locator:
    return _personalization(page).locator("#tab-personalization").get_by_role("switch").first


def _account_remembering(make_user, memory: str):
    account = make_user()
    with account.client() as client:
        added = client.post("/api/v1/memories/add", json={"content": memory, "type": "user"})
    added.raise_for_status()
    return account


def _sent_for(page: Page, upstream, question: str) -> str:
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    page.goto("/")
    send(page, question)
    expect_reply(page, "Noted.")
    return json.dumps(next(filter(reply.answering(question), upstream.chat_requests())))


def test_switching_memory_off_keeps_memories_out_of_the_chat(page_for, make_user, upstream):
    page = page_for(_account_remembering(make_user, FIRST))
    switch = _memory_switch(page)
    expect(switch).to_have_attribute("aria-checked", "true")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "false")

    sent = _sent_for(page, upstream, "where are the bees?")
    assert FIRST not in sent, "a switched-off memory was sent to the model"

    page.reload()
    expect(_memory_switch(page)).to_have_attribute("aria-checked", "false")

    other = page_for(_account_remembering(make_user, FIRST))
    assert FIRST in _sent_for(other, upstream, "where are my bees?")
