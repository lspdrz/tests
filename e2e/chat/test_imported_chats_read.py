"""Imported chats showed as unread in the sidebar, open-webui/open-webui#30750.

Fix commit `95d6af44d` (PR open-webui/open-webui#30754). An imported chat was stored without a
read time, so the sidebar marked it unread with its blue dot the moment it arrived. The dot has
no text or role of its own, so it is found by its colour class.

Twin of integration/models/test_imported_chats_read.py.

Discriminates: passes on dev efe63bd34, fails with `95d6af44d` reverted in a backend copy (the
imported chat carries the unread dot).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

LONG_AGO = 1_600_000_000
MESSAGE = {"id": "m1", "parentId": None, "childrenIds": [], "role": "user", "content": "hi"}


def import_chat(client, title: str) -> str:
    chat = {"title": title, "history": {"currentId": "m1", "messages": {"m1": MESSAGE}}}
    form = {"chat": chat, "created_at": LONG_AGO, "updated_at": LONG_AGO}
    imported = client.post("/api/v1/chats/import", json={"chats": [form]})
    assert imported.status_code == 200, imported.text
    return imported.json()[0]["id"]


def open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    return page.get_by_role("navigation", name="Chat history")


def sidebar_entry(sidebar: Locator, title: str) -> Locator:
    entry = sidebar.locator("#sidebar-chat-group").filter(has_text=title)
    expect(entry).to_be_visible()
    return entry


def unread_dot(entry: Locator) -> Locator:
    return entry.locator(".bg-sky-500.rounded-full")


def test_an_imported_chat_arrives_without_the_unread_dot(make_user, page_for):
    account = make_user()
    imported_title, unread_title = f"Imported {uuid.uuid4().hex[:6]}", "Marked unread"
    with account.client() as client:
        import_chat(client, imported_title)
        marked_id = import_chat(client, unread_title)
        assert client.post(f"/api/v1/chats/{marked_id}/unread").is_success
    sidebar = open_sidebar(page_for(account))

    # the chat marked unread shows the dot, so its absence below is meaningful
    expect(unread_dot(sidebar_entry(sidebar, unread_title))).to_be_visible()
    expect(unread_dot(sidebar_entry(sidebar, imported_title))).to_have_count(0)
