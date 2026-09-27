"""Imported and cloned chats showed as unread, open-webui/open-webui#30750.

Fix commit `95d6af44d` (PR open-webui/open-webui#30754). Import and clone both store the new
chat without a read time, and the sidebar counts a chat unread when it has none or was updated
after it. So every imported chat, and every clone, arrived bold in the sidebar and raised its
folder's unread badge although the user had just made it. Both paths now stamp the chat read.

Discriminates: passes on dev `efe63bd34`, fails with `95d6af44d` reverted (the imported and the
cloned chat carry no read time and count toward their folder's unread badge).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

LONG_AGO = 1_600_000_000


def create_folder(client: httpx.Client) -> str:
    created = client.post("/api/v1/folders/", json={"name": f"Folder {uuid.uuid4().hex[:6]}"})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def sidebar_entry(client: httpx.Client, chat_id: str) -> dict:
    listed = client.get("/api/v1/chats/", params={"include_folders": True, "include_pinned": True})
    assert listed.status_code == 200, listed.text
    return {chat["id"]: chat for chat in listed.json()}[chat_id]


def is_unread(chat: dict) -> bool:
    # the sidebar's rule for drawing a chat bold
    return chat["last_read_at"] is None or chat["updated_at"] > chat["last_read_at"]


def folder_unread_count(client: httpx.Client, folder_id: str) -> int:
    folders = client.get("/api/v1/folders/")
    assert folders.status_code == 200, folders.text
    return {folder["id"]: folder["unread_count"] for folder in folders.json()}[folder_id]


MESSAGE = {"id": "m1", "parentId": None, "childrenIds": [], "role": "user", "content": "hi"}
HISTORY = {"currentId": "m1", "messages": {"m1": MESSAGE}}


def import_chat(client: httpx.Client, folder_id: str | None = None, **times) -> str:
    chat = {"title": "Imported", "history": HISTORY}
    imported = client.post(
        "/api/v1/chats/import", json={"chats": [{"chat": chat, "folder_id": folder_id, **times}]}
    )
    assert imported.status_code == 200, imported.text
    return imported.json()[0]["id"]


@pytest.fixture
def owner(make_user):
    return make_user()


# --------------------------------------------------------------------------- narrow


def test_an_imported_chat_is_read_and_leaves_its_folder_badge_alone(owner):
    with owner.client() as client:
        folder_id = create_folder(client)
        chat_id = import_chat(client, folder_id, created_at=LONG_AGO, updated_at=LONG_AGO)

        chat = sidebar_entry(client, chat_id)
        badge = folder_unread_count(client, folder_id)

    assert chat["last_read_at"] is not None, "the imported chat has no read time"
    assert not is_unread(chat), f"the imported chat shows as unread: {chat}"
    assert badge == 0, "the imported chat counts toward its folder's unread badge"


def test_a_cloned_chat_is_read_and_leaves_its_folder_badge_alone(owner):
    with owner.client() as client:
        folder_id = create_folder(client)
        created = client.post(
            "/api/v1/chats/new",
            json={"chat": {"title": "Mine", "history": HISTORY}, "folder_id": folder_id},
        )
        assert created.status_code == 200, created.text

        cloned = client.post(f"/api/v1/chats/{created.json()['id']}/clone", json={})
        assert cloned.status_code == 200, cloned.text
        clone = sidebar_entry(client, cloned.json()["id"])
        badge = folder_unread_count(client, folder_id)

    assert cloned.json()["folder_id"] == folder_id
    assert not is_unread(clone), f"the clone shows as unread: {clone}"
    assert badge == 0, "the clone counts toward its folder's unread badge"


# --------------------------------------------------------------------------- broad


def test_every_chat_of_a_bulk_import_is_read(owner):
    forms = [
        {"chat": {"title": f"Batch {index}", "history": HISTORY}, "updated_at": LONG_AGO + index}
        for index in range(3)
    ]
    with owner.client() as client:
        imported = client.post("/api/v1/chats/import", json={"chats": forms})
        assert imported.status_code == 200, imported.text
        chats = [sidebar_entry(client, chat["id"]) for chat in imported.json()]

    assert [chat for chat in chats if is_unread(chat)] == []


# --------------------------------------------------------------------------- nearby


def test_an_imported_chat_can_still_be_marked_unread(owner):
    with owner.client() as client:
        folder_id = create_folder(client)
        chat_id = import_chat(client, folder_id)

        marked = client.post(f"/api/v1/chats/{chat_id}/unread")

        assert marked.status_code == 200, marked.text
        assert marked.json()["folder_unread_counts"][folder_id] == 1
        assert is_unread(sidebar_entry(client, chat_id))


def test_an_imported_chat_keeps_the_times_it_was_given(owner):
    with owner.client() as client:
        chat_id = import_chat(client, created_at=LONG_AGO, updated_at=LONG_AGO + 60)
        chat = sidebar_entry(client, chat_id)
    assert chat["updated_at"] == LONG_AGO + 60
    assert chat["created_at"] == LONG_AGO
