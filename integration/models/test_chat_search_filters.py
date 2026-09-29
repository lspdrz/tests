"""The chat search filters narrow a user's own chats and never reach another account's.

`GET /api/v1/chats/search` reads `tag:`, `folder:`, `pinned:`, `archived:` and `shared:` words
out of the query, the ones the search dialog suggests while typing, and matches the rest against
titles and message text. A filter keeps only the chats that carry the property; another account's
chat with the same tag, a folder of the same name or the same flag stays out. `folder:` also
reaches the chats in that folder's subfolders, `tag:none` lists the untagged chats and archived
chats are left out unless the query asks for them.

Twin of e2e/chat/test_search_modal.py.

Discriminates: passes on dev 176d31d1d. With the search no longer scoped to the caller the tag,
pinned, archived and shared cases fail (a folder name only ever resolves to the caller's own
folders); with the filters ignored every test fails.
"""

from __future__ import annotations

import uuid
from typing import Callable

import httpx
import pytest

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def unique_word() -> str:
    return f"wombat{uuid.uuid4().hex[:8]}"


def create_chat(client: httpx.Client, title: str, **import_fields) -> str:
    entry = {"chat": {"title": title}, **import_fields}
    imported = client.post("/api/v1/chats/import", json={"chats": [entry]})
    assert imported.status_code == 200, imported.text
    return imported.json()[0]["id"]


def post(client: httpx.Client, path: str, **body) -> dict:
    response = client.post(path, json=body or None)
    assert response.status_code == 200, response.text
    return response.json()


def search(client: httpx.Client, text: str) -> set[str]:
    response = client.get("/api/v1/chats/search", params={"text": text})
    assert response.status_code == 200, response.text
    return {chat["id"] for chat in response.json()}


def create_folder(client: httpx.Client, name: str, parent_id: str | None = None) -> str:
    return post(client, "/api/v1/folders/", name=name, parent_id=parent_id)["id"]


def tag(client: httpx.Client, chat_id: str) -> None:
    post(client, f"/api/v1/chats/{chat_id}/tags", name="Road Trip")


def file_in_folder(client: httpx.Client, chat_id: str) -> None:
    folder_id = create_folder(client, "Travel Plans")
    post(client, f"/api/v1/chats/{chat_id}/folder", folder_id=folder_id)


def pin(client: httpx.Client, chat_id: str) -> None:
    post(client, f"/api/v1/chats/{chat_id}/pin")


def archive(client: httpx.Client, chat_id: str) -> None:
    post(client, f"/api/v1/chats/{chat_id}/archive")


def share(client: httpx.Client, chat_id: str) -> None:
    post(client, f"/api/v1/chats/{chat_id}/share")


FILTERS: dict[str, tuple[str, Callable[[httpx.Client, str], None]]] = {
    "tag": ("tag:road_trip", tag),
    "folder": ("folder:travel_plans", file_in_folder),
    "pinned": ("pinned:true", pin),
    "archived": ("archived:true", archive),
    "shared": ("shared:true", share),
}


@pytest.mark.parametrize("name", FILTERS)
def test_a_filter_keeps_only_the_owners_chats_that_carry_it(make_user, name):
    query_filter, give_property = FILTERS[name]
    word = unique_word()
    with make_user().client() as other_client:
        theirs = create_chat(other_client, f"Theirs {word}")
        give_property(other_client, theirs)
    with make_user().client() as client:
        matching = create_chat(client, f"Matching {word}")
        give_property(client, matching)
        create_chat(client, f"Plain {word}")

        found_with_word = search(client, f"{word} {query_filter}")
        found_by_filter_alone = search(client, query_filter)

    assert found_with_word == {matching}, f"{query_filter} did not keep only the owner's chat"
    assert found_by_filter_alone == {matching}, (
        f"{query_filter} on its own listed another account's chat or an unfiltered one"
    )


def test_pinned_false_leaves_the_pinned_chats_out(make_user):
    word = unique_word()
    with make_user().client() as client:
        pinned = create_chat(client, f"Pinned {word}")
        pin(client, pinned)
        unpinned = create_chat(client, f"Unpinned {word}")

        assert search(client, f"{word} pinned:false") == {unpinned}


def test_archived_chats_are_left_out_unless_asked_for(make_user):
    word = unique_word()
    with make_user().client() as client:
        archived = create_chat(client, f"Archived {word}", archived=True)
        current = create_chat(client, f"Current {word}")

        assert search(client, word) == {current}
        assert search(client, f"{word} archived:false") == {current}
        assert search(client, f"{word} archived:true") == {archived}


def test_tag_none_lists_only_the_untagged_chats(make_user):
    word = unique_word()
    with make_user().client() as client:
        tagged = create_chat(client, f"Tagged {word}")
        tag(client, tagged)
        untagged = create_chat(client, f"Untagged {word}")

        assert search(client, f"{word} tag:none") == {untagged}


def test_a_folder_filter_reaches_its_subfolders_and_ignores_case(make_user):
    word = unique_word()
    with make_user().client() as client:
        parent_id = create_folder(client, "Travel Plans")
        child_id = create_folder(client, "Asia", parent_id=parent_id)
        in_parent = create_chat(client, f"Parent {word}", folder_id=parent_id)
        in_child = create_chat(client, f"Child {word}", folder_id=child_id)
        create_chat(client, f"Loose {word}")

        assert search(client, f"{word} folder:Travel_Plans") == {in_parent, in_child}
