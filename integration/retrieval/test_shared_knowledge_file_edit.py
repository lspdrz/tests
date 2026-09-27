"""Regression: an edit to a file of a shared knowledge base, made with write access, was dropped.

Fix a046e0704 (open-webui/open-webui#30448, issue open-webui/open-webui#30319). The file editor
saves through `/api/v1/files/{id}/data/content/update`, which lets a writer of the knowledge base
through, but the processing step behind it only looked the file up among the caller's own files
unless the caller was an admin. For a writer that found nothing, so the route answered success
and the old content stayed, in the file and in the knowledge base's search. The step now accepts
the file when the caller may write to it.

Discriminates: passes on dev efe63bd34, fails with a046e0704 reverted (the writer's edit answers
200 and the stored content is still the old text).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.access import grant
from harness.knowledge_bases import add_text_file, knowledge_base

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

OLD_TEXT = "The ferry leaves the harbour at nine."
NEW_TEXT = "The ferry now leaves the harbour at eleven."


def stored_text(client: httpx.Client, file_id: str) -> str:
    read = client.get(f"/api/v1/files/{file_id}/data/content")
    assert read.status_code == 200, read.text
    return read.json()["content"]


def edit(client: httpx.Client, file_id: str, text: str) -> httpx.Response:
    return client.post(f"/api/v1/files/{file_id}/data/content/update", json={"content": text})


def searched_text(client: httpx.Client, knowledge_id: str) -> str:
    found = client.post(
        "/api/v1/retrieval/query/collection",
        json={"collection_names": [knowledge_id], "query": "ferry", "k": 10},
    )
    assert found.status_code == 200, found.text
    return " ".join(found.json()["documents"][0])


@pytest.fixture
def shared_base(admin, make_user):
    """A knowledge base holding one file, shared with a writer and a reader."""
    writer, reader = make_user(), make_user()
    # the access editor grants read alongside write
    grants = [
        grant("user", writer.id, "read"),
        grant("user", writer.id, "write"),
        grant("user", reader.id, "read"),
    ]
    with admin.client() as client:
        name = f"Timetables {uuid.uuid4().hex[:6]}"
        with knowledge_base(client, name, access_grants=grants) as knowledge_id:
            file_id = add_text_file(client, knowledge_id, "timetable.txt", OLD_TEXT)
            yield {"id": knowledge_id, "file": file_id, "writer": writer, "reader": reader}


def test_a_writers_edit_to_a_shared_file_is_saved(shared_base, admin):
    file_id = shared_base["file"]
    with shared_base["writer"].client() as client:
        edited = edit(client, file_id, NEW_TEXT)
        assert edited.status_code == 200, edited.text
        assert stored_text(client, file_id) == NEW_TEXT, (
            "the writer's edit was answered as saved but the file kept its old text (#30319)"
        )
    with admin.client() as client:
        assert stored_text(client, file_id) == NEW_TEXT


def test_a_writers_edit_reaches_the_knowledge_base_search(shared_base, admin):
    with shared_base["writer"].client() as client:
        assert edit(client, shared_base["file"], NEW_TEXT).status_code == 200

    with admin.client() as client:
        found = searched_text(client, shared_base["id"])
    assert "eleven" in found and "at nine" not in found, found


def test_a_reader_cannot_edit_the_shared_file(shared_base, admin):
    with shared_base["reader"].client() as client:
        refused = edit(client, shared_base["file"], NEW_TEXT)
    assert refused.status_code in (401, 403, 404), refused.text
    with admin.client() as client:
        assert stored_text(client, shared_base["file"]) == OLD_TEXT


def test_a_stranger_cannot_edit_the_shared_file(shared_base, admin, make_user):
    with make_user().client() as client:
        refused = edit(client, shared_base["file"], NEW_TEXT)
    assert refused.status_code in (401, 403, 404), refused.text
    with admin.client() as client:
        assert stored_text(client, shared_base["file"]) == OLD_TEXT


def test_the_owner_still_edits_their_own_file(shared_base, admin):
    with admin.client() as client:
        assert edit(client, shared_base["file"], NEW_TEXT).status_code == 200
        assert stored_text(client, shared_base["file"]) == NEW_TEXT
