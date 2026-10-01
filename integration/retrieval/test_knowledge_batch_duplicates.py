"""Regression: batch add to a knowledge base refuses duplicate content (PR #31336, issue #31333).

`POST /api/v1/knowledge/{id}/files/batch/add` linked a file whose extracted text was already in the
knowledge base under another file, where the single add answers "Duplicate content detected", so the
same passages were embedded twice. The batch now runs the single add's check, also against the
earlier files of the same call, and reports a duplicate as a failed file while the rest still go
through. Chunks it writes carry the content hash, so a later single add sees them too.

Discriminates: passes on dev 015dbc861. In a backend copy, `process_files_batch` without the hash
check and without the hash in the chunk metadata (ecd0ff67e reverted) fails every duplicate case
here (both files linked); the re-add and distinct-content cases pass either way. The case that
reads the refusal in the batch answer is red on dev on purpose: `warnings` is built by the route
but dropped by its response model (as before the fix), so the failed file is never reported.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.knowledge_bases import knowledge_base

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _upload(client: httpx.Client, text: str) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (f"{uuid.uuid4().hex[:8]}.txt", text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def _text() -> str:
    return f"quarterly harbour report {uuid.uuid4().hex}"


def _batch_add(client: httpx.Client, knowledge_id: str, *file_ids: str) -> dict:
    added = client.post(
        f"/api/v1/knowledge/{knowledge_id}/files/batch/add",
        json=[{"file_id": file_id} for file_id in file_ids],
    )
    assert added.status_code == 200, added.text
    return added.json()


def _add(client: httpx.Client, knowledge_id: str, file_id: str) -> httpx.Response:
    return client.post(f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id})


def _linked(client: httpx.Client, knowledge_id: str) -> list[str]:
    listed = client.get(f"/api/v1/knowledge/{knowledge_id}/files")
    assert listed.status_code == 200, listed.text
    return [item["id"] for item in listed.json()["items"]]


@pytest.fixture
def base(admin):
    with admin.client() as client, knowledge_base(client) as knowledge_id:
        yield client, knowledge_id


def test_a_batch_refuses_content_already_in_the_knowledge_base(base):
    client, knowledge_id = base
    text = _text()
    first, copy = _upload(client, text), _upload(client, text)
    assert _add(client, knowledge_id, first).status_code == 200

    _batch_add(client, knowledge_id, copy)

    assert _linked(client, knowledge_id) == [first], "the batch linked duplicate content (#31336)"


def test_a_batch_links_one_of_two_files_with_the_same_text(base):
    client, knowledge_id = base
    text = _text()
    first, copy = _upload(client, text), _upload(client, text)

    _batch_add(client, knowledge_id, first, copy)

    # the batch is read in database order, so either file may be the one kept
    assert len(_linked(client, knowledge_id)) == 1


def test_a_batch_names_the_refused_duplicate_in_its_answer(base):
    client, knowledge_id = base
    text = _text()
    first, copy = _upload(client, text), _upload(client, text)

    result = _batch_add(client, knowledge_id, first, copy)

    # the route builds `warnings` for failed files but its response model has no such field
    errors = str(result.get("warnings", {}).get("errors"))
    assert "Duplicate content" in errors, (
        f"the batch answer drops the warnings naming the refused file: {result}"
    )


def test_a_batch_sibling_of_a_duplicate_still_goes_through(base):
    client, knowledge_id = base
    text = _text()
    first, copy, other = _upload(client, text), _upload(client, text), _upload(client, _text())

    _batch_add(client, knowledge_id, first, copy, other)

    linked = _linked(client, knowledge_id)
    assert other in linked and len(linked) == 2


def test_a_single_add_refuses_content_a_batch_added(base):
    client, knowledge_id = base
    text = _text()
    first, copy = _upload(client, text), _upload(client, text)
    _batch_add(client, knowledge_id, first)

    refused = _add(client, knowledge_id, copy)

    assert refused.status_code == 400, refused.text
    assert "Duplicate content" in refused.text
    assert _linked(client, knowledge_id) == [first]


def test_a_batch_accepts_the_same_file_again_and_files_with_distinct_text(base):
    client, knowledge_id = base
    first, second = _upload(client, _text()), _upload(client, _text())
    _batch_add(client, knowledge_id, first)

    again = _batch_add(client, knowledge_id, first, second)

    assert sorted(_linked(client, knowledge_id)) == sorted([first, second])
    assert not again.get("warnings")
