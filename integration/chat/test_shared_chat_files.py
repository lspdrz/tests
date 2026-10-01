"""Regression: a shared chat's reader opens every file attached to it (PR #31650, issue #31648).

A message's attachments were recorded as the chat's files only when none of them was already
attached to the chat under another message. With one old file in the list, the new files were
dropped without a log line: the owner still saw them, but the reader of a shared copy was refused
when opening them. The owner attaches file A, then sends A with a new file B, shares the chat and
the reader opens B and A; a file the chat never held stays closed to the reader.

Discriminates: passes on dev 015dbc861. In a backend copy, `insert_chat_files` looking only at the
files of the same message (3d43a497b reverted) fails the first case.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.access import grant
from harness.chat import ask

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _upload(client: httpx.Client, text: str) -> dict:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (f"{uuid.uuid4().hex[:8]}.txt", text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return {"type": "file", "id": uploaded.json()["id"]}


def _share_with(client: httpx.Client, chat_id: str, reader) -> None:
    shared = client.post(f"/api/v1/chats/{chat_id}/share")
    assert shared.status_code == 200, shared.text
    granted = client.post(
        f"/api/v1/chats/shared/{chat_id}/access/update",
        json={"access_grants": [grant("user", reader.id, "read")]},
    )
    assert granted.status_code == 200, granted.text


def _reader_status(reader, file: dict) -> int:
    with reader.client() as client:
        return client.get(f"/api/v1/files/{file['id']}/data/content").status_code


def _ask(client: httpx.Client, upstream, question: str, **options):
    upstream.queue(reply.text(f"answer to {question}"))
    return ask(client, question, **options)[0]


def test_a_new_file_sent_with_one_from_an_earlier_message_opens_in_the_share(make_user, upstream):
    owner, reader = make_user(), make_user()
    with owner.client() as client:
        old, new = _upload(client, "the old file"), _upload(client, "the new file")
        first = _ask(client, upstream, "first question", message_files=[old])
        _ask(
            client,
            upstream,
            "second question",
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            message_files=[old, new],
        )
        _share_with(client, first.chat_id, reader)

    assert _reader_status(reader, new) == 200, (
        "the new file is closed to the share's reader (#31650)"
    )
    assert _reader_status(reader, old) == 200


def test_a_file_outside_the_chat_stays_closed_to_the_reader(make_user, upstream):
    owner, reader = make_user(), make_user()
    with owner.client() as client:
        attached, private = _upload(client, "attached"), _upload(client, "private")
        turn = _ask(client, upstream, "question", message_files=[attached])
        _share_with(client, turn.chat_id, reader)

    assert _reader_status(reader, attached) == 200
    assert _reader_status(reader, private) == 404
