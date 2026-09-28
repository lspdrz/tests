"""A note edit a tab sends as binary reaches the other tabs as a plain list of bytes.

0.11.2 89716ea88 (PR #28180): the socket server used python-socketio's default packet, which
walks every outgoing payload for `bytes` and turns an event holding any into a binary one. Open
WebUI never sends bytes itself, so the server now uses a JSON-only packet: binary events are
off, and binary attachments a client sends (a Yjs update as raw bytes) come back as lists of
ints, the form the note handlers store, apply and pass on.

Here one tab sends its edit as raw bytes; the other tab of the note gets it as a list, and a tab
that opens the note afterwards starts from the edited document.

Twin of unit/chat/test_socket_cluster_and_packets.py for the packet.

Discriminates: passes on dev ef67cc3fa; with the server built on python-socketio's default
packet again, the other tab receives the edit as a binary attachment (`bytes`).
"""

from __future__ import annotations

import uuid

import pytest

from harness.socket_client import connected, note_edit, note_text

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _create_note(owner) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": f"binary {uuid.uuid4().hex[:8]}", "data": {"content": {"md": ""}}},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _send_binary_edit(tab, note_id: str, text: str) -> None:
    update = {"document_id": f"note:{note_id}", "update": bytes(note_edit(text))}
    tab.call("ydoc:document:update", update)


def test_a_binary_edit_reaches_the_other_tab_as_a_plain_list(make_user):
    owner = make_user()
    note_id = _create_note(owner)

    with connected(owner) as sender, connected(owner) as receiver:
        sender.join_note(note_id)
        receiver.join_note(note_id)
        _send_binary_edit(sender, note_id, "sent as bytes")
        passed_on = receiver.note_update(note_id)["update"]

    assert isinstance(passed_on, list), (
        f"the edit was passed on as {type(passed_on).__name__}, a binary event, not plain JSON"
    )
    assert note_text(passed_on) == "<paragraph>sent as bytes</paragraph>"


def test_a_tab_opening_the_note_after_a_binary_edit_starts_from_it(make_user):
    owner = make_user()
    note_id = _create_note(owner)

    with connected(owner) as sender:
        sender.join_note(note_id)
        _send_binary_edit(sender, note_id, "stored from bytes")
        with connected(owner) as later:
            later.join_note(note_id)
            document = later.note_state(note_id)

    assert document == "<paragraph>stored from bytes</paragraph>"
