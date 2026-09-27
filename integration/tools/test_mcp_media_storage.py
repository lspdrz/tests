"""Regression: media from MCP tools was stored twice, once more as base64 in the database.

Issue open-webui/open-webui#30411, fix b6191a051 (PR open-webui/open-webui#30419). An image or
audio clip an MCP tool returns is saved as a file. The file's metadata also carried the whole
tool result item, base64 payload included, so every such result wrote the media into the
database a second time and the database kept growing. The metadata now holds only the chat,
message and session the file belongs to.

Discriminates: passes on dev efe63bd34, fails with b6191a051 reverted (the file record carries
the base64 payload).
"""

from __future__ import annotations

import base64
import json
import secrets

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.mcp_server import (
    CHIME_WAV,
    SNAPSHOT_PNG,
    TOOL_SERVERS,
    mcp_connection,
    serving_mcp,
)
from harness.terminal_server import read_grant

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def media_server(admin, make_user, preserve):
    """(person, server id): an MCP server with media tools that `person` may use."""
    preserve(TOOL_SERVERS)
    person = make_user()
    server_id = f"media_{secrets.token_hex(4)}"
    with serving_mcp(media=True) as url:
        connection = mcp_connection(url, server_id, [read_grant(person.id)])
        with admin.client() as client:
            saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
        assert saved.status_code == 200, saved.text
        yield person, server_id


def stored_media(person, upstream, server_id: str, tool: str) -> tuple[dict, dict]:
    """Have the model call `tool`; returns (the turn's ids, the one file record it stored)."""
    upstream.queue(reply.tool_call(f"{server_id}_{tool}", {}), reply.text("Got it."))
    with person.client() as client:
        turn, _ = ask(client, f"use {tool}", tool_ids=[f"server:mcp:{server_id}"])
        listed = client.get("/api/v1/files/")
    assert listed.status_code == 200, listed.text
    [record] = listed.json()["items"]
    return {"chat_id": turn.chat_id, "message_id": turn.assistant_message_id}, record


@pytest.mark.parametrize(
    "tool,payload", [("snapshot", SNAPSHOT_PNG), ("chime", CHIME_WAV)], ids=["image", "audio"]
)
def test_the_file_record_does_not_hold_the_media_again(media_server, upstream, tool, payload):
    person, server_id = media_server

    _, record = stored_media(person, upstream, server_id, tool)

    assert base64.b64encode(payload).decode() not in json.dumps(record), record["meta"]


def test_the_file_keeps_its_chat_and_message(media_server, upstream):
    person, server_id = media_server

    turn, record = stored_media(person, upstream, server_id, "snapshot")

    assert record["meta"]["data"] == {**turn, "session_id": record["meta"]["data"]["session_id"]}
    assert record["meta"]["content_type"] == "image/png"
    with person.client() as client:
        content = client.get(f"/api/v1/files/{record['id']}/content")
    assert content.content == SNAPSHOT_PNG
