"""Regression: an image an MCP tool returned was shown in the chat but never sent to the model.

Issue open-webui/open-webui#30327, fix 6f6792d48 (PR open-webui/open-webui#30358). An MCP tool's
image is stored as a file and listed on the tool result for the chat to show. Only images that
arrived as data URIs were also added as image parts for the model, so the stored file URL of an
MCP image (a camera snapshot) reached the chat alone and the model answered without seeing it.
File-URL images now go to the model as well.

Discriminates: passes on dev efe63bd34, fails with 6f6792d48 reverted (the follow-up request
carries no image).
"""

from __future__ import annotations

import base64
import secrets

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.mcp_server import SNAPSHOT_PNG, TOOL_SERVERS, mcp_connection, serving_mcp
from harness.terminal_server import read_grant

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

SNAPSHOT_DATA_URL = f"data:image/png;base64,{base64.b64encode(SNAPSHOT_PNG).decode()}"


@pytest.fixture
def camera(admin, make_user, preserve):
    """(person, server id): an MCP server with media tools that `person` may use."""
    preserve(TOOL_SERVERS)
    person = make_user()
    server_id = f"camera_{secrets.token_hex(4)}"
    with serving_mcp(media=True) as url:
        connection = mcp_connection(url, server_id, [read_grant(person.id)])
        with admin.client() as client:
            saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
        assert saved.status_code == 200, saved.text
        yield person, server_id


def call_tool(person, upstream, server_id: str, tool: str, arguments: dict | None = None):
    """Have the model call one MCP tool; returns (stored reply, the follow-up request)."""
    upstream.queue(reply.tool_call(f"{server_id}_{tool}", arguments or {}), reply.text("I see."))
    with person.client() as client:
        _, message = ask(client, f"use {tool}", tool_ids=[f"server:mcp:{server_id}"])
    follow_up = upstream.chat_requests()[-1]["messages"]
    assert any(entry["role"] == "tool" for entry in follow_up), f"{tool} never ran: {follow_up}"
    return message, follow_up


def image_urls(messages: list[dict]) -> list[str]:
    return [
        part["image_url"]["url"]
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]


def tool_result_files(message: dict) -> list[dict]:
    return [
        file
        for item in message["output"]
        if item["type"] == "function_call_output"
        for file in item.get("files") or []
    ]


def test_an_mcp_tools_image_is_sent_to_the_model(camera, upstream):
    person, server_id = camera

    _, follow_up = call_tool(person, upstream, server_id, "snapshot")

    assert image_urls(follow_up) == [SNAPSHOT_DATA_URL], follow_up


def test_the_image_is_still_shown_in_the_tool_call(camera, upstream):
    person, server_id = camera

    message, _ = call_tool(person, upstream, server_id, "snapshot")

    [shown] = tool_result_files(message)
    assert shown["type"] == "image"
    with person.client() as client:
        content = client.get(shown["url"])
    assert content.status_code == 200, content.text
    assert content.content == SNAPSHOT_PNG


def test_audio_from_an_mcp_tool_is_not_sent_as_an_image(camera, upstream):
    person, server_id = camera

    message, follow_up = call_tool(person, upstream, server_id, "chime")

    assert image_urls(follow_up) == []
    assert [file["type"] for file in tool_result_files(message)] == ["audio"]


def test_a_text_result_sends_no_image(camera, upstream):
    person, server_id = camera

    _, follow_up = call_tool(person, upstream, server_id, "echo", {"text": "hello camera"})

    assert image_urls(follow_up) == []
    [tool_message] = [entry for entry in follow_up if entry["role"] == "tool"]
    assert "hello camera" in tool_message["content"]
