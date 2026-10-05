"""Journey: a user finds out what an MCP tool server offers before choosing it for a chat.

The chat's tool list asks `/api/v1/tools/id/server:mcp:<id>/specs` for the tools of one
connection and gets each tool's name and description; the connection's name filter applies, so a
filtered-out tool is not listed. A connection the person has no grant for, one an admin switched
off, an id that is no MCP connection and an unknown id all answer 404 Tool not found. A server
that cannot be reached answers 502 Unable to load tools, and a connection that needs sign-in
which the person has not done answers 401, the status the chat turns into "Auth required".
Nearby: a person with a grant sees the same tools as the admin who made the connection.

Discriminates: in a backend copy with the specs route removed, every test goes red (404 where the
tools should be listed); with the enable check dropped from the MCP connect, the switched-off
connection lists its tools.
"""

from __future__ import annotations

import secrets

import pytest

from harness.instance import free_port
from harness.mcp_server import ECHO_DESCRIPTION, TOOL_SERVERS, mcp_connection, serving_mcp
from harness.terminal_server import read_grant

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def _save(admin, preserve, *connections: dict) -> None:
    preserve(TOOL_SERVERS)
    with admin.client() as client:
        saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": list(connections)})
    assert saved.status_code == 200, saved.text


def _specs(person, server_id: str):
    with person.client() as client:
        return client.get(f"/api/v1/tools/id/server:mcp:{server_id}/specs")


@pytest.fixture
def server_id():
    return f"harbour_{secrets.token_hex(4)}"


def test_the_specs_list_each_tool_with_its_description(admin, make_user, preserve, server_id):
    person = make_user()
    with serving_mcp(media=True) as url:
        _save(admin, preserve, mcp_connection(url, server_id, [read_grant(person.id)]))
        answered = _specs(person, server_id)
    assert answered.status_code == 200, answered.text
    listed = {spec["name"]: spec["description"] for spec in answered.json()["specs"]}
    assert listed["echo"] == ECHO_DESCRIPTION
    assert {"snapshot", "chime"} <= set(listed), f"the media tools were not listed: {listed}"


def test_a_person_sees_the_tools_the_admin_sees(admin, make_user, preserve, server_id):
    person = make_user()
    with serving_mcp() as url:
        _save(admin, preserve, mcp_connection(url, server_id, [read_grant(person.id)]))
        for_admin = _specs(admin, server_id).json()
        for_person = _specs(person, server_id).json()
    assert for_person == for_admin
    assert [spec["name"] for spec in for_person["specs"]] == ["echo"]


def test_the_connections_name_filter_hides_the_other_tools(admin, make_user, preserve, server_id):
    person = make_user()
    with serving_mcp(media=True) as url:
        connection = mcp_connection(url, server_id, [read_grant(person.id)])
        connection["config"]["function_name_filter_list"] = "chime"
        _save(admin, preserve, connection)
        answered = _specs(person, server_id)
    assert [spec["name"] for spec in answered.json()["specs"]] == ["chime"]


def test_a_person_without_a_grant_is_told_the_tool_is_not_found(
    admin, make_user, preserve, server_id
):
    outsider = make_user()
    with serving_mcp() as url:
        _save(admin, preserve, mcp_connection(url, server_id, [read_grant(admin.id)]))
        answered = _specs(outsider, server_id)
    assert answered.status_code == 404, answered.text
    assert "echo" not in answered.text


def test_a_switched_off_connection_is_not_found(admin, make_user, preserve, server_id):
    person = make_user()
    with serving_mcp() as url:
        connection = mcp_connection(url, server_id, [read_grant(person.id)])
        connection["config"]["enable"] = False
        _save(admin, preserve, connection)
        answered = _specs(person, server_id)
    assert answered.status_code == 404, (
        f"a switched-off MCP connection still listed its tools: {answered.text}"
    )


@pytest.mark.parametrize(
    "tool_id", ["server:mcp:nowhere_at_all", "server:openapi:nowhere", "local"]
)
def test_an_id_that_is_no_mcp_connection_is_not_found(make_user, tool_id):
    with make_user().client() as client:
        answered = client.get(f"/api/v1/tools/id/{tool_id}/specs")
    assert answered.status_code == 404, answered.text


def test_a_server_that_cannot_be_reached_answers_bad_gateway(admin, make_user, preserve, server_id):
    person = make_user()
    url = f"http://127.0.0.1:{free_port()}/mcp"
    _save(admin, preserve, mcp_connection(url, server_id, [read_grant(person.id)]))
    answered = _specs(person, server_id)
    assert answered.status_code == 502, answered.text
    assert answered.json()["detail"] == "Unable to load tools"


def test_an_oauth_connection_without_a_sign_in_answers_unauthorized(
    admin, make_user, preserve, server_id
):
    person = make_user()
    with serving_mcp() as url:
        connection = mcp_connection(url, server_id, [read_grant(person.id)])
        connection["auth_type"] = "oauth_2.1"
        _save(admin, preserve, connection)
        answered = _specs(person, server_id)
    assert answered.status_code == 401, answered.text
