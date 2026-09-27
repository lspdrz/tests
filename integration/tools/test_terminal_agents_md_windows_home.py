"""Regression: the AGENTS.md in the home folder of a terminal running on Windows was skipped.

Fix `3f5881c52` (open-webui/open-webui#31342, issue open-webui/open-webui#31340): before each turn
on a terminal, Open WebUI asks it for its home folder and hands the AGENTS.md found there to the
model. The home was only accepted as a POSIX absolute path, so a drive-letter or UNC home such
as `C:\\ProgramData\\OpenTerminal\\inst` was treated as invalid and the file was never read.
Windows absolute paths are accepted now; relative and drive-relative homes are still skipped.

Discriminates: passes on dev efe63bd34; with 3f5881c52 reverted in a backend copy every Windows
home case fails (the terminal is never asked for the file). The POSIX and skipped-home cases
pass on both.
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import json_answer
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, serving_terminal

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

AGENTS_MD = "Answer in British English."
OPERATION = {"operationId": "run_command", "responses": {"200": {"description": "ok"}}}
SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "terminal", "version": "1"},
    "paths": {"/execute": {"post": OPERATION}},
}


@pytest.fixture
def terminal(admin, preserve):
    """A terminal the admin connected; yields the fake server and the connection's id."""
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/files/read", json_answer({"content": AGENTS_MD}))
        connection = server.connection()
        with admin.client() as client:
            configure_terminals(client, connection)
        yield server, connection["id"]


def _model_is_sent_agents_md(terminal, make_user, upstream, home: str) -> bool:
    server, terminal_id = terminal
    server.route("GET", "/files/cwd", json_answer({"cwd": home, "home": home}))
    upstream.queue(reply.text("ready"))
    with make_user(role="admin").client() as client:
        _, message = ask(client, "tidy up the docs", terminal_id=terminal_id)
    assert message["content"] == "ready", message
    sent = [entry["content"] for entry in upstream.chat_requests()[-1]["messages"]]
    return f"# AGENTS.md\n\n{AGENTS_MD}" in sent


@pytest.mark.parametrize(
    "home",
    [
        "C:\\ProgramData\\OpenTerminal\\inst",
        "C:/Users/ada",
        "D:\\",
        "\\\\fileserver\\homes\\ada",
    ],
    ids=["drive-letter", "forward-slashes", "drive-root", "unc"],
)
def test_a_windows_home_hands_its_agents_md_to_the_model(terminal, make_user, upstream, home):
    assert _model_is_sent_agents_md(terminal, make_user, upstream, home), (
        f"the AGENTS.md in the Windows home {home!r} never reached the model (#31340)"
    )


def test_a_posix_home_still_hands_its_agents_md_to_the_model(terminal, make_user, upstream):
    assert _model_is_sent_agents_md(terminal, make_user, upstream, "/home/ada")


@pytest.mark.parametrize(
    "home",
    ["Users\\ada", "C:Users\\ada", "home/ada"],
    ids=["relative", "drive-relative", "posix-relative"],
)
def test_a_home_that_is_not_absolute_is_skipped(terminal, make_user, upstream, home):
    server, _ = terminal

    assert not _model_is_sent_agents_md(terminal, make_user, upstream, home)
    assert server.requests_to("/files/read") == [], "a file was read from a relative home"
