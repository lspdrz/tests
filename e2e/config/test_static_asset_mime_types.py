"""Regression: the app did not load on a host whose MIME table types scripts as text/plain.

Commit `d8133c905` (PR #29139, issue #29133, open-webui 0.11.2). Windows registry entries can map
`.js`, `.mjs` and `.wasm` to `text/plain`, and the server served its files with whatever the host
said. A browser refuses to run a module script of that type, so the page stayed blank. The fix
registers the browser-safe types at import, over the host's. On an instance booted on such a
table (`harness.host_mime_types`) a user opens the app and chats.

The served types of every script and WebAssembly file are pinned in
integration/config/test_static_asset_mime_types.py.

Discriminates: passes on bbfa876af, fails with the three `mimetypes.add_type` calls removed from
`main.py` (the chat page never renders).
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.actors import create_user
from harness.host_mime_types import mistyped_host_env
from utils.chat_ui import expect_reply, send

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

QUESTION = "does the page load?"


@pytest.fixture(scope="module")
def mistyped_host(instance_with, tmp_path_factory):
    launched = instance_with(mistyped_host_env(tmp_path_factory.mktemp("mistyped-host")))
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return launched


def test_the_app_loads_and_chats_on_a_host_that_mistypes_scripts(page_for, mistyped_host):
    mistyped_host.upstream.queue(reply.text("It does.", match=reply.answering(QUESTION)))
    page = page_for(create_user(mistyped_host))

    send(page, QUESTION)

    expect_reply(page, "It does.")
