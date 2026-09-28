"""Journey: a chat streams its reply to the browser when websockets are switched off.

With `ENABLE_WEBSOCKET_SUPPORT=false`, for a proxy that cannot pass websockets, python-socketio
serves Socket.IO over HTTP long polling only and the web client connects that way. The reply to
a message reaches the page as socket events, so the chat works only if long polling carries
them. Twin, in the browser, of the long polling tests in integration/deps/test_transport_stack.py.

Discriminates: passes on the dev ef67cc3fa build; in a backend copy whose Socket.IO server keeps
the websocket transport whatever the setting, the page never connects and the reply never shows.
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.actors import create_user
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "are you there without a websocket?"


@pytest.fixture(scope="module")
def long_polling(instance_with):
    launched = instance_with({"ENABLE_WEBSOCKET_SUPPORT": "false"})
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return launched


def test_a_reply_streams_to_the_page_over_long_polling(page_for, long_polling):
    page = page_for(create_user(long_polling))
    socket_requests: list[str] = []
    page.on("request", lambda sent: socket_requests.append(sent.url))
    long_polling.upstream.queue(reply.text("here, polling", match=reply.answering(PROMPT)))

    send(page, PROMPT)

    expect_reply(page, "here, polling")
    socket_urls = [url for url in socket_requests if "/ws/socket.io/" in url]
    assert any("transport=polling" in url for url in socket_urls), socket_urls
    assert not any("transport=websocket" in url for url in socket_urls), socket_urls
