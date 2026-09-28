"""A chat opened while its reply is still streaming shows the reply so far.

0.11.0 `aadab2f`: `create_task` minted its own task id while the reply saved its progress under
the id `chat_completion` had stamped into the turn. Opening the chat looks its running tasks up
by the registered id, so it found no progress: the page showed an empty reply that only filled
in with the pieces streamed after it opened. `create_task` now registers the caller's id.

The reply is sent over the API and streams for ten seconds; a browser opens the chat after the
first pieces and must show them while the last one is still to come.

Twin of unit/chat/test_socket_and_redis_runtime.py for the task id.

Discriminates: passes on dev ef67cc3fa; with `create_task` ignoring the id it is handed, the
opened chat never shows the first piece before the reply ends.
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import expect

from harness.chat import wait_for_reply
from harness.inflight import LAST_PIECE, start_slow_reply
from utils.chat_ui import last_reply

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def test_a_chat_opened_mid_stream_shows_the_reply_so_far(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    with account.client() as client:
        turn = start_slow_reply(client, upstream, chunk_delay=0.5)
        time.sleep(1.5)

        page.goto(f"/c/{turn.chat_id}")
        reply = last_reply(page)
        expect(reply).to_contain_text("part-0", timeout=5000)
        shown_mid_stream = reply.inner_text()

        wait_for_reply(client, turn)

    assert LAST_PIECE not in shown_mid_stream, "the reply had already finished; nothing was shown"
