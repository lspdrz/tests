"""A chat sent under a temporary id is answered over the socket and never stored.

The web client marks a temporary chat by sending its completion with a `temporary:` chat id
made from its socket session id. The server streams the reply to that session only and creates
no chat, so the account's chat list and search stay empty while the same message under no id
makes a saved chat.

Twin of e2e/chat/test_temporary_chat.py.

Discriminates: passes on dev 176d31d1d, fails with `is_saved_chat_id` treating a `temporary:` id
as saved (the temporary chat is stored).
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.chat import send_message, wait_for_reply
from harness.socket_client import connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PROMPT = "Which lake is the deepest in Carinthia?"
ANSWER = "That is the Millstatter See."


def chats_about_carinthia(client) -> list[dict]:
    found = client.get("/api/v1/chats/search", params={"text": "Carinthia"})
    found.raise_for_status()
    return found.json()


def test_a_temporary_chat_is_answered_over_the_socket_and_not_stored(make_user, upstream):
    account = make_user()
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))
    with connected(account) as socket, account.client() as client:
        chat_id = f"temporary:{socket.client.sid}"

        send_message(client, PROMPT, chat_id=chat_id, session_id=socket.client.sid)

        streamed = socket.wait_for(chat_id, "chat:completion", done=True)
        assert ANSWER in str(streamed["data"]["output"])
        assert chats_about_carinthia(client) == []
        assert client.get("/api/v1/chats/").json() == []
        assert client.get(f"/api/v1/chats/{chat_id}").status_code != 200


def test_the_same_message_without_a_temporary_id_is_stored(make_user, upstream):
    account = make_user()
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))
    with account.client() as client:
        turn = send_message(client, PROMPT)
        wait_for_reply(client, turn)

        assert [chat["id"] for chat in chats_about_carinthia(client)] == [turn.chat_id]
