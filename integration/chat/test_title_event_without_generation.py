"""With title generation off, the live title event carried the reply, open-webui/open-webui#31348.

Fix commit `fccd75568` (PR open-webui/open-webui#31355). With title generation turned off the
chat is saved with its first message as the title, but the `chat:title` event sent to the open
tab carried the assistant message instead. A new chat kept showing "New Chat" in the header and
tab until a reload, and a note's chat panel showed the whole model reply as its title. The
event now carries the title that was saved.

Discriminates: passes on dev `efe63bd34`, fails with `fccd75568` reverted (the event carries the
still-empty assistant message, not the saved title).
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.chat import send_message, wait_for_reply
from harness.socket_client import connected

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PROMPT = "What is the capital of Austria?"
ANSWER = "Vienna is the capital of Austria."


def title_events(socket, chat_id: str) -> list:
    return [event["data"] for event in socket.events_of(chat_id) if event["type"] == "chat:title"]


def test_the_title_event_carries_the_saved_title(make_user, upstream):
    account = make_user()
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))
    with connected(account) as socket, account.client() as client:
        turn = send_message(client, PROMPT)
        wait_for_reply(client, turn)
        socket.wait_for(turn.chat_id, "chat:title")
        stored_title = client.get(f"/api/v1/chats/{turn.chat_id}").json()["title"]
        sent_titles = title_events(socket, turn.chat_id)

    assert stored_title == PROMPT
    assert sent_titles == [PROMPT], f"the open tab was told the title is {sent_titles}"


def test_a_long_first_message_is_titled_the_same_live_and_stored(make_user, upstream):
    account = make_user()
    prompt = "Please summarise this. " + "word " * 40
    upstream.queue(reply.text(ANSWER, match=reply.answering(prompt)))
    with connected(account) as socket, account.client() as client:
        turn = send_message(client, prompt)
        wait_for_reply(client, turn)
        socket.wait_for(turn.chat_id, "chat:title")
        stored_title = client.get(f"/api/v1/chats/{turn.chat_id}").json()["title"]
        sent_titles = title_events(socket, turn.chat_id)

    assert ANSWER not in sent_titles
    assert sent_titles == [stored_title]


def test_a_second_turn_keeps_the_title_and_sends_no_new_one(make_user, upstream):
    account = make_user()
    follow_up = "And of Germany?"
    upstream.queue(
        reply.text(ANSWER, match=reply.answering(PROMPT)),
        reply.text("Berlin.", match=reply.answering(follow_up)),
    )
    with connected(account) as socket, account.client() as client:
        first = send_message(client, PROMPT)
        first_reply = wait_for_reply(client, first)
        socket.wait_for(first.chat_id, "chat:title")
        history = [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": ANSWER}]
        second = send_message(
            client,
            follow_up,
            chat_id=first.chat_id,
            parent_id=first_reply["id"],
            history=history,
        )
        wait_for_reply(client, second)
        stored_title = client.get(f"/api/v1/chats/{first.chat_id}").json()["title"]
        sent_titles = title_events(socket, first.chat_id)

    assert stored_title == PROMPT
    assert sent_titles == [PROMPT]
