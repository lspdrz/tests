"""Regression: a new turn ran the approved tool calls of the stored message it happened to name.

Fix commit `9f680bb80` (PR #29142), `drain_approved_tool_calls`: the drain that runs tool calls a
user approved gated on the turn's `message_id`, which every ordinary send carries, so each new
turn read its assistant message back from the chat before answering. It now requires
`assistant_message_id`, which only a resume or continue payload sends. A send from the web
client names a fresh message, so the saved read shows nothing there; an API client that posts a
new turn under the id of a stored message holding approved calls sees them run instead of a
fresh answer, and in ask mode sees the turn paused for an approval it never asked about. (What
the reply does afterwards with the stored output it inherits is not pinned.)

Twin of unit/chat/test_tool_approval_drain_lookup.py.

Discriminates: passes on dev ef67cc3fa; with `message_id` accepted in place of
`assistant_message_id` again, the approved call runs before the provider is asked about the new
turn, and the ask-mode turn pauses without calling the provider.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ChatTurn
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
QUEUED_CALL_ID = "call_left_queued"


def _queued_call(approved: bool) -> dict:
    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": QUEUED_CALL_ID,
        "name": "get_current_timestamp",
        "arguments": "{}",
        "status": "queued",
    }
    return {**call, "approved": True} if approved else call


def _chat_with_a_queued_call(client, approved: bool) -> tuple[str, str, str]:
    """A chat whose earlier branch ends on a message holding a queued tool call.

    Returns the chat id, the last answered message and the id of the message with the call.
    """
    held_id = str(uuid.uuid4())
    chat_id, _ = seed_chat(
        client,
        [
            {"role": "user", "content": "what time is it?"},
            {"role": "assistant", "content": "Let me look."},
            {"role": "user", "content": "and now?"},
            {"id": held_id, "role": "assistant", "content": "", "output": [_queued_call(approved)]},
        ],
    )
    history = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
    answered_id = history[history[held_id]["parentId"]]["parentId"]
    return chat_id, answered_id, held_id


def _new_turn_named(client, chat_id: str, parent_id: str, message_id: str, content: str, **extra):
    """A new turn, sent the way the web client sends one, whose reply id is a stored message."""
    user_message_id = str(uuid.uuid4())
    user_message = {
        "id": user_message_id,
        "parentId": parent_id,
        "childrenIds": [message_id],
        "role": "user",
        "content": content,
        "models": [MOCK_MODEL_ID],
        "timestamp": int(time.time()),
    }
    accepted = client.post(
        "/api/chat/completions",
        json={
            "model": MOCK_MODEL_ID,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "chat_id": chat_id,
            "parent_id": parent_id,
            "id": message_id,
            "user_message": user_message,
            "session_id": f"harness-{uuid.uuid4().hex[:8]}",
            **extra,
        },
    )
    assert accepted.status_code == 200, accepted.text
    return ChatTurn(chat_id, user_message_id, message_id)


def _tool_results(sent: dict) -> list[str]:
    return [entry.get("tool_call_id") for entry in sent["messages"] if entry["role"] == "tool"]


def _first_request_answering(upstream, prompt: str, timeout: float = 20.0) -> dict | None:
    """The first provider request for the new turn, or None when none came in time."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for sent in upstream.chat_requests():
            if reply.answering(prompt)(sent):
                return sent
        time.sleep(0.2)
    return None


def test_a_new_turn_does_not_run_calls_approved_on_the_message_it_names(make_user, upstream):
    prompt = "tell me a joke instead"
    upstream.queue(reply.text("A fresh answer.", match=reply.answering(prompt)))
    with make_user().client() as client:
        chat_id, answered_id, held_id = _chat_with_a_queued_call(client, approved=True)
        _new_turn_named(client, chat_id, answered_id, held_id, prompt)

    first = _first_request_answering(upstream, prompt)
    assert first is not None, "the new turn never reached the provider"
    assert QUEUED_CALL_ID not in _tool_results(first), (
        "a new turn first ran a tool call approved on the stored message it named"
    )


@pytest.fixture
def tool_approval_on(admin, preserve) -> None:
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        client.post(
            CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_PERMISSIONS": True}
        ).raise_for_status()


def test_a_new_turn_in_ask_mode_is_answered_not_paused(tool_approval_on, make_user, upstream):
    prompt = "never mind the time"
    upstream.queue(reply.text("Sure, skipping it.", match=reply.answering(prompt)))
    with make_user().client() as client:
        chat_id, answered_id, held_id = _chat_with_a_queued_call(client, approved=False)
        _new_turn_named(
            client, chat_id, answered_id, held_id, prompt, params={"tool_approval_mode": "ask"}
        )

    assert _first_request_answering(upstream, prompt) is not None, (
        "the new turn was paused for the approval of a call it never made"
    )
