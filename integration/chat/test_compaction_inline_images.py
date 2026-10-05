"""Regression: an image stored inside a chat counted as text and set off context compaction.

Fix 95dd3321a (PR #31915, issue #31913). Images kept in the chat record itself (chats from older
versions, images that could not be saved as files, temporary chats) sit in a message's `files` as
a base64 data URI. The token estimate counted that data as text, a quarter token per character,
so one picture of a few hundred kilobytes counted as about a hundred thousand tokens: the next
turn summarized the older messages of a chat far under the limit, and the reported context usage
jumped. The encoded data is now left out of the estimate; the rest of the file entry still counts.

Discriminates: passes on dev b859124f9, fails with 95dd3321a reverted (the reported usage is over
the threshold and the next turn asks for a summary).
"""

from __future__ import annotations

import base64
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.chat_history import seed_chat

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
THRESHOLD = 20_000
# about 400 KB of base64, a hundred thousand tokens if counted as text
INLINE_IMAGE = "data:image/png;base64," + base64.b64encode(bytes(300_000)).decode()


@pytest.fixture
def compaction(admin, preserve):
    """Compaction on, at a threshold far above the chat's text and far below the image's data."""
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        updated = client.post(
            CHAT_CONFIG[1],
            json={
                **current,
                "ENABLE_CONTEXT_COMPACTION": True,
                "CONTEXT_COMPACTION_TOKEN_THRESHOLD": THRESHOLD,
                "CONTEXT_COMPACTION_TOKEN_CAP": THRESHOLD,
            },
        )
    assert updated.status_code == 200, updated.text


def is_summary_request(body: dict) -> bool:
    return not body.get("stream")


def chat_with_files(client: httpx.Client, files: list[dict]) -> tuple[str, str]:
    """Two short turns, the first question carrying `files`."""
    return seed_chat(
        client,
        [
            {"role": "user", "content": "what is in this picture?", "files": files},
            {"role": "assistant", "content": "a blank square"},
            {"role": "user", "content": "are you sure?"},
            {"role": "assistant", "content": "quite sure"},
        ],
    )


def inline_image() -> dict:
    return {"type": "image", "id": str(uuid.uuid4()), "url": INLINE_IMAGE}


def context_tokens(client: httpx.Client, chat_id: str) -> int:
    return client.get(f"/api/v1/chats/{chat_id}").json()["context_usage"]["tokens"]


def test_an_inline_image_does_not_count_its_data_as_tokens(make_user, compaction):
    with make_user().client() as client:
        chat_id, _ = chat_with_files(client, [inline_image()])
        tokens = context_tokens(client, chat_id)

    assert tokens < 1_000, f"the image's data counted as {tokens} tokens"


def test_an_inline_image_does_not_set_off_compaction(make_user, upstream, compaction):
    upstream.queue(
        reply.text("SUMMARY", match=is_summary_request),
        reply.text("still a blank square", match=reply.answering("look again")),
    )
    with make_user().client() as client:
        chat_id, last_id = chat_with_files(client, [inline_image()])
        _, answer = ask(client, "look again", chat_id=chat_id, parent_id=last_id)

    assert [body for body in upstream.chat_requests() if is_summary_request(body)] == []
    assert answer["content"] == "still a blank square"


# ---------------------------------------------------------------- nearby


def test_the_rest_of_a_file_entry_still_counts(make_user, compaction):
    long_name = "holiday photo " * 400
    with make_user().client() as client:
        plain_chat, _ = chat_with_files(client, [])
        named_chat, _ = chat_with_files(client, [{**inline_image(), "name": long_name}])
        plain, named = context_tokens(client, plain_chat), context_tokens(client, named_chat)

    assert named - plain >= len(long_name) // 4
