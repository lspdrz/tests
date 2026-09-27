"""Filters as the chat page shows their work: one switched off mid-reply, and an outlet edit.

1. Fix commit `7d694570a` in `utils/filter.py`: the active filters are read once per request and
   kept on it, where every filter stage of a chat turn used to read them from the database
   again. So a filter an admin switches off mid-reply still runs its `stream` stage to the end
   of that reply. Here a second filter's inlet switches it off, which makes the moment exact.
2. Unpinned, no issue filed: an `outlet` filter that edits the reply's `content`, as the filter
   docs' examples do, has its edit stored in `content` while the reply's `output` items keep the
   unedited text, and the chat page renders a reply from its `output`. So the edit never shows,
   live or after a reload. Red until the outlet edit reaches what the page renders.

Twin of integration/chat/test_filter_context_and_handlers.py (its mid-request cases).

Discriminates: the first test passes on dev ef67cc3fa and fails with 7d694570a reverted in a
backend copy (the reply arrives without the stream mark); the second fails on dev ef67cc3fa.
"""

from __future__ import annotations

import re
import time

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.plugins import installed_function
from utils.chat_ui import REPLY_TIMEOUT_MS, expect_reply, last_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

SWITCHING_FILTER = """
import httpx


class Filter:
    async def inlet(self, body):
        async with httpx.AsyncClient(base_url={base_url!r}) as client:
            response = await client.post(
                "/api/v1/functions/id/{target}/toggle",
                headers={{"Authorization": "Bearer {token}"}},
            )
            response.raise_for_status()
        return body
"""

LATE_STREAM_FILTER = """
class Filter:
    def stream(self, event):
        for choice in event.get("choices", []):
            if choice.get("delta", {}).get("content"):
                choice["delta"]["content"] += " [late stream]"
        return event
"""

# the filter docs' "highlight outputs" example
HIGHLIGHTING_OUTLET = """
class Filter:
    async def outlet(self, body):
        for message in body["messages"]:
            if message["role"] == "assistant":
                message["content"] = f"{message['content']} [highlighted]"
        return body
"""


def test_the_reply_keeps_the_filter_switched_off_during_it(
    page_for, instance, admin, make_user, upstream
):
    prompt = "tell me about filters"
    upstream.queue(reply.text("filtered reply", match=reply.answering(prompt)))
    page = page_for(make_user())
    with installed_function(admin, LATE_STREAM_FILTER, is_global=True) as late_filter:
        switching = SWITCHING_FILTER.format(
            base_url=instance.base_url, target=late_filter, token=admin.token
        )
        with installed_function(admin, switching, is_global=True):
            send(page, prompt)
            expect_reply(page, "filtered reply [late stream]")


def stored_reply_content(person, chat_id: str, expected: str) -> str:
    """The stored reply once the outlet stage has rewritten it, or as it stands at the deadline."""
    deadline = time.monotonic() + 10
    with person.client() as client:
        while True:
            history = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]
            content = history["messages"][history["currentId"]]["content"]
            if content == expected or time.monotonic() > deadline:
                return content
            time.sleep(0.2)


def test_an_outlet_edit_to_the_reply_shows_on_the_page(page_for, admin, make_user, upstream):
    prompt = "say something plain"
    upstream.queue(reply.text("plain answer", match=reply.answering(prompt)))
    person = make_user()
    page = page_for(person)
    with installed_function(admin, HIGHLIGHTING_OUTLET, is_global=True):
        send(page, prompt)
        expect_reply(page, "plain answer")
        expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))
        chat_id = page.url.rsplit("/", 1)[-1]
        stored = stored_reply_content(person, chat_id, "plain answer [highlighted]")
        assert stored == "plain answer [highlighted]"
        page.reload()

        # the edit is stored in content; the page renders the reply's unedited output items
        expect(last_reply(page)).to_contain_text(
            "plain answer [highlighted]", timeout=REPLY_TIMEOUT_MS
        )
