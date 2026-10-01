"""Regression: the page was never sent the start of a reply that a filter wrote.

Fix `1058444d7` (PR #31652, issue #31643). A filter that writes text into the reply before the
model answers (a `message` event in its inlet) has that text stored as the start of the reply's
first output item. The server streamed the model's words into that item by its id but never
sent the item itself to the page first, so the page put the model's words in place of the
filter's text until the reply ended, which closed the artifacts pane the filter's HTML block had
opened. The item, with the filter's text in it, now reaches the page before the model's first
words. Browser twin: e2e/chat/test_artifacts_pane_auto_open.py.

Discriminates: passes on dev 015dbc861; with `1058444d7` reverted in a backend copy the page's
socket gets the model's first words for an item it was never sent.
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.plugins import installed_function
from harness.socket_client import connected

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

FILTER_TEXT = "A preview first. "

PREVIEW_FILTER = f"""
class Filter:
    async def inlet(self, body: dict, __event_emitter__) -> dict:
        await __event_emitter__({{"type": "message", "data": {{"content": {FILTER_TEXT!r}}}}})
        return body
"""


def _item_texts(output: list[dict]) -> dict[str, str]:
    return {
        item["id"]: "".join(part.get("text", "") for part in item.get("content", []))
        for item in output
        if item.get("type") == "message"
    }


def test_the_filter_text_reaches_the_page_before_the_models_first_words(admin, make_user, upstream):
    account = make_user()
    prompt = "start the reply with a preview"
    upstream.queue(reply.text(["The model's ", "own words."], match=reply.answering(prompt)))
    with installed_function(admin, PREVIEW_FILTER, is_global=True), connected(account) as tab:
        with account.client() as client:
            turn, stored = ask(client, prompt)
        tab.wait_for(turn.chat_id, "chat:completion", done=True)

    assert stored["content"] == FILTER_TEXT + "The model's own words."
    events = [event for event in tab.events_of(turn.chat_id) if isinstance(event.get("data"), dict)]
    first_words_at = next(
        index
        for index, event in enumerate(events)
        if event["data"].get("type") == "response.output_text.delta"
    )
    items_sent: dict[str, str] = {}
    for event in events[:first_words_at]:
        if event.get("type") == "chat:completion" and "output" in event["data"]:
            items_sent.update(_item_texts(event["data"]["output"]))
    first_words = events[first_words_at]["data"]
    assert first_words["item_id"] in items_sent, (
        f"the model's first words went to an item the page was never sent (#31652): {first_words}"
    )
    assert items_sent[first_words["item_id"]] == FILTER_TEXT
