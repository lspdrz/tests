"""Journey: a long chat in the browser on the Prompt Caching page's setup only appends.

The Prompt Caching docs page promises that with its cache-optimal setup the request Open WebUI
sends to the provider keeps its start: every request repeats the tool list and every message of
the one before it byte for byte and only adds to the end. Its integration twin checks each
feature family over HTTP; here a person drives one chat through the web client: a file attached
in the chat input and searched with the file tools, a reply with reasoning, the model's knowledge
searched, a native tool call, a page reload, the chat continued from a second browser and a
while passing between turns. Every consecutive pair of the provider's requests is checked.

Discriminates: passes on dev 176d31d1d; in backend copies, a clock value added to the model's
system prompt and the tool list shuffled per request each turn it red.
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.prompt_caching import (
    assert_append_only,
    cache_optimal_model,
    turn_off_memory_system_context,
)
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NOTES = ("berths.txt", "Berth 3 is reserved for the pilot boat.\nBerth 5 is free on Mondays.\n")


@pytest.fixture
def cached_setup(admin, preserve):
    """The page's setup: its model, and the memory system context switched off."""
    preserve("admin_config")
    turn_off_memory_system_context(admin)
    with cache_optimal_model(admin) as model:
        yield model


def attach(page: Page, name: str, text: str) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="More", exact=True).last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    chooser.value.set_files({"name": name, "mimeType": "text/plain", "buffer": text.encode()})


def ask(page: Page, upstream, prompt: str, *replies: reply.Reply) -> None:
    """Send `prompt` and wait for the last scripted reply's text to show."""
    for scripted in replies:
        scripted.match = reply.answering(prompt)
        upstream.queue(scripted)
    send(page, prompt)
    expect_reply(page, replies[-1].content)


def calling(name: str, arguments: dict, call_id: str) -> reply.Reply:
    return reply.tool_call(name, arguments, call_id=call_id)


def test_a_long_chat_in_the_browser_only_appends(page_for, cached_setup, make_user, upstream):
    person = make_user()
    page = page_for(person)
    page.goto(f"/?models={cached_setup.id}")

    attach(page, *NOTES)
    ask(
        page,
        upstream,
        "which berth is free?",
        calling("query_chat_files", {"query": "free berth"}, "call_files"),
        reply.text("Berth 5 is free (berths.txt)."),
    )
    ask(
        page,
        upstream,
        "and for the pilot boat?",
        reply.text("Berth 3, as the file says.", reasoning="the file names berth 3"),
    )
    ask(
        page,
        upstream,
        "when does the ferry leave?",
        calling("query_knowledge_files", {"query": "ferry"}, "call_knowledge"),
        reply.text("At 06:40 (handbook.txt)."),
    )
    chat_url = page.url

    page.reload()
    ask(page, upstream, "is that every day?", reply.text("Every day, the handbook says."))

    second = page_for(person)
    second.goto(chat_url)
    expect(second.get_by_text("Every day, the handbook says.")).to_be_visible()
    time.sleep(1.5)  # a clock value in the prompt would move on between the turns
    ask(
        second,
        upstream,
        "what time is it now?",
        calling("get_current_timestamp", {}, "call_clock"),
        reply.text("Just after six."),
    )
    ask(second, upstream, "thanks, that is all", reply.text("Safe travels."))

    requests = [body for body in upstream.chat_requests() if body.get("stream")]
    assert len(requests) == 9, f"expected nine requests to the provider, got {len(requests)}"
    tool_results = "\n".join(
        str(entry["content"]) for entry in requests[-1]["messages"] if entry["role"] == "tool"
    )
    assert "Berth 5 is free" in tool_results and "06:40" in tool_results, tool_results
    assert "<attached_files>" in str(requests[0]["messages"][1]["content"])
    assert_append_only(requests)
