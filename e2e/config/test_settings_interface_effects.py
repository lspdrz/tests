"""Journey: Interface switches in Settings change what the chat shows and does.

Each switch is flipped in the Settings dialog by a fresh account, which then sees its effect in a
chat, still sees it after a reload, and a second account in a browser of its own does not.
Chat Bubble UI off labels the user's own message "You" and Display the Username puts the
account's name there instead; Rich Text Input off leaves typed markdown as plain characters; Copy
Formatted Text puts HTML on the clipboard next to the plain text; Title Auto-Generation off names
a new chat after its first message where the model would otherwise title it.

Discriminates: passes on dev 176d31d1d; in a frontend copy, reading `chatBubble` as always on
turns the bubble tests red, passing `richText` as always on turns the rich text test red, copying
a reply without the `copyFormatted` setting turns the copy test red, and sending
`title_generation` as always on turns the title test red.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ANSWER = "The **lamp** is lit."
TITLE_TASK = "Generate a concise title"


def interface_switch(page: Page, name: str) -> Locator:
    page.goto("/?settings=interface")
    switch = page.locator("#tab-interface").get_by_role("switch", name=name, exact=True)
    expect(switch).to_be_visible()
    return switch


def flip(page: Page, name: str) -> None:
    switch = interface_switch(page, name)
    checked = switch.get_attribute("aria-checked")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        switch.click()
    expect(switch).not_to_have_attribute("aria-checked", checked)


def chat_once(page: Page, upstream, question: str) -> None:
    upstream.queue(reply.text(ANSWER, match=reply.answering(question)))
    page.goto("/")
    send(page, question)
    expect_reply(page, "is lit")


def own_message_label(page: Page, text: str) -> Locator:
    return conversation(page).locator(".user-message").get_by_text(text, exact=True)


def test_chat_bubble_off_labels_the_users_message_you(page_for, make_user, upstream):
    page = page_for(make_user())
    flip(page, "Chat Bubble UI")

    chat_once(page, upstream, "Is the lamp lit?")
    expect(own_message_label(page, "You")).to_be_visible()

    page.reload()
    expect(own_message_label(page, "You")).to_be_visible()

    other = page_for(make_user())
    chat_once(other, upstream, "Is the lamp lit yet?")
    expect(own_message_label(other, "You")).to_have_count(0)


def test_display_the_username_puts_the_accounts_name_on_its_message(page_for, make_user, upstream):
    account = make_user(name="Keeper Brannigan")
    page = page_for(account)
    flip(page, "Chat Bubble UI")
    flip(page, "Display the Username Instead of You in the Chat")

    chat_once(page, upstream, "Who is on watch?")
    expect(own_message_label(page, "Keeper Brannigan")).to_be_visible()

    page.reload()
    expect(own_message_label(page, "Keeper Brannigan")).to_be_visible()
    expect(
        interface_switch(page, "Display the Username Instead of You in the Chat")
    ).to_have_attribute("aria-checked", "true")

    other = page_for(make_user(name="Keeper Other"))
    flip(other, "Chat Bubble UI")
    chat_once(other, upstream, "Who is on watch tonight?")
    expect(own_message_label(other, "You")).to_be_visible()


def type_markdown(page: Page) -> Locator:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type("**bold** words")
    return chat_input(page)


def test_rich_text_input_off_leaves_markdown_as_typed(page_for, make_user):
    page = page_for(make_user())
    flip(page, "Rich Text Input for Chat")

    composer = type_markdown(page)
    expect(composer).to_contain_text("**bold** words")
    expect(composer.locator("strong")).to_have_count(0)

    page.reload()
    composer = type_markdown(page)
    expect(composer).to_contain_text("**bold** words")

    other = page_for(make_user())
    composer = type_markdown(other)
    expect(composer.locator("strong")).to_have_text("bold")


def copy_reply(page: Page) -> tuple[str, str | None]:
    """The clipboard's types after the reply's Copy button, and its HTML (None when it has none)."""
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    last_reply(page).hover()
    conversation(page).get_by_role("button", name="Copy").last.click()
    expect(page.get_by_text("Copying to clipboard was successful!")).to_be_visible()
    return page.evaluate(
        """async () => {
            const [item] = await navigator.clipboard.read();
            const html = item.types.includes('text/html')
                ? await (await item.getType('text/html')).text()
                : null;
            return [item.types.join(','), html];
        }"""
    )


def test_copy_formatted_text_puts_html_on_the_clipboard(page_for, make_user, upstream):
    page = page_for(make_user())
    flip(page, "Copy Formatted Text")

    chat_once(page, upstream, "Is the lamp lit tonight?")
    types, html = copy_reply(page)
    assert "text/html" in types
    assert "<strong>lamp</strong>" in html

    page.reload()
    expect(interface_switch(page, "Copy Formatted Text")).to_have_attribute("aria-checked", "true")

    other = page_for(make_user())
    chat_once(other, upstream, "Is the lamp lit this evening?")
    types, html = copy_reply(other)
    assert "text/html" not in types
    assert html is None


@pytest.fixture
def title_generation_on(admin, preserve):
    preserve("tasks")
    with admin.client() as client:
        current = client.get("/api/v1/tasks/config").json()
        saved = client.post(
            "/api/v1/tasks/config/update", json={**current, "ENABLE_TITLE_GENERATION": True}
        )
    saved.raise_for_status()


def open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    return page.get_by_role("navigation", name="Chat history")


def title_requests(upstream) -> list[dict]:
    return [body for body in upstream.chat_requests() if reply.answering(TITLE_TASK)(body)]


def offer_a_title(upstream) -> None:
    # queued first, so a title request that also quotes the question cannot take the chat's reply
    upstream.queue(reply.text('{"title": "Ferry Timetable"}', match=reply.answering(TITLE_TASK)))


def test_title_auto_generation_off_names_the_chat_after_its_first_message(
    page_for, make_user, upstream, title_generation_on
):
    page = page_for(make_user())
    flip(page, "Title Auto-Generation")

    question = "When does the ferry leave?"
    offer_a_title(upstream)
    chat_once(page, upstream, question)
    expect(open_sidebar(page).get_by_text(question)).to_be_visible()
    assert title_requests(upstream) == []

    page.reload()
    expect(interface_switch(page, "Title Auto-Generation")).to_have_attribute(
        "aria-checked", "false"
    )

    other = page_for(make_user())
    offer_a_title(upstream)
    chat_once(other, upstream, "When does the last ferry leave?")
    expect(open_sidebar(other).get_by_text("Ferry Timetable")).to_be_visible()
