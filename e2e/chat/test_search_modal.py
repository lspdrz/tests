"""The chat search dialog finds a user's own chats, previews them and opens the one they pick.

The dialog opens from the sidebar's Search entry or the search shortcut. Typing sends the query to
`GET /api/v1/chats/search`, which matches chat titles and message text and understands `tag:`,
`folder:`, `pinned:` and `archived:` filters, offered as suggestions while typing; a match in the
text shows as a highlighted snippet. Hovering or arrowing onto a result previews its conversation
and a click opens it. "Start a new conversation" sends the typed text as the first message of a
new chat. Another account's matching chat never appears.

Two tests pin fixed bugs: Enter on a highlighted chat closed the dialog without opening the chat
(#31003, fixed by PR #31004), and a new conversation started from the dialog dropped the text after
an `&` because the query went into the page address unencoded (#31469, fixed by PR #31592).

Twin of integration/models/test_chat_search_filters.py.

Discriminates: passes on the a5bc78300 build; the two tests above fail on the 176d31d1d build,
before their fixes. On a build whose sidebar Search entries open nothing, whose snippet never
highlights, whose result rows ignore the pointer, whose ArrowUp stays put, whose empty result list
shows no text and whose new-conversation action drops the query, one test each goes red; on a
backend whose search matches neither titles nor message text, whose filters are ignored or that
searches every account's chats, the title, snippet, filter and other-account tests go red.
"""

from __future__ import annotations

import re
import uuid

import httpx
import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LONG_AGO = 1_700_000_000


def unique_word() -> str:
    return f"quokka{uuid.uuid4().hex[:8]}"


def conversation(question: str, answer: str) -> dict:
    asked = {"id": "q", "parentId": None, "childrenIds": ["a"], "role": "user"}
    answered = {"id": "a", "parentId": "q", "childrenIds": [], "role": "assistant"}
    messages = {"q": {**asked, "content": question}, "a": {**answered, "content": answer}}
    return {"currentId": "a", "messages": messages}


def import_chat(
    client: httpx.Client,
    title: str,
    *,
    question: str = "hello",
    answer: str = "hi there",
    updated_at: int = LONG_AGO,
    **import_fields,
) -> str:
    """A stored chat with one question and answer, last changed at `updated_at`."""
    chat = {"title": title, "models": [], "history": conversation(question, answer)}
    entry = {"chat": chat, "created_at": updated_at, "updated_at": updated_at, **import_fields}
    imported = client.post("/api/v1/chats/import", json={"chats": [entry]})
    assert imported.status_code == 200, imported.text
    return imported.json()[0]["id"]


def post(client: httpx.Client, path: str, **body) -> dict:
    response = client.post(path, json=body or None)
    assert response.status_code == 200, response.text
    return response.json()


def ready(page: Page) -> None:
    # the shortcut and the sidebar are live once the chat input renders
    expect(chat_input(page)).to_be_visible()


def search_box(dialog: Locator) -> Locator:
    return dialog.get_by_placeholder("Search")


def open_from_sidebar(page: Page) -> Locator:
    ready(page)
    page.get_by_role("button", name="Search", exact=True).first.click()
    dialog = page.get_by_role("dialog")
    expect(search_box(dialog)).to_be_visible()
    return dialog


def open_from_shortcut(page: Page) -> Locator:
    ready(page)
    page.keyboard.press("Control+K")
    dialog = page.get_by_role("dialog")
    expect(search_box(dialog)).to_be_visible()
    return dialog


def results(dialog: Locator) -> Locator:
    return dialog.get_by_role("link")


def result(dialog: Locator, title: str) -> Locator:
    return results(dialog).filter(has_text=title)


def type_query(dialog: Locator, text: str) -> None:
    """Types the way a person does, so the input's key handling runs as well."""
    search_box(dialog).click()
    search_box(dialog).press_sequentially(text)


def test_the_sidebar_entry_finds_a_chat_by_its_title_and_opens_it(page_for, make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        chat_id = import_chat(client, f"Trip {word}", answer="Pack a raincoat.")
        import_chat(client, "Weekly report")

    page = page_for(account)
    dialog = open_from_sidebar(page)
    search_box(dialog).fill(word)

    expect(result(dialog, f"Trip {word}")).to_be_visible()
    expect(results(dialog)).to_have_count(1)
    result(dialog, f"Trip {word}").click()

    expect(page).to_have_url(re.compile(f"/c/{chat_id}$"))
    expect(dialog).to_be_hidden()
    expect(page.get_by_text("Pack a raincoat.")).to_be_visible()


def test_the_shortcut_finds_a_chat_by_its_message_text_and_highlights_the_match(
    page_for, make_user
):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        import_chat(client, "Untitled trip", question=f"Where does the {word} live?")

    page = page_for(account)
    dialog = open_from_shortcut(page)
    search_box(dialog).fill(word)

    found = result(dialog, "Untitled trip")
    expect(found).to_be_visible()
    expect(found).to_contain_text(f"Where does the {word} live?")
    expect(found.get_by_role("mark")).to_have_text(word)


def test_hovering_a_result_previews_its_conversation(page_for, make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        import_chat(client, f"Recipes {word}", answer="Knead the dough for ten minutes.")

    page = page_for(account)
    dialog = open_from_shortcut(page)
    search_box(dialog).fill(word)
    expect(dialog.get_by_text("Select a conversation to preview")).to_be_visible()

    result(dialog, f"Recipes {word}").hover()

    expect(dialog.get_by_text("Knead the dough for ten minutes.")).to_be_visible()
    expect(dialog.get_by_text("Select a conversation to preview")).to_be_hidden()


@pytest.fixture
def two_matching_chats(make_user) -> tuple:
    """An account with two chats matching one word: the newer lists first."""
    account = make_user()
    word = unique_word()
    with account.client() as client:
        older_id = import_chat(client, f"Older {word}", answer="The older answer.")
        newer_id = import_chat(
            client, f"Newer {word}", answer="The newer answer.", updated_at=LONG_AGO + 100
        )
    return account, word, older_id, newer_id


def test_arrow_keys_move_the_preview_from_result_to_result(page_for, two_matching_chats):
    account, word, _older_id, _newer_id = two_matching_chats
    page = page_for(account)
    dialog = open_from_shortcut(page)
    type_query(dialog, word)
    expect(results(dialog)).to_have_count(2)

    # the selection stops at the last result, however far past it the arrow goes
    for _ in range(6):
        page.keyboard.press("ArrowDown")
    expect(dialog.get_by_text("The older answer.")).to_be_visible()

    page.keyboard.press("ArrowUp")
    expect(dialog.get_by_text("The newer answer.")).to_be_visible()
    expect(dialog.get_by_text("The older answer.")).to_be_hidden()


@pytest.mark.regression
def test_enter_opens_the_highlighted_chat(page_for, two_matching_chats):
    account, word, _older_id, newer_id = two_matching_chats
    page = page_for(account)
    dialog = open_from_shortcut(page)
    type_query(dialog, word)
    expect(results(dialog)).to_have_count(2)
    for _ in range(6):
        page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowUp")
    expect(dialog.get_by_text("The newer answer.")).to_be_visible()

    page.keyboard.press("Enter")

    expect(page, "Enter closed the dialog without opening the chat (#31003)").to_have_url(
        re.compile(f"/c/{newer_id}$")
    )


def test_a_query_nothing_matches_says_no_results(page_for, make_user):
    account = make_user()
    with account.client() as client:
        import_chat(client, "Weekly report")

    page = page_for(account)
    dialog = open_from_shortcut(page)
    expect(result(dialog, "Weekly report")).to_be_visible()
    search_box(dialog).fill(unique_word())

    expect(dialog.get_by_text("No results found")).to_be_visible()
    expect(results(dialog)).to_have_count(0)


def test_another_accounts_matching_chat_never_shows(page_for, make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        import_chat(client, f"Mine {word}")
    with make_user().client() as other_client:
        import_chat(other_client, f"Theirs {word}", question=f"about {word}")

    page = page_for(account)
    dialog = open_from_shortcut(page)
    search_box(dialog).fill(word)

    expect(result(dialog, f"Mine {word}")).to_be_visible()
    expect(results(dialog)).to_have_count(1)


def test_the_tag_suggestion_narrows_the_results_to_that_tag(page_for, make_user):
    account = make_user()
    word = unique_word()
    tag = f"Road {word}"
    with account.client() as client:
        tagged_id = import_chat(client, f"Tagged {word}")
        import_chat(client, f"Plain {word}")
        post(client, f"/api/v1/chats/{tagged_id}/tags", name=tag)

    page = page_for(account)
    dialog = open_from_shortcut(page)
    type_query(dialog, "tag:")
    dialog.get_by_role("button").filter(has_text=tag).click()

    expect(search_box(dialog)).to_have_value(f"tag:road_{word} ")
    expect(result(dialog, f"Tagged {word}")).to_be_visible()
    expect(results(dialog)).to_have_count(1)


def test_the_folder_suggestion_narrows_the_results_to_that_folder(page_for, make_user):
    account = make_user()
    word = unique_word()
    folder = f"Travel {word}"
    with account.client() as client:
        folder_id = post(client, "/api/v1/folders/", name=folder)["id"]
        import_chat(client, f"Filed {word}", folder_id=folder_id)
        import_chat(client, f"Loose {word}")

    page = page_for(account)
    dialog = open_from_shortcut(page)
    type_query(dialog, "folder:")
    dialog.get_by_role("button").filter(has_text=folder).click()

    expect(search_box(dialog)).to_have_value(f"folder:travel_{word} ")
    expect(result(dialog, f"Filed {word}")).to_be_visible()
    expect(results(dialog)).to_have_count(1)


def test_pinned_true_lists_only_the_pinned_chats(page_for, make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        import_chat(client, f"Pinned {word}", pinned=True)
        import_chat(client, f"Unpinned {word}")

    page = page_for(account)
    dialog = open_from_shortcut(page)
    search_box(dialog).fill(f"{word} pinned:true")

    expect(result(dialog, f"Pinned {word}")).to_be_visible()
    expect(results(dialog)).to_have_count(1)


def test_archived_chats_show_only_under_archived_true(page_for, make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        import_chat(client, f"Archived {word}", archived=True)
        import_chat(client, f"Current {word}")

    page = page_for(account)
    dialog = open_from_shortcut(page)
    search_box(dialog).fill(word)
    expect(result(dialog, f"Current {word}")).to_be_visible()
    expect(results(dialog)).to_have_count(1)

    search_box(dialog).fill(f"{word} archived:true")

    expect(result(dialog, f"Archived {word}")).to_be_visible()
    expect(results(dialog)).to_have_count(1)


def start_a_new_conversation(page: Page, text: str) -> None:
    dialog = open_from_shortcut(page)
    search_box(dialog).fill(text)
    dialog.get_by_role("button", name="Start a new conversation").click()


def test_start_a_new_conversation_sends_the_typed_text(page_for, make_user, upstream):
    question = f"How far can a {unique_word()} hop?"
    upstream.queue(reply.text("About two metres.", match=reply.answering(question)))
    page = page_for(make_user())

    start_a_new_conversation(page, question)

    expect_reply(page, "About two metres.")
    expect(page.get_by_label("Chat Conversation").get_by_text(question)).to_be_visible()


@pytest.mark.regression
def test_start_a_new_conversation_keeps_an_ampersand(page_for, make_user, upstream):
    question = f"Q&A about the {unique_word()}"
    upstream.queue(reply.text("Ask away.", match=reply.answering("Q")))
    page = page_for(make_user())

    start_a_new_conversation(page, question)

    expect_reply(page, "Ask away.")
    expect(
        page.get_by_label("Chat Conversation").get_by_text(question),
        "the text after the & was dropped from the new chat's first message (#31469)",
    ).to_be_visible()
