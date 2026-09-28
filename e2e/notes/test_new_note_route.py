"""Journey: opening the new-note address creates a note and lands in its editor.

Visiting `/notes/new` makes a note on the spot and replaces the address with the note's own. The
note is titled with today's date unless the address carries a title, and starts with the text of
the address's content when it has one. The saved note is listed on the Notes page and text typed
into its editor is still there after a reload.

Discriminates: passes on dev 176d31d1d; in a frontend copy, with the route ignoring the title and
content of its address the listing and prefilled tests fail, and with the route not creating a note
all four fail.
"""

from __future__ import annotations

import re
import time
from datetime import date

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NOTE_URL = re.compile(r"/notes/[0-9a-f-]+$")
TITLE = "Trip packing"
PREFILLED_TEXT = "passport and charger"
TYPED_TEXT = "and a rain jacket"


def _stored_notes(author: Actor) -> list[dict]:
    with author.client() as client:
        return client.get("/api/v1/notes/").json()


def _wait_until_stored(author: Actor, note_id: str, text: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    stored = ""
    with author.client() as client:
        while time.monotonic() < deadline:
            note = client.get(f"/api/v1/notes/{note_id}").json()
            stored = str((note.get("data") or {}).get("content"))
            if text in stored:
                return
            time.sleep(0.2)
    raise AssertionError(f"the note never stored {text!r}; it holds {stored}")


def _open_new_note(page: Page, query: str = "") -> str:
    page.goto(f"/notes/new{query}")
    expect(page).to_have_url(NOTE_URL)
    return page.url.rsplit("/", 1)[1]


def test_new_note_address_lands_in_an_editor_for_a_note_titled_with_today(page_for, make_user):
    author = make_user()
    page = page_for(author)
    note_id = _open_new_note(page)

    editor_page = page.get_by_role("main")
    expect(editor_page.get_by_role("textbox", name="Title")).to_have_value(date.today().isoformat())
    stored = _stored_notes(author)
    assert [note["id"] for note in stored] == [note_id]


def test_new_note_is_listed_on_the_notes_page(page_for, make_user):
    author = make_user()
    page = page_for(author)
    _open_new_note(page, f"?title={TITLE}")

    page.goto("/notes")
    card = page.get_by_role("main").get_by_role("button", name="Open note").filter(has_text=TITLE)
    expect(card).to_have_count(1)


def test_new_note_address_prefills_the_title_and_the_content(page_for, make_user):
    author = make_user()
    page = page_for(author)
    note_id = _open_new_note(page, f"?title={TITLE}&content={PREFILLED_TEXT.replace(' ', '%20')}")

    editor_page = page.get_by_role("main")
    expect(editor_page.get_by_role("textbox", name="Title")).to_have_value(TITLE)
    expect(editor_page.get_by_label("Write something...")).to_have_text(PREFILLED_TEXT)
    with author.client() as client:
        note = client.get(f"/api/v1/notes/{note_id}").json()
    assert note["title"] == TITLE
    assert note["data"]["content"]["md"] == PREFILLED_TEXT


def test_text_typed_into_a_new_note_survives_a_reload(page_for, make_user):
    author = make_user()
    page = page_for(author)
    note_id = _open_new_note(page)

    editor_page = page.get_by_role("main")
    editor_page.get_by_label("Write something...").click()
    page.keyboard.type(TYPED_TEXT)
    _wait_until_stored(author, note_id, TYPED_TEXT)

    page.reload()
    expect(editor_page.get_by_label("Write something...")).to_have_text(TYPED_TEXT)
