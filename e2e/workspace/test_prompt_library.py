"""Journey: editing, versioning, switching off and sharing a prompt, as the chat's `/` menu sees it.

An edit saved in the prompt editor with Set as Production unticked adds a version to the
editor's history without going live: the `/` menu still inserts the old text until that version
is set as production from the history. A prompt switched off in the workspace list leaves the
`/` menu and returns when switched back on. A prompt shared with a group is offered to its
members, who see it read-only in the editor, while an account outside the group is only offered
what was made public.

Discriminates: passes on the 176d31d1d build. In a backend copy where saving a version ignores
`is_production`, the draft test goes red (the draft goes live at once); where the prompt list
ignores `is_active`, the switched-off prompt stays offered; where the prompt list skips the
read-grant check, the stranger is offered the group's prompt.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant, make_group
from harness.actors import Actor
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT_USER = {"workspace": {"prompts": True}}


@pytest.fixture
def librarian(make_user):
    """A fresh admin; `add(command, content)` saves a prompt of theirs and returns its id."""
    account = make_user(role="admin")
    created: list[str] = []

    def add(command: str, content: str) -> str:
        form = {"command": command, "name": f"Prompt {command}", "content": content}
        with account.client() as client:
            response = client.post("/api/v1/prompts/create", json=form)
        assert response.status_code == 200, response.text
        created.append(response.json()["id"])
        return created[-1]

    yield account, add
    with account.client() as client:
        for prompt_id in created:
            client.delete(f"/api/v1/prompts/id/{prompt_id}/delete")


def _share(owner: Actor, prompt_id: str, grants: list[dict]) -> None:
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/prompts/id/{prompt_id}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text


def _slash_menu(page: Page, typed: str) -> Locator:
    """Type `/typed` into a fresh chat; returns the prompt list it opens."""
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(f"/{typed}")
    return page.get_by_role("tooltip")


def _offered(menu: Locator, command: str) -> Locator:
    return menu.get_by_role("button", name=command)


def _inserted_text(page: Page, command: str) -> Locator:
    _offered(_slash_menu(page, command), command).click()
    return chat_input(page)


def test_an_edit_saved_as_a_draft_goes_live_only_when_set_as_production(page_for, librarian):
    account, add = librarian
    command = f"brief{uuid.uuid4().hex[:8]}"
    prompt_id = add(command, "Summarise this in three lines.")
    page = page_for(account)

    page.goto(f"/workspace/prompts/{prompt_id}")
    page.get_by_role("button", name="Edit", exact=True).click()
    editing = page.get_by_role("dialog").filter(has_text="Edit Prompt")
    editing.get_by_role("textbox", name=re.compile("^Write a summary in 50 words")).fill(
        "Summarise this in one line."
    )
    editing.get_by_role("textbox", name="Commit Message").fill("Shorter summary")
    editing.get_by_role("checkbox", name="Set as Production").uncheck()
    editing.get_by_role("button", name="Save", exact=True).click()
    draft = page.get_by_role("button").filter(has_text="Shorter summary")
    expect(draft).to_be_visible()
    expect(draft).not_to_contain_text("Live")

    expect(_inserted_text(page, command)).to_have_text("Summarise this in three lines.")

    page.goto(f"/workspace/prompts/{prompt_id}")
    draft.click()
    page.get_by_role("button", name="Set as Production", exact=True).click()
    expect(draft).to_contain_text("Live")

    expect(_inserted_text(page, command)).to_have_text("Summarise this in one line.")


def test_a_switched_off_prompt_leaves_the_slash_menu_until_switched_on(page_for, librarian):
    account, add = librarian
    prefix = f"memo{uuid.uuid4().hex[:6]}"
    kept, switched = f"{prefix}kept", f"{prefix}off"
    add(kept, "Write a memo.")
    add(switched, "Write a short memo.")
    page = page_for(account)

    def flip(expected: str) -> None:
        page.goto("/workspace/prompts")
        page.get_by_role("textbox", name="Search Prompts").fill(switched)
        row = page.get_by_role("main").get_by_role("button").filter(has_text=f"Prompt {switched}")
        row.get_by_role("switch").click()
        expect(row.get_by_role("switch")).to_have_attribute("aria-checked", expected)

    flip("false")
    menu = _slash_menu(page, prefix)
    expect(_offered(menu, kept)).to_be_visible()
    expect(_offered(menu, switched)).to_have_count(0)

    flip("true")
    menu = _slash_menu(page, prefix)
    expect(_offered(menu, kept)).to_be_visible()
    expect(_offered(menu, switched)).to_be_visible()


def test_a_prompt_shared_with_a_group_reaches_only_its_members(
    page_for, librarian, admin, make_user
):
    account, add = librarian
    member, stranger = make_user(), make_user()
    group_id = make_group(admin, [member], PROMPT_USER)
    prefix = f"team{uuid.uuid4().hex[:6]}"
    shared, public = f"{prefix}group", f"{prefix}all"
    shared_id = add(shared, "Draft the team update.")
    _share(account, shared_id, [grant("group", group_id, "read")])
    _share(account, add(public, "Draft the public update."), [grant("user", "*", "read")])

    member_page = page_for(member)
    expect(_inserted_text(member_page, shared)).to_have_text("Draft the team update.")
    member_page.goto(f"/workspace/prompts/{shared_id}")
    expect(member_page.get_by_text("Read Only", exact=True)).to_be_visible()
    expect(member_page.get_by_role("button", name="Edit", exact=True)).to_have_count(0)

    stranger_menu = _slash_menu(page_for(stranger), prefix)
    expect(_offered(stranger_menu, public)).to_be_visible()
    expect(_offered(stranger_menu, shared)).to_have_count(0)
