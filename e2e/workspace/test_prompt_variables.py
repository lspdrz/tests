"""Journey: a prompt with input variables asks for them in a form before it is sent.

A prompt's content can carry typed variables (`{{topic | text:required}}`, a `select` with its
options, a `number`, a `date`, a `checkbox`). Picking the prompt from the chat's `/` menu opens
the Input Variables form with one field of the matching kind per variable; saving the form puts
the answers in place of the placeholders, and that filled text is what the model is sent. A
field marked `required` holds the form open until it has a value, and a field left alone keeps
its `default`.

Discriminates: passes on the 176d31d1d build. In a copy of that build where the form's select and
date fields fall back to the plain text area, where a bare flag such as `required` is dropped
when a variable is parsed and where the form starts every field empty in place of its default,
each test goes red; the chat journeys pass on that copy.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

EVERY_KIND = (
    'Plan {{topic | text:placeholder="Topic"}} for'
    ' {{audience | select:options=["Kids","Adults"]:placeholder="Audience"}},'
    ' {{count | number:placeholder="How many"}} stops,'
    ' on {{day | date:placeholder="Day"}}, notes {{notes | checkbox:label="Add notes"}}.'
)


@pytest.fixture
def prompt_with(make_user):
    """A fresh admin; `prompt_with(content)` saves a prompt of theirs and returns its command."""
    account = make_user(role="admin")

    def save(content: str) -> str:
        command = f"plan{uuid.uuid4().hex[:8]}"
        form = {"command": command, "name": f"Plan {command}", "content": content}
        with account.client() as client:
            created = client.post("/api/v1/prompts/create", json=form)
        assert created.status_code == 200, created.text
        return command

    return account, save


def _pick_prompt(page: Page, command: str) -> Locator:
    """Pick `/command` in the chat input; returns the variables form it opens."""
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(f"/{command}")
    page.get_by_role("tooltip").get_by_role("button", name=command).click()
    form = page.get_by_role("dialog").filter(has_text="Input Variables")
    expect(form).to_be_visible()
    return form


def _send_and_read(page: Page, upstream, expected: str) -> str:
    upstream.queue(reply.text("Planned.", match=reply.answering(expected)))
    page.keyboard.press("Enter")
    expect_reply(page, "Planned.")
    return next(filter(reply.answering(expected), upstream.chat_requests()))["messages"][-1][
        "content"
    ]


def test_each_kind_of_variable_is_asked_for_and_the_filled_text_is_sent(
    page_for, prompt_with, upstream
):
    account, save = prompt_with
    command = save(EVERY_KIND)
    page = page_for(account)

    form = _pick_prompt(page, command)
    form.get_by_placeholder("Topic").fill("a harbour walk")
    form.get_by_role("combobox").select_option("Adults")
    form.get_by_placeholder("How many").fill("4")
    form.get_by_placeholder("Day").fill("2026-10-03")
    form.get_by_role("checkbox", name="Add notes").check()
    form.get_by_role("button", name="Save", exact=True).click()
    expect(form).to_be_hidden()

    filled = "Plan a harbour walk for Adults, 4 stops, on 2026-10-03, notes true."
    expect(chat_input(page)).to_have_text(filled)
    assert _send_and_read(page, upstream, filled) == filled


def test_a_required_variable_holds_the_form_until_it_is_filled(page_for, prompt_with):
    account, save = prompt_with
    command = save('Review {{title | text:placeholder="Title":required}} today.')
    page = page_for(account)

    form = _pick_prompt(page, command)
    form.get_by_role("button", name="Save", exact=True).click()
    expect(form).to_be_visible()
    assert form.get_by_placeholder("Title").evaluate("field => field.validity.valueMissing")

    form.get_by_placeholder("Title").fill("the launch notes")
    form.get_by_role("button", name="Save", exact=True).click()
    expect(form).to_be_hidden()
    expect(chat_input(page)).to_have_text("Review the launch notes today.")


def test_a_variable_left_alone_keeps_its_default(page_for, prompt_with, upstream):
    account, save = prompt_with
    command = save(
        'Pack {{count | number:placeholder="How many":default=3}} bags for'
        ' {{trip | select:options=["Hiking","Sailing"]:default="Sailing"}}.'
    )
    page = page_for(account)

    form = _pick_prompt(page, command)
    expect(form.get_by_placeholder("How many")).to_have_value("3")
    expect(form.get_by_role("combobox")).to_have_value("Sailing")
    form.get_by_role("button", name="Save", exact=True).click()
    expect(form).to_be_hidden()

    filled = "Pack 3 bags for Sailing."
    expect(chat_input(page)).to_have_text(filled)
    assert _send_and_read(page, upstream, filled) == filled
