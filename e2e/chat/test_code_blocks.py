"""Code blocks in an assistant reply: the label, the highlighting and the controls a person uses.

A fenced block in a reply shows its language above the code, highlights the code by that language
and offers Collapse, Copy and Save (plus Preview on HTML and SVG). A `mermaid` fence draws a
diagram instead of showing its source, and falls back to the source with an error when the
diagram does not parse. Running a Python block is legacy and not covered. The artifacts pane
opened by an HTML block is covered in test_artifacts_pane_after_delete.py.

Discriminates: passes on the 176d31d1d build; with the language label, the highlighter, the
clipboard write, the collapse toggle, the save handler, the preview button or the mermaid
render removed, the matching test goes red.
"""

from __future__ import annotations

import json
import time

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SCRIPT = "function greet(name) {\n  return `hi ${name}` + '$&';\n}\n\ngreet(\"x\");"
BROKEN_DIAGRAM = "graph TD\n  A[Start --> ((("


def fenced(lang: str, code: str) -> str:
    return f"```{lang}\n{code}\n```"


def ask_for(page: Page, upstream, answer: str, prompt: str = "show me the code") -> Locator:
    upstream.queue(reply.text(answer + "\n\nThat is all.", match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "That is all.")
    return last_reply(page)


def editor_lines(block: Locator) -> Locator:
    return block.locator(".cm-line")


def button(reply_box: Locator, name: str) -> Locator:
    return reply_box.get_by_role("button", name=name, exact=True)


def test_a_block_shows_its_language_and_highlights_the_code(page_for, make_user, upstream):
    page = page_for(make_user())

    box = ask_for(page, upstream, fenced("javascript", "const answer = 42;\n// plain note"))

    expect(box.get_by_text("javascript", exact=True)).to_be_visible()
    expect(editor_lines(box).first).to_have_text("const answer = 42;")
    expect(editor_lines(box).last).to_have_text("// plain note")
    # a highlighted token sits in its own element, told apart from its neighbours by colour
    keyword = editor_lines(box).first.locator("span", has_text="const")
    number = editor_lines(box).first.locator("span", has_text="42")
    expect(keyword).to_be_visible()
    expect(number).to_be_visible()
    colours = {
        node.evaluate("element => getComputedStyle(element).color") for node in (keyword, number)
    }
    assert len(colours) == 2, f"keyword and number share one colour: {colours}"


def test_a_block_without_a_language_has_no_highlighted_tokens(page_for, make_user, upstream):
    page = page_for(make_user())

    box = ask_for(page, upstream, fenced("", "const answer = 42;"))

    expect(editor_lines(box).first).to_have_text("const answer = 42;")
    expect(editor_lines(box).first.locator("span")).to_have_count(0)


def test_copy_puts_the_exact_code_on_the_clipboard(page_for, make_user, upstream):
    page = page_for(make_user())
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    box = ask_for(page, upstream, fenced("javascript", SCRIPT))

    button(box, "Copy").click()

    expect(button(box, "Copied")).to_be_visible()
    assert page.evaluate("navigator.clipboard.readText()") == SCRIPT
    expect(button(box, "Copy")).to_be_visible()


def test_each_block_copies_its_own_code(page_for, make_user, upstream):
    page = page_for(make_user())
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    answer = fenced("bash", "echo first") + "\n\n" + fenced("bash", "echo second")
    box = ask_for(page, upstream, answer)

    button(box, "Copy").last.click()

    expect(button(box, "Copied")).to_be_visible()
    assert page.evaluate("navigator.clipboard.readText()") == "echo second"


def test_a_mermaid_fence_is_drawn_as_a_diagram(page_for, make_user, upstream):
    page = page_for(make_user())

    box = ask_for(page, upstream, fenced("mermaid", "graph TD\n  A[Start node] --> B[End node]"))

    diagram = box.locator("svg[aria-roledescription='flowchart-v2']")
    expect(diagram).to_be_visible()
    expect(diagram).to_contain_text("Start node")
    expect(diagram).to_contain_text("End node")
    expect(box).not_to_contain_text("graph TD")
    expect(box.locator(".cm-editor")).to_have_count(0)


def test_a_mermaid_fence_that_does_not_parse_shows_its_source_and_an_error(
    page_for, make_user, upstream
):
    page = page_for(make_user())

    box = ask_for(page, upstream, fenced("mermaid", BROKEN_DIAGRAM))

    expect(box.get_by_text("Failed to render diagram")).to_be_visible()
    expect(box.locator("pre")).to_have_text(BROKEN_DIAGRAM)
    expect(box.locator("svg[aria-roledescription='flowchart-v2']")).to_have_count(0)


def test_collapse_hides_the_code_and_expand_brings_it_back(page_for, make_user, upstream):
    page = page_for(make_user())
    box = ask_for(page, upstream, fenced("python", "first = 1\nsecond = 2\nthird = 3"))
    expect(editor_lines(box)).to_have_count(3)

    button(box, "Collapse").click()

    expect(box.get_by_text("3 hidden lines")).to_be_visible()
    expect(box.locator(".cm-editor")).to_have_count(0)
    expect(button(box, "Copy")).to_be_visible()
    button(box, "Expand").click()
    expect(editor_lines(box)).to_have_count(3)
    expect(box.get_by_text("3 hidden lines")).to_have_count(0)


def test_blocks_start_collapsed_when_the_account_prefers_it(page_for, make_user, upstream):
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"collapseCodeBlocks": True}}
        )
    saved.raise_for_status()
    page = page_for(account)

    box = ask_for(page, upstream, fenced("python", "first = 1\nsecond = 2"))

    expect(box.get_by_text("2 hidden lines")).to_be_visible()
    expect(box.locator(".cm-editor")).to_have_count(0)
    button(box, "Expand").click()
    expect(editor_lines(box)).to_have_count(2)


def stored_reply_text(client, chat_id: str) -> str:
    messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
    return json.dumps(list(messages.values()))


def test_saving_an_edited_block_rewrites_the_reply(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    box = ask_for(page, upstream, fenced("python", "total = 1"))

    editor_lines(box).first.click()
    page.keyboard.press("End")
    page.keyboard.type(" + 41")
    button(box, "Save").click()

    expect(button(box, "Saved")).to_be_visible()
    expect(editor_lines(box).first).to_have_text("total = 1 + 41")
    chat_id = page.url.rsplit("/", 1)[-1]
    deadline = time.monotonic() + 15
    with account.client() as client:
        while "total = 1 + 41" not in (stored := stored_reply_text(client, chat_id)):
            assert time.monotonic() < deadline, f"the edit was never stored: {stored!r}"
            time.sleep(0.2)
    # the stored reply keeps the edit in its output items
    assert "```python\\ntotal = 1 + 41\\n```" in stored
    page.reload()
    expect(editor_lines(last_reply(page)).first).to_have_text("total = 1 + 41")


def test_only_html_and_svg_blocks_offer_a_preview(page_for, make_user, upstream):
    page = page_for(make_user())
    answer = fenced("python", "x = 1") + "\n\n" + fenced("html", "<p>page</p>")

    box = ask_for(page, upstream, answer)

    expect(button(box, "Copy")).to_have_count(2)
    expect(button(box, "Preview")).to_have_count(1)


def test_preview_opens_the_html_block_in_the_artifacts_pane(page_for, make_user, upstream):
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"detectArtifacts": False}}
        )
    saved.raise_for_status()
    page = page_for(account)
    box = ask_for(page, upstream, fenced("html", "<h1>hello preview</h1>"))
    pane = page.locator("#artifacts-container")
    expect(pane).to_have_count(0)

    button(box, "Preview").click()

    expect(pane.frame_locator("iframe").locator("h1")).to_have_text("hello preview")
