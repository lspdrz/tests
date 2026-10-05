"""Regression: Read Aloud and auto-playback ignored the Speech Playback Speed setting.

Issue open-webui/open-webui#31870, fix 273180a0d (PR open-webui/open-webui#31881). The audio queue
plays a reply one sentence at a time through a single audio element, and the browser puts the
element's playback rate back to its default each time a new source is loaded. The queue only
set the current rate, so the chosen speed was lost at every sentence. It now sets the default
rate too.

Discriminates: passes on the dev b859124f9 build, fails on that build with 273180a0d reverted
(the sentences play at normal speed).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_ui import expect_reply, send
from utils.speech_audio import silent_wav

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

SPEED = 2
SENTENCES = 3
# every time the element starts playing a sentence, with the rate it plays at; the silent clip
# the player plays to unlock the element on the first gesture is no sentence
RECORD_RATES = """() => {
    window.playedRates = [];
    document.addEventListener(
        'playing',
        (event) => {
            if (!event.target.src.startsWith('blob:')) return;
            window.playedRates.push(event.target.playbackRate);
        },
        true
    );
}"""
PLAYED_RATES = "() => window.playedRates"


@pytest.fixture(autouse=True)
def short_sentences(speech_engine):
    speech_engine.speech = silent_wav(1.5)


def reply_text() -> str:
    # the server caches speech by its text, so the reply is this test's own
    tag = uuid.uuid4().hex[:6]
    # the player merges sentences shorter than 50 characters into the one before
    return " ".join(
        f"The {ordinal} keeper lights the harbour lamp at dusk {tag}."
        for ordinal in ("first", "second", "third")
    )


def set_speed(page: Page) -> None:
    page.goto("/?settings=audio")
    tab = page.locator("#tab-audio")
    speed = tab.get_by_role("spinbutton", name="Speech Playback Speed")
    expect(speed).to_be_visible()
    speed.fill(str(SPEED))
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        tab.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def played_rates(page: Page) -> list[float]:
    page.wait_for_function(f"() => window.playedRates.length >= {SENTENCES}")
    return page.evaluate(PLAYED_RATES)


def test_every_sentence_read_aloud_plays_at_the_chosen_speed(page_for, make_user):
    owner = make_user()
    page = page_for(owner)
    page.add_init_script(f"({RECORD_RATES})()")
    set_speed(page)
    text = reply_text()
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "lamps?"}, {"role": "assistant", "content": text}],
        )
    page.goto(f"/c/{chat_id}")
    expect(page.get_by_text(text)).to_be_visible()

    page.get_by_role("button", name="Read Aloud").click()

    assert played_rates(page)[:SENTENCES] == [SPEED] * SENTENCES


def test_every_sentence_of_an_auto_played_reply_plays_at_the_chosen_speed(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    page.add_init_script(f"({RECORD_RATES})()")
    set_speed(page)
    switch = page.locator("#tab-audio").get_by_role("switch", name="Auto-Playback Response")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        switch.click()
    expect(switch).to_be_checked()
    question = "which lamps burn?"
    text = reply_text()
    upstream.queue(reply.text(text, match=reply.answering(question)))
    page.goto("/")

    send(page, question)
    expect_reply(page, text)

    assert played_rates(page)[:SENTENCES] == [SPEED] * SENTENCES
