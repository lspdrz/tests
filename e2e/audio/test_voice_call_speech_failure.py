"""Regression: a voice call hung in "speaking" with no sound when text-to-speech failed.

Issue open-webui/open-webui#30052, fix 24c01bd76 (PR open-webui/open-webui#30372). When the
speech provider failed during a call, the failed sentence never reached the audio cache and the
playback loop queued it again every 200 ms, forever. The overlay stayed on "Tap to interrupt",
showed no error and ignored the microphone until tapped. A failed sentence now marks its reply
as failed: the provider's error shows once, and the call goes back to listening.

Discriminates: passes on the dev efe63bd34 build, fails on that build with 24c01bd76 reverted
(no error shows and the call never takes the next turn).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.listener import json_answer
from utils.voice_call import TURN_TIMEOUT_MS, call_status, start_call, wait_for_transcriptions

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PROVIDER_ERROR = "the speech provider is down"
SPEECH_ERROR = re.compile(r"External: 500.*/audio/speech")  # how the server reports the failure


@pytest.fixture(autouse=True)
def fresh_replies(upstream):
    # the server caches speech by its text, so a reply spoken in an earlier test never fails
    words = uuid.uuid4().hex
    upstream.queue(reply.text(f"First {words}."), reply.text(f"Second {words}."))


@pytest.fixture
def failing_speech(speech_engine, listener):
    listener.route(
        "POST", "/audio/speech", json_answer({"error": {"message": PROVIDER_ERROR}}, status=500)
    )
    return speech_engine


def test_a_speech_failure_shows_the_error_and_the_call_listens_again(
    voice_page_for, make_user, failing_speech
):
    page = voice_page_for(make_user())
    start_call(page)
    wait_for_transcriptions(failing_speech, 1)

    expect(page.get_by_text(SPEECH_ERROR).first).to_be_visible(timeout=TURN_TIMEOUT_MS)
    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    wait_for_transcriptions(failing_speech, 2)


def test_the_next_turn_asks_for_speech_again(voice_page_for, make_user, failing_speech):
    page = voice_page_for(make_user())
    start_call(page)
    wait_for_transcriptions(failing_speech, 2)

    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    assert len(failing_speech.speech_requests()) >= 2, "the second reply was never spoken"


def test_a_working_speech_provider_keeps_the_call_going(voice_page_for, make_user, speech_engine):
    page = voice_page_for(make_user())
    start_call(page)
    wait_for_transcriptions(speech_engine, 2)

    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    assert speech_engine.speech_requests(), "the reply was never spoken"
