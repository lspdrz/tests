"""Journey: dictating into the chat input with local Whisper, on a model built on disk.

The chat input's microphone records the browser's own WebM (Opus) and sends it to the server,
which transcribes it with local Whisper: faster-whisper decodes the recording with PyAV (av)
and runs the tiny model of `harness.local_whisper`, whose transcript is made of its `WORDS`.
The transcript lands in the chat input, unsent. Twin, in the browser, of the local Whisper
tests in integration/deps/test_audio_stack.py.

Discriminates: passes on the dev ef67cc3fa build; in a backend copy whose `av.open` fails, the
recording is never transcribed and the chat input stays empty, as it does when faster-whisper's
`transcribe` is given `beams` for `beam_size` or a segment is read as `txt` for `text`.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from harness.local_whisper import WORDS, save_tiny_whisper, using_local_whisper
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TRANSCRIBE_TIMEOUT_MS = 30_000
TINY_MODEL_WORDS = re.compile("|".join(WORDS))


@pytest.fixture(scope="module")
def tiny_whisper(tmp_path_factory):
    return save_tiny_whisper(tmp_path_factory.mktemp("whisper"))


@pytest.fixture
def local_whisper(admin, e2e_instance, tiny_whisper):
    with admin.client() as client, using_local_whisper(client, tiny_whisper):
        yield


def test_dictation_puts_the_local_whisper_transcript_in_the_chat_input(
    voice_page_for, make_user, local_whisper
):
    page = voice_page_for(make_user())
    expect(chat_input(page)).to_be_visible()

    page.get_by_role("button", name="Voice Input").click()
    expect(page.get_by_text("0:02", exact=True)).to_be_visible()
    page.get_by_role("button", name="Confirm recording").click()

    expect(chat_input(page)).to_contain_text(TINY_MODEL_WORDS, timeout=TRANSCRIBE_TIMEOUT_MS)
