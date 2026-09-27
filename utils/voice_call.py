"""Starting a voice call on the chat page and reading what the call overlay shows."""

from __future__ import annotations

import time

from playwright.sync_api import Page, expect

from harness.audio_engine import AudioEngine
from utils.chat_ui import chat_input

TURN_TIMEOUT_MS = 20_000


def start_call(page: Page) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Voice mode").click()
    expect(page.get_by_role("button", name="End call")).to_be_visible()


def call_status(page: Page, status: str):
    return page.get_by_text(status, exact=True)


def wait_for_transcriptions(engine: AudioEngine, count: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while len(engine.transcription_requests()) < count:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"expected {count} recordings to be transcribed, got "
                f"{len(engine.transcription_requests())}"
            )
        time.sleep(0.1)
