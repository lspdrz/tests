"""Journey: Read Aloud with the server's own text-to-speech, on a voice built on disk.

With the `transformers` engine, Read Aloud on an answer asks the server for each sentence, which
speaks it with the SpeechT5 pipeline of `harness.local_voices` and a speaker embedding loaded with
datasets, and the chat's audio element plays what comes back. Twin, in the browser, of
integration/deps/test_local_text_to_speech.py.

Discriminates: passes on the dev ef67cc3fa build; in a backend copy whose pipeline is built for
text generation the speech request fails and the answer is never played.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness.actors import admin_of, create_user
from harness.audio_engine import AUDIO_CONFIG
from harness.chat_history import seed_chat
from harness.local_voices import local_voices_env, save_tiny_speech, speaker

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ANSWER = "The keeper lights the lamp at dusk."
PLAYED = """() => {
    const audio = document.getElementById('audioElement');
    return audio.src.startsWith('blob:') && audio.played.length > 0;
}"""


@pytest.fixture(scope="module")
def speaking(instance_with, tmp_path_factory):
    instance = instance_with(local_voices_env(save_tiny_speech(tmp_path_factory.mktemp("voice"))))
    if not instance.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    with admin_of(instance).client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        tts = {**current["tts"], "ENGINE": "transformers", "MODEL": speaker(12)}
        stt = {**current["stt"], "ENGINE": "openai"}
        client.post(AUDIO_CONFIG[1], json={"tts": tts, "stt": stt}).raise_for_status()
    return instance


def _is_speech(response) -> bool:
    return response.url.endswith("/api/v1/audio/speech")


def test_an_answer_is_read_aloud_by_the_local_voice(page_for, speaking):
    owner = create_user(speaking)
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "lamp?"}, {"role": "assistant", "content": ANSWER}],
        )
    page = page_for(owner)
    page.goto(f"/c/{chat_id}")
    expect(page.get_by_text(ANSWER)).to_be_visible()

    with page.expect_response(_is_speech) as spoken:
        page.get_by_role("button", name="Read Aloud").click()

    assert spoken.value.ok, spoken.value.text()
    page.wait_for_function(PLAYED)
    expect(page.get_by_text("Audio playback failed")).to_have_count(0)
