"""Dependency smoke: text to speech on the server itself, with transformers and datasets.

The `transformers` engine builds `pipeline("text-to-speech", "microsoft/speecht5_tts")`, which
brings its HiFi-GAN vocoder along, and loads the `Matthijs/cmu-arctic-xvectors` speaker
embeddings with `datasets.load_dataset`, both once. For each sentence it picks the speaker whose
`filename` is the admin's TTS model (speaker 6799 when none is), runs the pipeline with that
embedding and writes the waveform with soundfile. The voice is `harness/local_voices.py`, a tiny
model in a Hugging Face home of the instance's own, read offline (twin of the text-to-speech part
of unit/deps/test_transformers.py).

Discriminates: passes on dev ef67cc3fa. In a backend copy, a pipeline built for text generation
fails both tests, and a lookup that never finds the named speaker speaks its sentence in the
fallback voice.
"""

from __future__ import annotations

import io

import pytest

from harness.actors import admin_of
from harness.audio_engine import AUDIO_CONFIG
from harness.local_voices import SAMPLING_RATE, local_voices_env, save_tiny_speech, speaker

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

SENTENCE = "the keeper lights the lamp at dusk."


@pytest.fixture(scope="module")
def speaking(instance_with, tmp_path_factory):
    home = save_tiny_speech(tmp_path_factory.mktemp("speech"))
    return instance_with(local_voices_env(home))


@pytest.fixture(scope="module")
def admin_client(speaking):
    with admin_of(speaking).client() as client:
        yield client


def _speak_as(client, voice: str, text: str = SENTENCE) -> bytes:
    """Speech for `text` with the admin's TTS model set to `voice`."""
    current = client.get(AUDIO_CONFIG[0])
    current.raise_for_status()
    tts = {**current.json()["tts"], "ENGINE": "transformers", "MODEL": voice}
    # a blank speech-to-text engine would load local Whisper on every save
    stt = {**current.json()["stt"], "ENGINE": "openai"}
    saved = client.post(AUDIO_CONFIG[1], json={"tts": tts, "stt": stt})
    assert saved.status_code == 200, saved.text
    spoken = client.post("/api/v1/audio/speech", json={"input": text, "voice": voice})
    assert spoken.status_code == 200, spoken.text
    return spoken.content


def _decoded(audio: bytes):
    import soundfile

    samples, sampling_rate = soundfile.read(io.BytesIO(audio))
    return samples, sampling_rate


def test_a_sentence_is_spoken_by_the_local_model(admin_client):
    samples, sampling_rate = _decoded(_speak_as(admin_client, speaker(12)))

    assert sampling_rate == SAMPLING_RATE
    # the model speaks until its length limit: ten output frames of 16 samples per character
    assert len(samples) >= 10 * 16 * len(SENTENCE), len(samples)
    assert abs(samples).max() > 0, "the speech is silent"


def test_the_admins_tts_model_names_the_speaker(admin_client):
    named = _speak_as(admin_client, speaker(12), "the ferry leaves at noon.")
    other = _speak_as(admin_client, speaker(13), "the ferry leaves at noon.")
    fallback = _speak_as(admin_client, speaker(6799), "the ferry leaves at noon.")
    unknown = _speak_as(admin_client, "no such speaker", "the ferry leaves at noon.")

    assert named != other, "two speakers read the sentence in one voice"
    assert named != fallback
    assert unknown == fallback, "an unknown speaker did not fall back to speaker 6799"
