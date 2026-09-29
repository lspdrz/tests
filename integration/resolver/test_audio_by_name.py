"""Journey: speech engines reached by host name, with the threaded and the c-ares resolver.

`AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` decides which resolver the shared aiohttp pool uses, and every
text-to-speech and speech-to-text call an admin can point at a local service goes through that
pool. Each test drives one engine by a host name (`localhost`, a hosts-file name for `::1` alone
and a hosts-file name for another local address) and expects the same outcome under both
resolvers: an OpenAI-compatible engine's voices, models, speech and transcript (the transcript
sent as a form and as JSON), Azure's voices, speech and batch transcript, and Mistral's voices,
speech and transcript (through the transcription endpoint and through chat completions). An engine
whose name does not resolve answers each of those calls with the same status and message, the
resolver's own wording aside, and the voice and model lists fall back to their defaults. Left out:
Deepgram, ElevenLabs' models and voices and the `/openai/audio/speech` route, whose hosts are
hardcoded or are only settable in the environment (`ELEVENLABS_API_BASE_URL`) or must be
`api.openai.com`, so no local name reaches them.

Discriminates: on dev 176d31d1d, a backend copy whose `env.py` installs a resolver that fails every
lookup when the flag is on turns every c-ares run of a by-name test red and leaves every threaded
run green; the same resolver installed for the flag off does the reverse. A failure test stays green
under a failing resolver by design; it goes red under c-ares on a host whose DNS server replays
cached answers with a stale EDNS cookie, which c-ares drops (the lookup times out).
"""

from __future__ import annotations

import base64
import contextlib
import io
import uuid
import wave

import pytest

from harness.audio_engine import (
    AUDIO_CONFIG,
    AUDIO_NAMESPACE,
    CONFIG_IMPORT,
    SPEECH,
    SPEECH_MODEL,
    TRANSCRIPT,
    VOICE,
    serve_audio_engine,
    using_audio_engine,
)
from harness.host_names import (
    FAILS_WITHIN,
    UNRESOLVABLE,
    name_forms,
    serving_by_name,
    timed,
    without_resolver_reason,
)
from harness.listener import json_answer

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

AZURE_VOICE = "en-US-JennyNeural"
AZURE_KEY = "azure-key-0123456789"
MISTRAL_KEY = "mistral-key-0123456789"
MISTRAL_MODEL = "voxtral-mini-latest"
CONNECTION_ERROR = "Open WebUI: Server Connection Error"
TRANSCRIPTION_FAILED = "Error transcribing chunk"
UNRESOLVABLE_URL = f"http://{UNRESOLVABLE}:8000"
DEFAULT_OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
TTS_DEFAULTS = [{"id": "tts-1"}, {"id": "tts-1-hd"}]


def _recording() -> bytes:
    """A tenth of a second of silence, as a WAV file."""
    recording = io.BytesIO()
    with wave.open(recording, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 1600)
    return recording.getvalue()


@contextlib.contextmanager
def _audio_settings(client, tts: dict, stt: dict):
    """Save these text-to-speech and speech-to-text settings, put the old ones back on exit."""
    snapshot = client.get(AUDIO_NAMESPACE)
    snapshot.raise_for_status()
    current = client.get(AUDIO_CONFIG[0])
    current.raise_for_status()
    saved = client.post(
        AUDIO_CONFIG[1],
        json={"tts": {**current.json()["tts"], **tts}, "stt": {**current.json()["stt"], **stt}},
    )
    assert saved.status_code == 200, f"saving the audio settings failed: {saved.text}"
    try:
        yield
    finally:
        restored = client.post(CONFIG_IMPORT, json={"config": snapshot.json()})
        assert restored.status_code == 200, f"restoring the audio settings failed: {restored.text}"


def _speak(client, text: str | None = None):
    """A read-aloud request whose text no earlier test used, so the speech cache never answers."""
    return client.post(
        "/api/v1/audio/speech",
        json={"input": text or f"read this aloud {uuid.uuid4().hex}", "voice": VOICE},
    )


def _transcribe(client):
    return client.post(
        "/api/v1/audio/transcriptions", files={"file": ("recording.wav", _recording(), "audio/wav")}
    )


def _serve_azure(listener) -> None:
    listener.route("POST", "/cognitiveservices/v1", (200, {"Content-Type": "audio/mpeg"}, SPEECH))
    voices = [{"ShortName": AZURE_VOICE, "DisplayName": "Jenny"}]
    listener.route("GET", "/cognitiveservices/voices/list", json_answer(voices))
    listener.route(
        "POST",
        "/speechtotext/transcriptions:transcribe",
        json_answer({"combinedPhrases": [{"text": TRANSCRIPT}]}),
    )


def _serve_mistral(listener) -> None:
    speech = {"audio_data": base64.b64encode(SPEECH).decode()}
    listener.route("POST", "/audio/speech", json_answer(speech))
    listener.route(
        "GET", "/audio/voices", json_answer({"items": [{"id": VOICE, "name": "Narrator"}]})
    )
    listener.route("POST", "/audio/transcriptions", json_answer({"text": TRANSCRIPT}))
    reply = {"choices": [{"message": {"content": TRANSCRIPT}}]}
    listener.route("POST", "/chat/completions", json_answer(reply))


def _azure_settings(base_url: str) -> tuple[dict, dict]:
    tts = {
        "ENGINE": "azure",
        "API_KEY": AZURE_KEY,
        "VOICE": AZURE_VOICE,
        "AZURE_SPEECH_REGION": "eastus",
        "AZURE_SPEECH_BASE_URL": base_url,
        "AZURE_SPEECH_OUTPUT_FORMAT": "audio-24khz-48kbitrate-mono-mp3",
    }
    stt = {
        "ENGINE": "azure",
        "AZURE_API_KEY": AZURE_KEY,
        "AZURE_REGION": "eastus",
        "AZURE_LOCALES": "en-US",
        "AZURE_BASE_URL": base_url,
    }
    return tts, stt


def _mistral_settings(base_url: str, chat_completions: bool = False) -> tuple[dict, dict]:
    tts = {
        "ENGINE": "mistral",
        "MODEL": "voxtral-mini-tts-2603",
        "MISTRAL_API_KEY": MISTRAL_KEY,
        "MISTRAL_API_BASE_URL": base_url,
    }
    stt = {
        "ENGINE": "mistral",
        "MODEL": MISTRAL_MODEL,
        "MISTRAL_API_KEY": MISTRAL_KEY,
        "MISTRAL_API_BASE_URL": base_url,
        "MISTRAL_USE_CHAT_COMPLETIONS": chat_completions,
    }
    return tts, stt


@pytest.mark.parametrize("name_form", name_forms())
def test_an_openai_compatible_engine_answers_every_call_by_name(
    resolver, name_form, resolving_admin
):
    with serving_by_name(name_form) as listener, resolving_admin.client() as client:
        engine = serve_audio_engine(listener)
        with using_audio_engine(client, engine):
            voices = client.get("/api/v1/audio/voices")
            models = client.get("/api/v1/audio/models")
            spoken = _speak(client)
            transcribed = _transcribe(client)

    assert voices.status_code == 200, voices.text
    assert voices.json()["voices"] == [{"id": VOICE, "name": "The Narrator"}]
    assert models.status_code == 200, models.text
    assert models.json()["models"] == [{"id": SPEECH_MODEL}]
    assert spoken.status_code == 200, spoken.text
    assert spoken.content == SPEECH
    assert transcribed.status_code == 200, transcribed.text
    assert transcribed.json()["text"] == TRANSCRIPT
    assert len(engine.speech_requests()) == 1
    assert len(engine.transcription_requests()) == 1


@pytest.mark.parametrize("name_form", name_forms())
def test_a_json_transcription_request_reaches_the_engine_by_name(
    resolver, name_form, resolving_admin
):
    with serving_by_name(name_form) as listener, resolving_admin.client() as client:
        engine = serve_audio_engine(listener)
        with using_audio_engine(client, engine):
            stt = client.get(AUDIO_CONFIG[0]).json()["stt"]
            with _audio_settings(client, {}, {**stt, "OPENAI_API_REQUEST_FORMAT": "json"}):
                transcribed = _transcribe(client)

    assert transcribed.status_code == 200, transcribed.text
    assert transcribed.json()["text"] == TRANSCRIPT
    [sent] = engine.transcription_requests()
    assert sent.headers["Content-Type"] == "application/json"
    assert sent.json()["input_audio"]["format"] == "wav"


@pytest.mark.parametrize("name_form", name_forms())
def test_an_azure_engine_answers_every_call_by_name(resolver, name_form, resolving_admin):
    with serving_by_name(name_form) as listener, resolving_admin.client() as client:
        _serve_azure(listener)
        tts, stt = _azure_settings(listener.base_url)
        with _audio_settings(client, tts, stt):
            voices = client.get("/api/v1/audio/voices")
            spoken = _speak(client)
            transcribed = _transcribe(client)

    assert voices.status_code == 200, voices.text
    assert voices.json()["voices"] == [{"id": AZURE_VOICE, "name": f"Jenny ({AZURE_VOICE})"}]
    assert spoken.status_code == 200, spoken.text
    assert spoken.content == SPEECH
    assert transcribed.status_code == 200, transcribed.text
    assert transcribed.json()["text"] == TRANSCRIPT
    [speech] = listener.requests_to("/cognitiveservices/v1")
    assert speech.headers["Ocp-Apim-Subscription-Key"] == AZURE_KEY
    assert len(listener.requests_to("/speechtotext/transcriptions:transcribe")) == 1


@pytest.mark.parametrize("chat_completions", [False, True], ids=["transcriptions", "chat"])
@pytest.mark.parametrize("name_form", name_forms())
def test_a_mistral_engine_answers_every_call_by_name(
    resolver, name_form, chat_completions, resolving_admin
):
    with serving_by_name(name_form) as listener, resolving_admin.client() as client:
        _serve_mistral(listener)
        tts, stt = _mistral_settings(listener.base_url, chat_completions)
        with _audio_settings(client, tts, stt):
            voices = client.get("/api/v1/audio/voices")
            spoken = _speak(client)
            transcribed = _transcribe(client)

    assert voices.status_code == 200, voices.text
    assert voices.json()["voices"] == [{"id": VOICE, "name": "Narrator"}]
    assert spoken.status_code == 200, spoken.text
    assert spoken.content == SPEECH
    assert transcribed.status_code == 200, transcribed.text
    assert transcribed.json()["text"] == TRANSCRIPT
    path = "/chat/completions" if chat_completions else "/audio/transcriptions"
    assert len(listener.requests_to(path)) == 1
    assert len(listener.requests_to("/audio/speech")) == 1


def _unresolvable_settings(engine: str) -> tuple[dict, dict]:
    if engine == "openai":
        connection = {"OPENAI_API_BASE_URL": f"{UNRESOLVABLE_URL}/v1", "OPENAI_API_KEY": "sk-gone"}
        return (
            {**connection, "ENGINE": "openai", "MODEL": SPEECH_MODEL, "VOICE": VOICE},
            {**connection, "ENGINE": "openai", "MODEL": "whisper-1"},
        )
    if engine == "azure":
        return _azure_settings(UNRESOLVABLE_URL)
    return _mistral_settings(f"{UNRESOLVABLE_URL}/v1")


@pytest.mark.parametrize("engine", ["openai", "azure", "mistral"])
def test_an_unresolvable_engine_fails_speech_and_transcription_the_same_way(
    resolver, engine, resolving_admin
):
    tts, stt = _unresolvable_settings(engine)
    with resolving_admin.client() as client, _audio_settings(client, tts, stt):
        spoken, speech_seconds = timed(_speak, client)
        transcribed, transcription_seconds = timed(_transcribe, client)

    assert (spoken.status_code, spoken.json()) == (500, {"detail": CONNECTION_ERROR})
    assert transcribed.status_code == 500, transcribed.text
    failure = without_resolver_reason(transcribed.json()["detail"])
    assert failure.startswith(TRANSCRIPTION_FAILED), failure
    assert CONNECTION_ERROR in failure or UNRESOLVABLE in failure, failure
    assert max(speech_seconds, transcription_seconds) < FAILS_WITHIN, "the lookup failed slowly"


@pytest.mark.parametrize("engine", ["openai", "azure", "mistral"])
def test_an_unresolvable_engine_leaves_the_voice_and_model_lists_at_their_defaults(
    resolver, engine, resolving_admin
):
    tts, stt = _unresolvable_settings(engine)
    with resolving_admin.client() as client, _audio_settings(client, tts, stt):
        voices, voice_seconds = timed(client.get, "/api/v1/audio/voices")
        models, model_seconds = timed(client.get, "/api/v1/audio/models")

    assert voices.status_code == 200, voices.text
    assert models.status_code == 200, models.text
    expected = {
        "openai": ([{"id": name, "name": name} for name in DEFAULT_OPENAI_VOICES], TTS_DEFAULTS),
        "azure": ([], []),
        "mistral": ([], [{"id": "voxtral-mini-tts-2603"}]),
    }[engine]
    assert (voices.json()["voices"], models.json()["models"]) == expected
    assert max(voice_seconds, model_seconds) < FAILS_WITHIN, "the lookup failed slowly"
