"""Dependency smoke: speech-to-text and text-to-speech through an OpenAI-compatible engine.

python-mimeparse decides which uploads count as audio: without a list every audio type and WebM
video, with the admin's list only its types, parameters included, and an unparseable type is
refused cleanly. An audio file uploaded as a document is transcribed the same way. aiofiles
writes the upload, reads it back for the engine (streamed as a form with the language asked
for, or base64 in JSON for an engine that asks for that), writes the speech the engine returns
and the cache that answers a repeated request. pydub works through ffmpeg: it transcodes speech
that does not come back as MP3, raw PCM at the rate its type names (24 kHz when it names none),
converts a recording in a codec ffprobe says the engine may not take to MP3, and squeezes one
over 20 MB into 16 kHz mono MP3; those tests skip on a host without ffmpeg (or ffprobe), and a
FLAC recording, which the engine may take, reaches it as it came. The engine is the audio
stand-in of `harness/audio_engine.py`, saved into the shared instance's audio settings for each
test. Local Whisper runs a tiny model built on disk (`harness/local_whisper.py`), and
faster-whisper decodes each recording with PyAV (av) before it hears it, so a recording av
cannot decode is refused.

Discriminates: passes on dev ac00d40e3; in a backend copy, `strict_match_mime_type` taking the
first supported type without `mimeparse.best_match` lets the text upload through, skipping the
cache lookup in `speech` asks the engine twice and dropping the `aiofiles` write of the speech
serves an empty file. On dev ef67cc3fa, an empty read of the upload for the JSON request sends
the engine no audio and an `av.open` that fails fails both local Whisper recordings, as do
faster-whisper's `transcribe` given `beams` for `beam_size`, a segment read as `txt` for `text`
and the transcription info read as `lang` for `language`; `WhisperModel` given `model_path` for
`model_size_or_path` fails the switch to local Whisper. Also on ef67cc3fa (with ffmpeg and ffprobe
on PATH), one copy that ignores the PCM rate, skips the 16 kHz mono downmix, never converts, drops
the parameter check after `mimeparse.best_match`, drops the language and never transcribes an
uploaded file fails exactly the rate-given, codec, over-20-MB, language, two parameter and document
tests; a default PCM rate of 16 kHz fails the default-rate case. Twin of unit/deps/test_aiofiles.py,
unit/deps/test_av.py, unit/deps/test_faster_whisper.py, unit/deps/test_pydub.py and
unit/deps/test_python_mimeparse.py.
"""

from __future__ import annotations

import base64
import email.policy
import io
import re
import shutil
import uuid
import wave
from email.parser import BytesParser

import av
import numpy
import pytest
import soundfile

from harness.audio_engine import (
    AUDIO_CONFIG,
    SPEECH,
    TRANSCRIPT,
    serve_audio_engine,
    using_audio_engine,
)
from harness.local_whisper import WORDS, save_tiny_whisper, using_local_whisper

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="pydub decodes and encodes through ffmpeg"
)
# pydub asks ffprobe what a recording holds before it decides to convert it
NEEDS_FFPROBE = pytest.mark.skipif(
    shutil.which("ffprobe") is None, reason="pydub reads the codec with ffprobe"
)


def _ogg_recording() -> bytes:
    """A tenth of a second of silence, Ogg Vorbis like a browser's recording."""
    recording = io.BytesIO()
    silence = numpy.zeros(1600, dtype="float32")
    soundfile.write(recording, silence, 16000, format="OGG", subtype="VORBIS")
    return recording.getvalue()


def _flac_recording(seconds: float = 0.1, channels: int = 1, rate: int = 16000) -> bytes:
    """Noise in FLAC, a format Open WebUI hands to the engine as it came."""
    recording = io.BytesIO()
    noise = numpy.random.default_rng(7).uniform(-0.5, 0.5, (int(rate * seconds), channels))
    soundfile.write(recording, noise, rate, format="FLAC", subtype="PCM_16")
    return recording.getvalue()


def _wav_speech() -> bytes:
    speech = io.BytesIO()
    with wave.open(speech, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x00\x00" * 2400)
    return speech.getvalue()


@pytest.fixture
def engine(admin, listener):
    engine = serve_audio_engine(listener)
    with admin.client() as client, using_audio_engine(client, engine):
        yield engine


@pytest.fixture
def speaker(make_user):
    return make_user()


def _transcribe(speaker, filename: str, content: bytes, content_type: str):
    with speaker.client() as client:
        return client.post(
            "/api/v1/audio/transcriptions", files={"file": (filename, content, content_type)}
        )


def _speak(speaker, text: str):
    with speaker.client() as client:
        return client.post("/api/v1/audio/speech", json={"input": text, "voice": "alloy"})


def test_a_recording_is_written_and_transcribed(speaker, engine):
    recording = _flac_recording()
    before = len(engine.transcription_requests())

    transcribed = _transcribe(speaker, "recording.flac", recording, "audio/flac")

    assert transcribed.status_code == 200, transcribed.text
    assert transcribed.json()["text"] == TRANSCRIPT
    sent = engine.transcription_requests()[before:]
    assert len(sent) == 1 and sent[0].headers["Content-Type"].startswith("multipart/form-data")
    assert recording in sent[0].body


def test_an_engine_asking_for_json_gets_the_recording_as_base64(admin, speaker, engine):
    with admin.client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        stt = {**current["stt"], "OPENAI_API_REQUEST_FORMAT": "json"}
        saved = client.post(AUDIO_CONFIG[1], json={"tts": current["tts"], "stt": stt})
    assert saved.status_code == 200, saved.text
    recording = _flac_recording()
    before = len(engine.transcription_requests())

    transcribed = _transcribe(speaker, "recording.flac", recording, "audio/flac")

    assert transcribed.status_code == 200, transcribed.text
    assert transcribed.json()["text"] == TRANSCRIPT
    [sent] = engine.transcription_requests()[before:]
    assert sent.json()["input_audio"] == {
        "data": base64.b64encode(recording).decode(),
        "format": "flac",
    }


def test_a_text_upload_is_not_taken_for_audio(speaker, engine):
    before = len(engine.transcription_requests())

    refused = _transcribe(speaker, "notes.ogg", b"just some text", "text/plain")

    assert refused.status_code >= 400, refused.text
    assert "text" not in refused.json()
    assert len(engine.transcription_requests()) == before


def _speech_requests_for(engine, text: str) -> list:
    requests = engine.listener.requests_to("/audio/speech")
    return [request for request in requests if text.encode() in request.body]


def test_speech_is_asked_for_once_and_then_served_from_the_cache(speaker, engine):
    text = f"read this aloud {uuid.uuid4().hex}"

    first = _speak(speaker, text)
    second = _speak(speaker, text)

    assert first.status_code == 200, first.text
    assert first.content == SPEECH
    assert second.status_code == 200, second.text
    assert second.content == SPEECH
    assert len(_speech_requests_for(engine, text)) == 1, "the cached speech was not reused"


@NEEDS_FFMPEG
def test_wav_speech_is_transcoded_to_mp3(speaker, engine):
    engine.speech, engine.speech_type = _wav_speech(), "audio/wav"

    spoken = _speak(speaker, f"transcode this {uuid.uuid4().hex}")

    assert spoken.status_code == 200, spoken.text
    assert spoken.content[:3] == b"ID3" or spoken.content[:2] in (b"\xff\xfb", b"\xff\xf3")
    assert not spoken.content.startswith(b"RIFF"), "the WAV speech was passed through as is"


def _form_parts(request) -> dict[str, bytes]:
    """The fields of a multipart form the engine was sent, by name."""
    head = f"Content-Type: {request.headers['Content-Type']}\r\n\r\n".encode()
    message = BytesParser(policy=email.policy.HTTP).parsebytes(head + request.body)
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in message.iter_parts()
    }


def _decoded(audio: bytes) -> tuple[int, int, float]:
    """(sample rate, channels, seconds) of an encoded recording, as av decodes it."""
    with av.open(io.BytesIO(audio)) as container:
        stream = container.streams.audio[0]
        samples = sum(frame.samples for frame in container.decode(stream))
        return stream.rate, stream.channels, samples / stream.rate


def _is_mp3(audio: bytes) -> bool:
    return audio[:3] == b"ID3" or (audio[0] == 0xFF and audio[1] & 0xE0 == 0xE0)


@pytest.mark.parametrize(
    ("content_type", "rate"),
    [
        pytest.param("audio/L16; rate=16000; channels=1", 16000, id="rate-given"),
        pytest.param("audio/pcm", 24000, id="default-rate"),
    ],
)
@NEEDS_FFMPEG
def test_raw_pcm_speech_is_encoded_as_mp3_at_its_rate(speaker, engine, content_type, rate):
    # one second of samples at the rate the engine names, or 24 kHz when it names none
    engine.speech = numpy.zeros(rate, dtype="<i2").tobytes()
    engine.speech_type = content_type

    spoken = _speak(speaker, f"raw speech {uuid.uuid4().hex}")

    assert spoken.status_code == 200, spoken.text
    assert _is_mp3(spoken.content), spoken.content[:16]
    _, channels, seconds = _decoded(spoken.content)
    assert channels == 1
    assert seconds == pytest.approx(1.0, abs=0.1)


@NEEDS_FFMPEG
@NEEDS_FFPROBE
def test_a_recording_in_a_codec_the_engine_may_not_take_reaches_it_as_mp3(speaker, engine):
    before = len(engine.transcription_requests())

    transcribed = _transcribe(speaker, "recording.ogg", _ogg_recording(), "audio/ogg")

    assert transcribed.status_code == 200, transcribed.text
    [sent] = engine.transcription_requests()[before:]
    audio = _form_parts(sent)["file"]
    assert _is_mp3(audio), audio[:16]
    assert _decoded(audio)[2] == pytest.approx(0.1, abs=0.1)


@NEEDS_FFMPEG
@pytest.mark.slow
def test_a_recording_over_20_mb_reaches_the_engine_as_16_khz_mono_mp3(speaker, engine):
    # noise does not compress, so two minutes of CD-quality stereo stay over 20 MB as FLAC
    recording = _flac_recording(seconds=130, channels=2, rate=44100)
    assert len(recording) > 20 * 1024 * 1024
    before = len(engine.transcription_requests())

    transcribed = _transcribe(speaker, "interview.flac", recording, "audio/flac")

    assert transcribed.status_code == 200, transcribed.text
    [sent] = engine.transcription_requests()[before:]
    audio = _form_parts(sent)["file"]
    assert _is_mp3(audio), audio[:16]
    assert len(audio) < 1024 * 1024
    rate, channels, seconds = _decoded(audio)
    assert (rate, channels) == (16000, 1)
    assert seconds == pytest.approx(130, abs=1)


def test_the_language_asked_for_is_passed_to_the_engine(speaker, engine):
    before = len(engine.transcription_requests())
    with speaker.client() as client:
        transcribed = client.post(
            "/api/v1/audio/transcriptions",
            files={"file": ("recording.flac", _flac_recording(), "audio/flac")},
            data={"language": "nl"},
        )

    assert transcribed.status_code == 200, transcribed.text
    [sent] = engine.transcription_requests()[before:]
    assert _form_parts(sent)["language"] == b"nl"


# ---------------------------------------------------------------- which uploads count as audio


def _save_supported_types(admin, supported: list[str]) -> None:
    """Save the admin's list of audio types; the `engine` fixture puts the settings back."""
    with admin.client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        stt = {**current["stt"], "SUPPORTED_CONTENT_TYPES": supported}
        saved = client.post(AUDIO_CONFIG[1], json={"tts": current["tts"], "stt": stt})
    assert saved.status_code == 200, saved.text


def _is_taken_for_audio(speaker, engine, content_type: str) -> bool:
    before = len(engine.transcription_requests())
    answer = _transcribe(speaker, "recording.flac", _flac_recording(), content_type)
    taken = answer.status_code == 200
    assert taken == (len(engine.transcription_requests()) > before), answer.text
    assert taken or answer.status_code == 400, answer.text
    return taken


@pytest.mark.parametrize(
    ("content_type", "taken"),
    [
        pytest.param("audio/x-harbour-recorder", True, id="any-audio"),
        pytest.param("video/webm", True, id="webm-video"),
        pytest.param("video/mp4", False, id="other-video"),
        pytest.param("audio", False, id="unparseable"),
    ],
)
def test_without_a_list_every_audio_type_and_webm_video_count_as_audio(
    speaker, engine, content_type, taken
):
    assert _is_taken_for_audio(speaker, engine, content_type) is taken


@pytest.mark.parametrize(
    ("content_type", "taken"),
    [
        pytest.param("audio/ogg", True, id="listed"),
        pytest.param("audio/webm; codecs=opus", True, id="listed-parameter"),
        pytest.param("audio/webm; codecs=vp8", False, id="other-parameter"),
        pytest.param("audio/webm", False, id="parameter-missing"),
        pytest.param("audio/wav", False, id="not-listed"),
    ],
)
def test_the_admins_list_decides_which_uploads_count_as_audio(
    admin, speaker, engine, content_type, taken
):
    _save_supported_types(admin, ["audio/ogg", "audio/webm;codecs=opus"])

    assert _is_taken_for_audio(speaker, engine, content_type) is taken


def test_an_audio_file_uploaded_as_a_document_is_read_as_its_transcript(speaker, engine):
    with speaker.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": ("minutes.flac", _flac_recording(), "audio/flac")},
        )
        assert uploaded.status_code == 200, uploaded.text
        read = client.get(f"/api/v1/files/{uploaded.json()['id']}/data/content")

    assert read.status_code == 200, read.text
    assert read.json()["content"] == TRANSCRIPT


@pytest.fixture(scope="module")
def tiny_whisper(tmp_path_factory):
    return save_tiny_whisper(tmp_path_factory.mktemp("whisper"))


@pytest.fixture
def local_whisper(admin, tiny_whisper):
    with admin.client() as client, using_local_whisper(client, tiny_whisper):
        yield


@pytest.mark.parametrize(
    ("filename", "recording", "content_type"),
    [
        pytest.param("recording.ogg", _ogg_recording, "audio/ogg", id="ogg"),
        pytest.param("recording.wav", _wav_speech, "audio/wav", id="wav"),
    ],
)
def test_local_whisper_decodes_and_transcribes_a_recording(
    speaker, local_whisper, filename, recording, content_type
):
    transcribed = _transcribe(speaker, filename, recording(), content_type)

    assert transcribed.status_code == 200, transcribed.text
    heard = transcribed.json()["text"]
    assert re.fullmatch(f"(?:{'|'.join(WORDS)}|\\s)+", heard), f"not the tiny model's: {heard!r}"


def test_a_recording_local_whisper_cannot_decode_is_refused(speaker, local_whisper):
    refused = _transcribe(speaker, "recording.ogg", b"OggS but not really a recording", "audio/ogg")

    assert refused.status_code >= 400, refused.text
    assert "text" not in refused.json()
