"""Dependency contract: av (PyAV), what local Whisper's transcripts cannot show.

faster-whisper decodes every recording local Whisper transcribes with PyAV (`av.open`, the
audio stream's frames, `AudioResampler` to 16 kHz mono). That path is driven from outside in
integration/deps/test_audio_stack.py, on a tiny Whisper model built on disk: an Ogg and a WAV
recording are decoded and transcribed, and one av cannot decode is refused. Two things stay
here. The tiny model's words do not depend on how much audio it heard, so a decode that drops
part of the recording shows in no transcript. And requirements.txt pins av 14 because a newer
av's bundled FFmpeg crashes under a FIPS-mode OpenSSL, a packaging concern no request sees.
"""

from __future__ import annotations

import io
import math
import struct
import wave

import pytest

pytestmark = pytest.mark.depcheck

AUDIO_FREQ = 440


def _wav_bytes(rate: int, seconds: int) -> io.BytesIO:
    """A mono 16-bit PCM WAV of a 440 Hz sine, in memory."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        samples = (
            int(1000 * math.sin(2 * math.pi * AUDIO_FREQ * index / rate))
            for index in range(rate * seconds)
        )
        writer.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
    buffer.seek(0)
    return buffer


def test_version_is_pinned_major(depcheck):
    """The backend pins av==14.0.1 for FIPS reasons; a new major line needs that re-checked."""
    mod = depcheck.load("av")
    major = int(mod.__version__.split(".")[0])
    assert major == 14, f"av major changed to {mod.__version__}; re-verify the FIPS selftest note"


def test_decoded_sample_count_matches_input(depcheck):
    """Decoding yields every encoded sample, not a truncated prefix."""
    mod = depcheck.load("av")
    rate, seconds = 8000, 1
    container = mod.open(_wav_bytes(rate, seconds), mode="r")
    try:
        total = sum(frame.samples for frame in container.decode(audio=0))
    finally:
        container.close()
    assert total == rate * seconds, f"decoded {total} samples, expected {rate * seconds}"
