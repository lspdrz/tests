"""Speech audio Chromium can play, long enough for a test to act while it is still playing."""

from __future__ import annotations

import io
import wave

SAMPLE_RATE = 8000


def silent_wav(seconds: float) -> bytes:
    recording = io.BytesIO()
    with wave.open(recording, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(b"\x00\x00" * int(SAMPLE_RATE * seconds))
    return recording.getvalue()
