"""Regression: a streamed /api/v1/messages reply that failed partway ended as a finished answer.

Fix PR open-webui/open-webui#31405 (merged as `0fe9ed0d3`, issue open-webui/open-webui#31403)
in `utils/anthropic.py`. The converter that turns the provider's OpenAI-shaped stream into
Anthropic events skipped a chunk carrying `error` like any chunk without choices, then closed
the stream with `message_delta` and `message_stop`, so Claude Code and the Anthropic SDKs took
the cut-off text as the complete answer. It now ends the stream with an Anthropic `error` event
carrying the provider's message and sends no `message_stop`.

The provider is a raw connection that streams some text and then an error chunk.

Discriminates: passes on dev bc2416c5d; with the fix reverted the failed streams end in
`message_stop` with no `error` event, while the successful stream passes on both.
"""

from __future__ import annotations

import json

import pytest

from harness import raw_provider
from harness.raw_provider import RAW_MODEL_ID, chunk, sse
from harness.second_provider import OPENAI_CONFIG

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PARTIAL_TEXT = "The first half of the answer"
DEFAULT_MESSAGE = "Chat completion stream failed"


@pytest.fixture
def raw(admin, preserve, listener) -> raw_provider.RawProvider:
    preserve(OPENAI_CONFIG)
    return raw_provider.connect(admin, listener)


def stream_messages(admin, raw, body: bytes) -> list[dict]:
    raw.stream(body)
    request = {
        "model": RAW_MODEL_ID,
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "tell me everything"}],
    }
    with admin.client() as client:
        response = client.post("/api/v1/messages", json=request)
    assert response.status_code == 200, response.text
    return [
        json.loads(line.removeprefix("data:"))
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]


def event_types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def streamed_text(events: list[dict]) -> str:
    return "".join(
        event["delta"].get("text", "") for event in events if event["type"] == "content_block_delta"
    )


def failing_stream(error) -> bytes:
    return sse(chunk({"role": "assistant", "content": PARTIAL_TEXT}), {"error": error})


# --- narrow: a provider error mid-stream ends the stream with an error ----------------------


def test_a_provider_error_mid_stream_ends_with_an_error_event(admin, raw):
    events = stream_messages(admin, raw, failing_stream({"message": "the provider overloaded"}))

    assert event_types(events)[-1] == "error", event_types(events)
    assert events[-1]["error"] == {"type": "api_error", "message": "the provider overloaded"}
    assert "message_stop" not in event_types(events)


# --- broad: every shape of provider error is reported as a failure --------------------------


@pytest.mark.parametrize(
    "error,message",
    [
        pytest.param("rate limited", "rate limited", id="plain-string"),
        pytest.param({"code": 502}, DEFAULT_MESSAGE, id="object-without-message"),
        pytest.param({"message": ""}, DEFAULT_MESSAGE, id="empty-message"),
    ],
)
def test_any_error_shape_is_never_reported_as_finished(admin, raw, error, message):
    events = stream_messages(admin, raw, failing_stream(error))

    assert "message_stop" not in event_types(events), event_types(events)
    assert events[-1] == {"type": "error", "error": {"type": "api_error", "message": message}}


def test_an_error_before_any_text_is_reported_too(admin, raw):
    events = stream_messages(admin, raw, sse({"error": {"message": "refused upfront"}}))

    assert "message_stop" not in event_types(events), event_types(events)
    assert events[-1]["error"]["message"] == "refused upfront"


# --- nearby: what arrived before the error, and a stream that succeeds ----------------------


def test_the_text_before_the_error_is_still_delivered(admin, raw):
    events = stream_messages(admin, raw, failing_stream({"message": "cut off"}))

    assert event_types(events)[0] == "message_start"
    assert streamed_text(events) == PARTIAL_TEXT


def test_a_successful_stream_still_ends_with_message_stop(admin, raw):
    body = sse(chunk({"role": "assistant", "content": PARTIAL_TEXT}), chunk({}, "stop"))

    events = stream_messages(admin, raw, body)

    assert event_types(events)[-1] == "message_stop"
    assert "error" not in event_types(events)
    assert streamed_text(events) == PARTIAL_TEXT
    [delta] = [event for event in events if event["type"] == "message_delta"]
    assert delta["delta"]["stop_reason"] == "end_turn"
