"""Streamed tool-call arguments through /api/v1/messages assemble into one closed block.

Fix commit `061f5e3a6` (#28858) in `utils/anthropic.py` stopped the OpenAI-to-Anthropic stream
converter from re-parsing a tool call's whole argument buffer after every delta: it now only
tries once the buffer can be complete (it ends in `}`, or it never opened with `{`). The
re-parse shows as the CPU the server burns: a long tool call streamed in 2000 pieces costs about
the same as in one piece with the fix, and seconds more without it, since each failed parse walks
the whole buffer so far. The other tests pin what a client of the endpoint sees however the
provider splits the arguments: the `input_json_delta` pieces join to exactly the provider's
JSON, and each tool block closes exactly once, inline, so a fragment arriving after the close is
dropped.

Twin of unit/chat/test_anthropic_tool_call_streaming.py.

Discriminates: with the guard removed on dev ef67cc3fa the CPU test fails (about 6 s of extra
CPU against 0.01 s with it) and the assembly tests pass; a guard that also skips non-object
buffers leaves the array case open until the end of the stream, so
`test_a_non_object_argument_closes_as_soon_as_it_parses` fails and the rest pass.
"""

from __future__ import annotations

import json

import httpx
import pytest

from harness import raw_provider
from harness.raw_provider import RAW_MODEL_ID, chunk, sse
from harness.second_provider import OPENAI_CONFIG

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

FLAT_ARGUMENTS = '{"query": "quarterly revenue", "limit": 25, "verbose": true}'


@pytest.fixture
def raw(admin, preserve, listener) -> raw_provider.RawProvider:
    preserve(OPENAI_CONFIG)
    return raw_provider.connect(admin, listener)


def tool_call_opened(index: int, call_id: str, name: str) -> dict:
    call = {"index": index, "id": call_id, "type": "function"}
    return chunk({"tool_calls": [{**call, "function": {"name": name, "arguments": ""}}]})


def argument_piece(index: int, piece: str) -> dict:
    return chunk({"tool_calls": [{"index": index, "function": {"arguments": piece}}]})


def single_tool_call(pieces: list[str], name: str = "search") -> bytes:
    events = [tool_call_opened(0, "call_1", name)]
    events += [argument_piece(0, piece) for piece in pieces]
    return sse(*events, chunk({}, "tool_calls"))


def split(value: str, size: int) -> list[str]:
    return [value[start : start + size] for start in range(0, len(value), size)]


def stream_messages(admin, raw, body: bytes) -> list[dict]:
    raw.stream(body)
    request = {
        "model": RAW_MODEL_ID,
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "look it up"}],
    }
    with admin.client() as client:
        response: httpx.Response = client.post("/api/v1/messages", json=request)
    assert response.status_code == 200, response.text
    return [
        json.loads(line.removeprefix("data:"))
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]


def arguments_by_block(events: list[dict]) -> dict[int, str]:
    assembled: dict[int, str] = {}
    for event in events:
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "input_json_delta":
            block = event["index"]
            assembled[block] = assembled.get(block, "") + event["delta"]["partial_json"]
    return assembled


def stops_by_block(events: list[dict]) -> list[int]:
    return [event["index"] for event in events if event["type"] == "content_block_stop"]


@pytest.mark.parametrize(
    "pieces",
    [
        pytest.param(split(FLAT_ARGUMENTS, 1), id="one-character-pieces"),
        pytest.param(
            ['{"query": "quar', "terly rev", 'enue", "limit"', ": 25, ", '"verbose": true}'],
            id="split-inside-a-string",
        ),
        pytest.param([FLAT_ARGUMENTS], id="whole-json-in-one-piece"),
    ],
)
def test_arguments_assemble_exactly_and_close_once(admin, raw, pieces):
    events = stream_messages(admin, raw, single_tool_call(pieces))

    assert arguments_by_block(events) == {0: FLAT_ARGUMENTS}
    assert stops_by_block(events) == [0]


def test_parallel_tool_calls_keep_their_own_arguments(admin, raw):
    events = [tool_call_opened(0, "call_a", "alpha"), tool_call_opened(1, "call_b", "beta")]
    for index, arguments in enumerate(['{"a": 1}', '{"b": 2}']):
        events += [argument_piece(index, piece) for piece in split(arguments, 3)]

    received = stream_messages(admin, raw, sse(*events, chunk({}, "tool_calls")))

    assert sorted(arguments_by_block(received).values()) == ['{"a": 1}', '{"b": 2}']
    assert sorted(stops_by_block(received)) == sorted(arguments_by_block(received))


def test_a_tool_call_without_arguments_still_opens_its_block(admin, raw):
    events = stream_messages(admin, raw, single_tool_call([], name="ping"))

    starts = [event["content_block"] for event in events if event["type"] == "content_block_start"]
    assert [(start["type"], start["name"]) for start in starts] == [("tool_use", "ping")]
    assert arguments_by_block(events) == {}


@pytest.mark.parametrize(
    "pieces,assembled",
    [
        pytest.param(["{", "}"], "{}", id="empty-object"),
        pytest.param(['{"a": 1', "}\n", "late"], '{"a": 1}\n', id="brace-then-whitespace"),
    ],
)
def test_a_complete_object_closes_its_block_inline(admin, raw, pieces, assembled):
    events = stream_messages(admin, raw, single_tool_call(pieces))

    assert arguments_by_block(events) == {0: assembled}
    assert stops_by_block(events) == [0]


def test_a_non_object_argument_closes_as_soon_as_it_parses(admin, raw):
    events = stream_messages(admin, raw, single_tool_call(["[1,", " 2]", "late"]))

    assert arguments_by_block(events) == {0: "[1, 2]"}
    assert stops_by_block(events) == [0]


# numbers make a failed parse of the partial buffer costly, as a long tool call's tokens do
LONG_ARGUMENTS = '{"values": [' + "7," * 200_000 + "7]}"
# splitting costs next to nothing with the fix and about six seconds without it
ALLOWED_EXTRA_CPU_SECONDS = 1.0


def server_cpu_for(instance, admin, raw, pieces: list[str]) -> float:
    """The server CPU one streamed tool call costs, the cheaper of two readings."""
    readings = []
    for _ in range(2):
        before = instance.cpu_seconds()
        events = stream_messages(admin, raw, single_tool_call(pieces))
        readings.append(instance.cpu_seconds() - before)
        assert arguments_by_block(events) == {0: LONG_ARGUMENTS}
    return min(readings)


@pytest.mark.slow
def test_a_long_tool_call_costs_about_the_same_however_it_is_split(instance, admin, raw):
    whole = server_cpu_for(instance, admin, raw, [LONG_ARGUMENTS])
    pieces = server_cpu_for(instance, admin, raw, split(LONG_ARGUMENTS, 200))

    measured = f"in one piece: {whole:.2f}s CPU, in 2000 pieces: {pieces:.2f}s CPU"
    print(measured)  # the reading is the point; `-s` shows it whichever way the guard lands
    assert pieces - whole < ALLOWED_EXTRA_CPU_SECONDS, (
        f"the argument buffer is re-parsed after every piece (#28858): {measured}"
    )
