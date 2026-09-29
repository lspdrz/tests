"""Journey: a streamed reply ends up the same whether its text is appended in place or copied.

`ENABLE_CHAT_RESPONSE_STREAM_INPLACE_APPEND` (off by default, commits 113c56fc8 and 924a4a10f,
comments corrected in #30066) makes the streaming handler extend each growing text field in place
instead of copying it per chunk: the answer text, reasoning from a delta field or from tags,
solution text, tool call arguments split across deltas, OpenRouter's `reasoning_details` and the
socket deltas it merges when a chat asks for `stream_delta_chunk_size`. Only an allocation failure
is meant to behave differently. Every test here runs once with the toggle off and once on and holds
both to the same expectations: the stored reply, every delta a browser tab is sent and the history
the provider gets on the next turn. Each case lands at least two pieces in one field, since the
first piece is never appended.

Discriminates: in a backend copy, prefixing each piece the in-place branch appends with `|` fails
every case with the toggle on and none with it off; the same edit to the copying branch fails every
case with it off and none with it on.
"""

from __future__ import annotations

import json

import pytest

from harness import upstream as reply
from harness.actors import create_user
from harness.chat import ask, send_message, wait_for_reply
from harness.python_tools import python_tool
from harness.second_provider import OPENAI_CONFIG, attach, sse
from harness.socket_client import connected

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

INPLACE_APPEND = "ENABLE_CHAT_RESPONSE_STREAM_INPLACE_APPEND"
SECOND_MODEL = "second-model"
FOLLOW_UP = "and then?"


@pytest.fixture(params=["false", "true"], ids=["append-copies", "append-in-place"])
def streaming(request, instance_with):
    """A scratch instance with the toggle off, then on; every test below runs on both."""
    return instance_with({INPLACE_APPEND: request.param})


@pytest.fixture
def owner(streaming):
    """An admin, so the second connection's model needs no access grant."""
    return create_user(streaming, role="admin")


@pytest.fixture
def second_provider(streaming, owner, preserve, listener):
    """The listener as one more connection serving `SECOND_MODEL`, answering from a script."""
    preserve(OPENAI_CONFIG, on=streaming)
    with owner.client() as client:
        attach(client, listener, SECOND_MODEL)
    return listener


def _script(listener, *answers) -> None:
    """Answer the n-th chat request with the n-th answer."""

    def answer(_request):
        return answers[len(listener.requests_to("/v1/chat/completions")) - 1]

    listener.route("POST", "/v1/chat/completions", answer)


def _sent(listener) -> list[dict]:
    return [entry.json() for entry in listener.requests_to("/v1/chat/completions")]


def _items(message: dict, kind: str) -> list[dict]:
    return [item for item in message["output"] if item["type"] == kind]


def _text(item: dict) -> str:
    return "".join(part.get("text", "") for part in item.get("content") or [])


def _deltas(socket, chat_id: str) -> list[tuple[str, str]]:
    """Every streamed delta the tab got, as (kind, text), in order."""
    return [
        (
            event["data"]["type"].removeprefix("response.").removesuffix(".delta"),
            event["data"]["delta"],
        )
        for event in socket.events_of(chat_id)
        if event.get("type") == "response:completion"
        and str(event["data"].get("type", "")).endswith(".delta")
    ]


def _ask_watched(actor, prompt: str, **options) -> tuple[object, dict, list[tuple[str, str]]]:
    """Ask as a web client with a tab open; returns the turn, the stored reply and the deltas."""
    with connected(actor) as socket, actor.client() as client:
        turn, message = ask(client, prompt, **options)
        return turn, message, _deltas(socket, turn.chat_id)


def _history_after(actor, turn, **options) -> list[dict]:
    """Send the next turn in the chat; the caller reads the provider request it made."""
    with actor.client() as client:
        ask(client, FOLLOW_UP, chat_id=turn.chat_id, parent_id=turn.assistant_message_id, **options)


def _replayed_assistant(request: dict) -> list[dict]:
    return [entry for entry in request["messages"] if entry["role"] in ("assistant", "tool")]


# --- answer text ----------------------------------------------------------------------------

MANY_PIECES = [f"word{index} " for index in range(300)] + ["end."]
LARGE_PIECES = [f"{index}:" + "lorem ipsum dolor sit amet " * 1200 for index in range(8)] + ["end."]
UNICODE_PIECES = [
    "Cafe",
    "\u0301 au lait, ",
    "\U0001f469",
    "\u200d",
    "\U0001f4bb codes, ",
    "日本",
    "語, ",
    "مرحبا",
    " ✓",
]
CODE_PIECES = ["Here:\n\n``", "`python\nprint(", "'hi')\n", "``", "`\n\nDone."]


@pytest.mark.parametrize(
    "pieces",
    [MANY_PIECES, LARGE_PIECES, UNICODE_PIECES, CODE_PIECES],
    ids=["many-small-pieces", "long-reply", "unicode-split", "code-block-split"],
)
def test_answer_text_streams_saves_and_replays_whole(streaming, owner, pieces):
    expected = "".join(pieces)
    streaming.upstream.queue(reply.text(pieces, match=reply.answering("write it")))
    turn, message, deltas = _ask_watched(owner, "write it")

    assert deltas == [("output_text", piece) for piece in pieces]
    assert message["content"] == expected
    assert [(item["type"], item["status"]) for item in message["output"]] == [
        ("message", "completed")
    ]
    assert _text(message["output"][0]) == expected

    streaming.upstream.queue(reply.text("ok", match=reply.answering(FOLLOW_UP)))
    _history_after(owner, turn)
    replayed = _replayed_assistant(streaming.upstream.chat_requests()[-1])
    assert replayed == [{"role": "assistant", "content": expected}]


# --- reasoning ------------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "thinking"])
def test_reasoning_from_a_delta_field_is_kept_apart_from_the_answer(
    streaming, owner, second_provider, field
):
    _script(
        second_provider,
        sse(
            {field: "Let me "},
            {field: "weigh "},
            {field: "this."},
            {"content": "The "},
            {"content": "answer."},
        ),
        sse({"content": "ok"}),
    )
    turn, message, deltas = _ask_watched(owner, "think first", model=SECOND_MODEL)

    assert deltas == [
        ("reasoning_text", "Let me "),
        ("reasoning_text", "weigh "),
        ("reasoning_text", "this."),
        ("output_text", "The "),
        ("output_text", "answer."),
    ]
    assert [item["type"] for item in message["output"]] == ["reasoning", "message"]
    assert _text(message["output"][0]) == "Let me weigh this."
    assert message["output"][0]["status"] == "completed"
    assert message["content"] == "The answer."

    _history_after(owner, turn, model=SECOND_MODEL)
    assert _replayed_assistant(_sent(second_provider)[-1]) == [
        {"role": "assistant", "content": "The answer."}
    ]


TAG_PAIRS = [
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reason>", "</reason>"),
    ("<reasoning>", "</reasoning>"),
    ("<thought>", "</thought>"),
    ("<Thought>", "</Thought>"),
    ("<|begin_of_thought|>", "<|end_of_thought|>"),
    ("◁think▷", "◁/think▷"),
]


@pytest.mark.parametrize("start, end", TAG_PAIRS, ids=[start for start, _ in TAG_PAIRS])
def test_reasoning_between_tags_is_lifted_out_of_the_answer(streaming, owner, start, end):
    pieces = [start, "deep ", "and ", "slow", end, "Answer ", "given."]
    streaming.upstream.queue(reply.text(pieces, match=reply.answering("ponder")))
    turn, message, deltas = _ask_watched(owner, "ponder")

    [reasoning] = _items(message, "reasoning")
    assert _text(reasoning) == "deep and slow"
    assert (reasoning["start_tag"], reasoning["end_tag"], reasoning["status"]) == (
        start,
        end,
        "completed",
    )
    assert message["content"] == "Answer given."
    assert _text(message["output"][-1]) == "Answer given."
    assert deltas == [
        ("reasoning_text", "deep "),
        ("reasoning_text", "and "),
        ("reasoning_text", "slow"),
        ("output_text", "Answer "),
        ("output_text", "given."),
    ]

    streaming.upstream.queue(reply.text("ok", match=reply.answering(FOLLOW_UP)))
    _history_after(owner, turn)
    assert _replayed_assistant(streaming.upstream.chat_requests()[-1]) == [
        {"role": "assistant", "content": "Answer given."}
    ]


def test_a_chats_own_reasoning_tags_are_lifted_out(streaming, owner):
    pieces = ["<plan>", "step ", "one", "</plan>", "Done ", "now."]
    streaming.upstream.queue(reply.text(pieces, match=reply.answering("plan it")))
    _, message, _ = _ask_watched(owner, "plan it", params={"reasoning_tags": ["<plan>", "</plan>"]})

    assert [_text(item) for item in _items(message, "reasoning")] == ["step one"]
    assert message["content"] == "Done now."


def test_solution_tags_keep_the_solution_text(streaming, owner):
    pieces = ["<|begin_of_solution|>", "The ", "answer ", "is 4.", "<|end_of_solution|>"]
    streaming.upstream.queue(reply.text(pieces, match=reply.answering("solve it")))
    _, message, _ = _ask_watched(owner, "solve it")

    assert message["content"] == "The answer is 4."


def test_reasoning_details_are_merged_per_index(streaming, owner, second_provider):
    def thought(text: str, summary: str) -> dict:
        details = [
            {"type": "reasoning.text", "index": 0, "text": text, "format": "unknown"},
            {"type": "reasoning.summary", "index": 1, "summary": summary, "format": "unknown"},
        ]
        return {"reasoning": text, "reasoning_details": details}

    _script(
        second_provider,
        sse(
            thought("First ", "Short"),
            thought("second ", " summary"),
            thought("third.", "."),
            {"content": "Done."},
        ),
        sse({"content": "ok"}),
    )
    turn, message, deltas = _ask_watched(owner, "reason openly", model=SECOND_MODEL)

    [reasoning] = _items(message, "reasoning")
    assert _text(reasoning) == "First second third."
    merged = [
        {"type": "reasoning.text", "index": 0, "text": "First second third.", "format": "unknown"},
        {"type": "reasoning.summary", "index": 1, "summary": "Short summary.", "format": "unknown"},
    ]
    assert reasoning["reasoning_details"] == merged
    assert [text for kind, text in deltas if kind == "reasoning_text"] == [
        "First ",
        "second ",
        "third.",
    ]
    assert message["content"] == "Done."

    _history_after(owner, turn, model=SECOND_MODEL)
    assert _replayed_assistant(_sent(second_provider)[-1]) == [
        {"role": "assistant", "content": "Done.", "reasoning_details": merged}
    ]


# --- tool calls -----------------------------------------------------------------------------

WEATHER_TOOL = '''
class Tools:
    def weather(self, city: str, country: str) -> str:
        """Report the weather in a city.

        :param city: the city
        :param country: the country
        """
        return f"{city}, {country}: sunny"
'''


def _call(
    index: int, call_id: str | None = None, name: str | None = None, arguments: str = ""
) -> dict:
    function = {"arguments": arguments, **({"name": name} if name else {})}
    call = {"index": index, "function": function}
    if call_id:
        call.update(id=call_id, type="function")
    return {"tool_calls": [call]}


def _split(arguments: dict, size: int = 4) -> list[str]:
    encoded = json.dumps(arguments, ensure_ascii=False)
    return [encoded[start : start + size] for start in range(0, len(encoded), size)]


def _streamed_call(index: int, call_id: str, arguments: dict) -> list[dict]:
    pieces = _split(arguments)
    return [_call(index, call_id, "weather", pieces[0])] + [
        _call(index, arguments=p) for p in pieces[1:]
    ]


@pytest.fixture
def weather_tool(streaming):
    from harness.actors import admin_of

    with python_tool(admin_of(streaming), WEATHER_TOOL, name="Weather") as tool_id:
        yield tool_id


def _tool_calls_sent_back(request: dict) -> list[tuple[str, dict]]:
    calls = [
        call
        for entry in request["messages"]
        if entry["role"] == "assistant"
        for call in entry.get("tool_calls") or []
    ]
    return [(call["function"]["name"], json.loads(call["function"]["arguments"])) for call in calls]


def _tool_results_sent_back(request: dict) -> list[str]:
    return [entry["content"] for entry in request["messages"] if entry["role"] == "tool"]


def test_split_tool_arguments_reach_the_tool_whole(streaming, owner, second_provider, weather_tool):
    arguments = {"city": "São Paulo", "country": "Brasil"}
    _script(
        second_provider,
        sse(*_streamed_call(0, "call_a", arguments), finish_reason="tool_calls"),
        sse({"content": "It is "}, {"content": "sunny."}),
        sse({"content": "ok"}),
    )
    turn, message, _ = _ask_watched(owner, "weather?", model=SECOND_MODEL, tool_ids=[weather_tool])

    [call] = _items(message, "function_call")
    assert (call["name"], json.loads(call["arguments"])) == ("weather", arguments)
    [result] = _items(message, "function_call_output")
    assert "São Paulo, Brasil: sunny" in json.dumps(result, ensure_ascii=False)
    assert message["content"].endswith("It is sunny.")

    tool_round = _sent(second_provider)[1]
    assert _tool_calls_sent_back(tool_round) == [("weather", arguments)]
    assert _tool_results_sent_back(tool_round) == ["São Paulo, Brasil: sunny"]

    _history_after(owner, turn, model=SECOND_MODEL, tool_ids=[weather_tool])
    replayed = _sent(second_provider)[-1]
    assert _tool_calls_sent_back(replayed) == [("weather", arguments)]
    assert _tool_results_sent_back(replayed) == ["São Paulo, Brasil: sunny"]


def test_parallel_calls_with_interleaved_arguments_run_apart(
    streaming, owner, second_provider, weather_tool
):
    lisbon, oslo = {"city": "Lisbon", "country": "Portugal"}, {"city": "Oslo", "country": "Norway"}
    first, second = _streamed_call(0, "call_a", lisbon), _streamed_call(1, "call_b", oslo)
    interleaved = [delta for pair in zip(first, second) for delta in pair]
    interleaved += first[len(second) :] + second[len(first) :]
    _script(
        second_provider,
        sse(*interleaved, finish_reason="tool_calls"),
        sse({"content": "Both "}, {"content": "sunny."}),
    )
    _, message, _ = _ask_watched(owner, "two cities?", model=SECOND_MODEL, tool_ids=[weather_tool])

    calls = [
        (call["name"], json.loads(call["arguments"])) for call in _items(message, "function_call")
    ]
    assert calls == [("weather", lisbon), ("weather", oslo)]
    tool_round = _sent(second_provider)[1]
    assert _tool_calls_sent_back(tool_round) == [("weather", lisbon), ("weather", oslo)]
    assert _tool_results_sent_back(tool_round) == [
        "Lisbon, Portugal: sunny",
        "Oslo, Norway: sunny",
    ]
    assert message["content"].endswith("Both sunny.")


def test_a_tool_loop_of_several_rounds_keeps_every_round(
    streaming, owner, second_provider, weather_tool
):
    cities = [
        {"city": "Kyoto", "country": "日本"},
        {"city": "Cairo", "country": "مصر"},
        {"city": "Quito", "country": "Ecuador"},
    ]
    rounds = [
        sse(
            {"reasoning_content": f"Round {step} "},
            {"reasoning_content": "next."},
            *_streamed_call(0, f"call_{step}", city),
            finish_reason="tool_calls",
        )
        for step, city in enumerate(cities)
    ]
    _script(second_provider, *rounds, sse({"content": "All "}, {"content": "sunny."}))
    _, message, _ = _ask_watched(
        owner, "three cities?", model=SECOND_MODEL, tool_ids=[weather_tool]
    )

    calls = [json.loads(call["arguments"]) for call in _items(message, "function_call")]
    assert calls == cities
    assert [_text(item) for item in _items(message, "reasoning")] == [
        f"Round {step} next." for step in range(3)
    ]
    assert _tool_results_sent_back(_sent(second_provider)[-1]) == [
        "Kyoto, 日本: sunny",
        "Cairo, مصر: sunny",
        "Quito, Ecuador: sunny",
    ]
    assert message["content"].endswith("All sunny.")


# --- usage, errors and stopping ---------------------------------------------------------------


def test_usage_on_the_last_chunk_is_stored(streaming, owner):
    usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    streaming.upstream.queue(
        reply.text(["counted ", "and ", "stored."], usage=usage, match=reply.answering("count"))
    )
    _, message, _ = _ask_watched(owner, "count")

    assert message["content"] == "counted and stored."
    assert {key: message["usage"][key] for key in usage} == usage


def test_an_error_mid_stream_keeps_the_text_that_arrived(streaming, owner, second_provider):
    _script(
        second_provider,
        sse(
            {"content": "Half "},
            {"content": "an "},
            {"content": "answer"},
            '{"error": "rate limit exceeded"}',
        ),
    )
    _, message, deltas = _ask_watched(owner, "fail halfway", model=SECOND_MODEL)

    assert deltas == [("output_text", "Half "), ("output_text", "an "), ("output_text", "answer")]
    assert message["content"] == "Half an answer"
    assert message["error"]["content"] == "rate limit exceeded"


def test_a_stopped_reply_keeps_whole_pieces(streaming, owner):
    pieces = [f"part-{index} " for index in range(40)]
    streaming.upstream.queue(reply.text(pieces, chunk_delay=0.15, match=reply.answering("go slow")))
    with connected(owner) as socket, owner.client() as client:
        turn = send_message(client, "go slow")
        _wait_for_deltas(socket, turn.chat_id, 3)
        stopped = client.post(f"/api/tasks/chat/{turn.chat_id}/stop")
        message = wait_for_reply(client, turn)
        streamed = "".join(text for _, text in _deltas(socket, turn.chat_id))

    assert stopped.status_code == 200, stopped.text
    kept = _text(message["output"][0])
    assert kept == streamed
    whole_prefixes = ["".join(pieces[:count]) for count in range(3, len(pieces))]
    assert kept in whole_prefixes, kept
    assert message["output"][0]["status"] == "incomplete"


def _wait_for_deltas(socket, chat_id: str, count: int, timeout: float = 30.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while len(_deltas(socket, chat_id)) < count:
        assert time.monotonic() < deadline, f"fewer than {count} deltas arrived"
        time.sleep(0.05)


# --- merged socket deltas -------------------------------------------------------------------


def test_merged_deltas_carry_the_pieces_in_order(streaming, owner, second_provider):
    reasoning = [f"r{index} " for index in range(7)]
    answer = [f"a{index} " for index in range(10)] + ["end."]
    _script(
        second_provider,
        sse(
            *({"reasoning_content": piece} for piece in reasoning),
            *({"content": piece} for piece in answer),
        ),
    )
    _, message, deltas = _ask_watched(
        owner, "batch it", model=SECOND_MODEL, params={"stream_delta_chunk_size": 3}
    )

    assert deltas == [
        ("reasoning_text", "r0 r1 r2 "),
        ("reasoning_text", "r3 r4 r5 "),
        ("reasoning_text", "r6 "),
        ("output_text", "a0 a1 a2 "),
        ("output_text", "a3 a4 a5 "),
        ("output_text", "a6 a7 a8 "),
        ("output_text", "a9 end."),
    ]
    assert _text(_items(message, "reasoning")[0]) == "".join(reasoning)
    assert message["content"] == "".join(answer)


# --- continue and regenerate ----------------------------------------------------------------


def _stored_chat(client, chat_id: str) -> dict:
    stored = client.get(f"/api/v1/chats/{chat_id}")
    stored.raise_for_status()
    return stored.json()["chat"]


def _resend(client, turn, assistant_id: str, messages: list[dict], **extra) -> None:
    """Post a turn for the stored user message again, as the web client does."""
    user_message = _stored_chat(client, turn.chat_id)["history"]["messages"][turn.user_message_id]
    payload = {
        "model": reply.MOCK_MODEL_ID,
        "messages": messages,
        "stream": True,
        "chat_id": turn.chat_id,
        "id": assistant_id,
        "parent_id": user_message["parentId"],
        "user_message": user_message,
        "session_id": "harness-resend",
        "background_tasks": {},
        **extra,
    }
    accepted = client.post("/api/chat/completions", json=payload)
    assert accepted.status_code == 200, accepted.text


def test_continuing_a_reply_extends_the_stored_text(streaming, owner):
    streaming.upstream.queue(
        reply.text(["Once ", "upon ", "a time"], match=reply.answering("tell a tale"))
    )
    turn, first, _ = _ask_watched(owner, "tell a tale")
    assert first["content"] == "Once upon a time"

    streaming.upstream.queue(reply.text([" there ", "was ", "a fox."]))
    history = [
        {"role": "user", "content": "tell a tale"},
        {"role": "assistant", "content": "Once upon a time"},
    ]
    with connected(owner) as socket, owner.client() as client:
        _resend(
            client,
            turn,
            turn.assistant_message_id,
            history,
            assistant_message_id=turn.assistant_message_id,
        )
        socket.wait_for(turn.chat_id, "chat:completion", done=True)
        message = wait_for_reply(client, turn)
        outputs = [
            event["data"]["output"]
            for event in socket.events_of(turn.chat_id)
            if event.get("type") == "chat:completion" and "output" in (event.get("data") or {})
        ]

    assert message["content"] == "Once upon a time there was a fox."
    assert [item["type"] for item in message["output"]] == ["message"]
    streamed_texts = list(dict.fromkeys(_text(output[-1]) for output in outputs))
    assert streamed_texts == [
        "Once upon a time",
        "Once upon a time there ",
        "Once upon a time there was ",
        "Once upon a time there was a fox.",
    ]
    continued = streaming.upstream.chat_requests()[-1]["messages"]
    assert continued[-1] == {"role": "assistant", "content": "Once upon a time"}


def test_regenerating_keeps_both_versions(streaming, owner):
    streaming.upstream.queue(
        reply.text(["First ", "version."], match=reply.answering("say something"))
    )
    turn, _, _ = _ask_watched(owner, "say something")

    streaming.upstream.queue(reply.text(["Second ", "take, ", "better."]))
    regenerated_id = "regenerated-" + turn.assistant_message_id
    with owner.client() as client:
        _resend(client, turn, regenerated_id, [{"role": "user", "content": "say something"}])
        again = type(turn)(turn.chat_id, turn.user_message_id, regenerated_id)
        second = wait_for_reply(client, again)
        messages = _stored_chat(client, turn.chat_id)["history"]["messages"]

    assert second["content"] == "Second take, better."
    assert messages[turn.assistant_message_id]["content"] == "First version."
    assert messages[turn.user_message_id]["childrenIds"] == [
        turn.assistant_message_id,
        regenerated_id,
    ]
    regenerated_request = streaming.upstream.chat_requests()[-1]["messages"]
    assert [entry["role"] for entry in regenerated_request] == ["user"]
