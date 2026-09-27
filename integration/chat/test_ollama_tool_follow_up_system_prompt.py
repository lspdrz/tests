"""Regression: an Ollama model's system prompt was missing from every request after a tool result.

Issue open-webui/open-webui#30161, fix 2f9263540 (PR open-webui/open-webui#30375). With native
function calling, the follow-up after a tool result is rebuilt from the chat's message list and
skips the system prompt step, expecting the first request to have put the prompt there. The
Ollama path added it only to its converted copy, so the follow-up reached Ollama without the
model's instructions. The prompt is now applied before the conversion, once.

Discriminates: passes on dev efe63bd34, fails with 2f9263540 reverted (the follow-up carries no
system message).
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.ollama_provider import OLLAMA_CONFIG, chat_stream, connect_ollama, serve_ollama

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

OLLAMA_MODEL = "llama3:latest"
SYSTEM_PROMPT = "Answer like a pirate."
CHAT_SYSTEM_PROMPT = "Keep it under ten words."
TIMESTAMP_CALL = {"function": {"name": "get_current_timestamp", "arguments": {}}}


@pytest.fixture
def ollama(admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, OLLAMA_MODEL)
    with admin.client() as client:
        connect_ollama(client, listener)
        client.get("/api/models").raise_for_status()
        yield server, client


def _create_preset(client, base_model_id: str, system: str) -> str:
    preset_id = f"prompted-{uuid.uuid4().hex[:8]}"
    form = {
        "id": preset_id,
        "name": preset_id,
        "base_model_id": base_model_id,
        "meta": {},
        "params": {"system": system},
    }
    created = client.post("/api/v1/models/create", json=form)
    assert created.status_code == 200, created.text
    client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    return preset_id


@pytest.fixture
def ollama_preset(ollama):
    server, client = ollama
    preset_id = _create_preset(client, OLLAMA_MODEL, SYSTEM_PROMPT)
    yield server, client, preset_id
    client.post("/api/v1/models/model/delete", json={"id": preset_id})


def _system_messages(sent: dict) -> list[str]:
    return [message["content"] for message in sent["messages"] if message["role"] == "system"]


def _answer_after_one_tool_call(server, client, model: str, **options) -> tuple[dict, dict]:
    server.queue_chat(
        chat_stream(OLLAMA_MODEL, {"tool_calls": [TIMESTAMP_CALL]}),
        chat_stream(OLLAMA_MODEL, {"content": "Arr, it be late."}),
    )
    _, message = ask(client, "what time is it?", model=model, **options)
    assert message["content"] == "Arr, it be late.", message
    first, follow_up = server.chat_requests()[-2:]
    assert follow_up["messages"][-1]["role"] == "tool", "the tool never ran"
    return first, follow_up


def test_the_follow_up_after_a_tool_result_keeps_the_system_prompt(ollama_preset):
    server, client, preset_id = ollama_preset

    _, follow_up = _answer_after_one_tool_call(server, client, preset_id)

    assert _system_messages(follow_up) == [SYSTEM_PROMPT]


def test_every_request_of_the_turn_carries_the_prompt_exactly_once(ollama_preset):
    server, client, preset_id = ollama_preset

    first, follow_up = _answer_after_one_tool_call(server, client, preset_id)

    assert _system_messages(first) == [SYSTEM_PROMPT]
    assert _system_messages(follow_up) == [SYSTEM_PROMPT]
    assert follow_up["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_a_chat_system_prompt_sits_with_the_models_on_the_follow_up(ollama_preset):
    server, client, preset_id = ollama_preset

    first, follow_up = _answer_after_one_tool_call(
        server, client, preset_id, history=[{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    )

    for sent in (first, follow_up):
        [system] = _system_messages(sent)
        assert SYSTEM_PROMPT in system and CHAT_SYSTEM_PROMPT in system, system


def test_a_model_without_a_system_prompt_gets_none(ollama):
    server, client = ollama

    first, follow_up = _answer_after_one_tool_call(server, client, OLLAMA_MODEL)

    assert _system_messages(first) == _system_messages(follow_up) == []


def test_the_openai_path_keeps_the_prompt_on_the_follow_up(admin, upstream):
    with admin.client() as client:
        preset_id = _create_preset(client, reply.MOCK_MODEL_ID, SYSTEM_PROMPT)
        try:
            upstream.queue(
                reply.tool_call("get_current_timestamp", {}), reply.text("Arr, it be late.")
            )
            ask(client, "what time is it?", model=preset_id)
        finally:
            client.post("/api/v1/models/model/delete", json={"id": preset_id})

    first, follow_up = upstream.chat_requests()[-2:]
    assert _system_messages(first) == _system_messages(follow_up) == [SYSTEM_PROMPT]
