"""An empty or null tool list on /api/v1/messages broke text-only requests, #31341.

Fix commit `b91a558c9` (open-webui/open-webui#31343). Anthropic clients such as Claude Code send
`"tools": []` on text-only requests. The Anthropic-to-OpenAI converter carried the empty array
into the provider request, which vLLM and the OpenAI API refuse with a 400, and a `"tools": null`
body crashed the converter with a 500. It now forwards tools only when there are some, and a
`tool_choice` only alongside them, since the same backends also refuse a choice without tools.

Discriminates: passes on dev efe63bd34; with b91a558c9 reverted the empty list reaches the
provider, the null list answers 500 and a bare `tool_choice` is forwarded without tools.
"""

from __future__ import annotations

import httpx
import pytest

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Current weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def send_messages(client: httpx.Client, prompt: str, **extra) -> httpx.Response:
    request = {
        "model": MOCK_MODEL_ID,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
        **extra,
    }
    return client.post("/api/v1/messages", json=request)


def provider_request_for(upstream, prompt: str) -> dict:
    matching = [
        body
        for body in upstream.chat_requests()
        if any(message.get("content") == prompt for message in body.get("messages", []))
    ]
    assert len(matching) == 1, matching
    return matching[0]


# ---------------------------------------------------------------- narrow


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"tools": []}, id="empty-tools"),
        pytest.param({"tools": None}, id="null-tools"),
        pytest.param({"tools": [], "tool_choice": {"type": "auto"}}, id="empty-tools-with-choice"),
        pytest.param({"tool_choice": {"type": "any"}}, id="choice-without-tools"),
    ],
)
@pytest.mark.parametrize("stream", [False, True], ids=["batched", "streamed"])
def test_a_text_only_request_sends_the_provider_no_tools(user, upstream, extra, stream):
    prompt = f"just text {sorted(extra)} {stream}"
    upstream.queue(reply.text("plain answer"))

    with user.client() as client:
        response = send_messages(client, prompt, stream=stream, **extra)

    assert response.status_code == 200, response.text
    assert "plain answer" in response.text
    sent = provider_request_for(upstream, prompt)
    assert "tools" not in sent, "an empty tool list reached the provider, which vLLM refuses"
    assert "tool_choice" not in sent, "a tool choice without tools reached the provider"


# ---------------------------------------------------------------- nearby


def test_real_tools_and_their_choice_are_still_forwarded(user, upstream):
    prompt = "what is the weather in Vienna"
    upstream.queue(reply.text("sunny"))

    with user.client() as client:
        response = send_messages(
            client,
            prompt,
            tools=[WEATHER_TOOL],
            tool_choice={"type": "tool", "name": "get_weather"},
        )

    assert response.status_code == 200, response.text
    sent = provider_request_for(upstream, prompt)
    assert [tool["function"]["name"] for tool in sent["tools"]] == ["get_weather"]
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_a_request_without_a_tools_key_is_unchanged(user, upstream):
    prompt = "no tools key at all"
    upstream.queue(reply.text("fine"))

    with user.client() as client:
        response = send_messages(client, prompt)

    assert response.status_code == 200, response.text
    assert response.json()["content"][0]["text"] == "fine"
    assert "tools" not in provider_request_for(upstream, prompt)
