"""Journey: model providers reached by host name, with the threaded and the c-ares resolver.

`AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` (off by default since c5ec01b1f, PR #28242, after c-ares broke
name resolution in #28013 and #28215) decides which resolver every aiohttp connector Open WebUI
opens uses. Each test drives one provider family by a host name (`localhost`, a hosts-file name for
`::1` alone and a hosts-file name for another local address) and expects the same outcome under both
resolvers: an OpenAI connection verified, listed per connection and in the model list, chatted with
from the web client, from an API client streamed and not, through the Anthropic Messages endpoint
and for embeddings; a Responses API connection chatted with; an Ollama connection verified, listed,
chatted with, embedding, pulling and unloading; a llama.cpp runner's catalog, load and unload. A
provider whose name does not resolve fails verification, chats and a pull at once with the same
status and message, the resolver's own wording aside, and the model list still answers. That
wording is how the last test proves the flag live: only c-ares words a failed lookup its own way.

Discriminates: on dev 176d31d1d, a backend copy whose `env.py` installs a resolver that fails every
lookup when the flag is on turns every c-ares run of the by-name tests and of the liveness test red
and leaves every threaded run green; the same resolver installed for the flag off does the reverse.
Dropping the flag's branch from `env.py` (c-ares always) fails the threaded half of the liveness
test, and reading the flag as always off fails its c-ares half. The unresolvable-name tests go red
under c-ares on a host whose DNS server replays cached answers with a stale EDNS cookie, which
c-ares drops (c-ares issues 1081 and 1271): the lookup times out after about 20 seconds.
"""

from __future__ import annotations

import json

import pytest

from harness.chat import ask
from harness.host_names import (
    C_ARES_REASONS,
    FAILS_WITHIN,
    UNRESOLVABLE,
    name_forms,
    resolver_reason,
    serving_by_name,
    timed,
    without_resolver_reason,
)
from harness.listener import ReceivedRequest, json_answer
from harness.model_runners import connect_runner, serve_llama_cpp
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama
from harness.responses_provider import (
    RESPONSES_MODEL,
    completed,
    connect_responses,
    events_stream,
    message,
)
from harness.second_provider import OPENAI_CONFIG, attach, sse
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

NAMED_MODEL = "named-model"
OLLAMA_MODEL = "named-llama:1b"
RUNNER_MODEL = "named-runner"
REPLY = "answered by name"  # also the Responses API reply, streamed in two deltas
CONNECTION_ERROR = "Open WebUI: Server Connection Error"
UNRESOLVABLE_OLLAMA = f"http://{UNRESOLVABLE}:11434"


def _chat_completion(request: ReceivedRequest):
    if request.json().get("stream"):
        return sse({"content": REPLY})
    choice = {"index": 0, "message": {"role": "assistant", "content": REPLY}}
    return json_answer({"id": "named", "object": "chat.completion", "choices": [choice]})


def _embeddings(request: ReceivedRequest):
    inputs = request.json()["input"]
    rows = [{"index": index, "embedding": [0.5, 0.25]} for index in range(len(inputs))]
    return json_answer({"object": "list", "data": rows})


def _streamed_text(response) -> str:
    """The content a streamed chat completion carried, chunk by chunk."""
    text = ""
    for line in response.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for choice in json.loads(line[len("data: ") :]).get("choices", []):
            text += choice.get("delta", {}).get("content") or ""
    return text


@pytest.mark.parametrize("name_form", name_forms())
def test_an_openai_connection_answers_every_call_by_name(
    resolver, name_form, resolving_instance, resolving_admin, preserve
):
    preserve(OPENAI_CONFIG, on=resolving_instance)
    with serving_by_name(name_form) as provider, resolving_admin.client() as client:
        provider.route("GET", "/v1/models", json_answer({"data": [{"id": NAMED_MODEL}]}))
        provider.route("POST", "/v1/chat/completions", _chat_completion)
        provider.route("POST", "/v1/embeddings", _embeddings)
        verified = client.post(
            "/openai/verify", json={"url": f"{provider.base_url}/v1", "key": "sk-named"}
        )
        attach(client, provider, NAMED_MODEL)
        index = len(client.get(OPENAI_CONFIG[0]).json()["OPENAI_API_BASE_URLS"]) - 1
        listed_on_its_own = client.get(f"/openai/models/{index}")
        _, stored = ask(client, "from the web client", model=NAMED_MODEL)
        request = {"model": NAMED_MODEL, "messages": [{"role": "user", "content": "api"}]}
        whole = client.post("/api/chat/completions", json={**request, "stream": False})
        streamed = client.post("/api/chat/completions", json={**request, "stream": True})
        anthropic = client.post("/api/v1/messages", json={**request, "max_tokens": 16})
        embedded = client.post("/api/embeddings", json={"model": NAMED_MODEL, "input": ["a", "b"]})

    assert verified.status_code == 200, verified.text
    assert [model["id"] for model in verified.json()["data"]] == [NAMED_MODEL]
    assert listed_on_its_own.status_code == 200, listed_on_its_own.text
    assert [model["id"] for model in listed_on_its_own.json()["data"]] == [NAMED_MODEL]
    assert stored["content"] == REPLY, stored
    assert whole.status_code == 200, whole.text
    assert whole.json()["choices"][0]["message"]["content"] == REPLY
    assert streamed.status_code == 200, streamed.text
    assert _streamed_text(streamed) == REPLY
    assert anthropic.status_code == 200, anthropic.text
    assert anthropic.json()["content"][0]["text"] == REPLY
    assert embedded.status_code == 200, embedded.text
    assert [row["embedding"] for row in embedded.json()["data"]] == [[0.5, 0.25]] * 2
    assert len(provider.requests_to("/v1/chat/completions")) == 4


@pytest.mark.parametrize("name_form", name_forms())
def test_an_ollama_connection_answers_every_call_by_name(
    resolver, name_form, resolving_instance, resolving_admin, preserve
):
    preserve(OLLAMA_CONFIG, on=resolving_instance)
    with serving_by_name(name_form) as listener, resolving_admin.client() as client:
        server = serve_ollama(listener, OLLAMA_MODEL)
        server.loaded.append(OLLAMA_MODEL)
        verified = client.post("/ollama/verify", json={"url": listener.base_url, "key": ""})
        connect_ollama(client, listener)
        tags = client.get("/ollama/api/tags")
        listed = client.get("/api/models")
        _, stored = ask(client, "from the web client", model=OLLAMA_MODEL)
        chatted = client.post(
            "/ollama/api/chat",
            json={"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
        embedded = client.post("/ollama/api/embed", json={"model": OLLAMA_MODEL, "input": ["a"]})
        pulled = client.post("/ollama/api/pull", json={"model": "pulled-by-name:1b"})
        unloaded = client.post("/api/models/unload", json={"model": OLLAMA_MODEL})

    assert verified.status_code == 200, verified.text
    assert verified.json() == {"version": server.version}
    assert tags.status_code == 200, tags.text
    assert [model["name"] for model in tags.json()["models"]] == [OLLAMA_MODEL]
    assert OLLAMA_MODEL in [model["id"] for model in listed.json()["data"]]
    assert stored["content"] == "pong", stored
    assert chatted.status_code == 200, chatted.text
    assert '"content": "pong"' in chatted.text or '"content":"pong"' in chatted.text
    assert embedded.status_code == 200, embedded.text
    assert embedded.json()["embeddings"] == [[0.1, 0.2, 0.3]]
    assert pulled.status_code == 200 and '"status": "success"' in pulled.text, pulled.text
    assert unloaded.status_code == 200, unloaded.text
    assert server.loaded == []


@pytest.mark.parametrize("name_form", name_forms())
def test_a_model_runner_is_managed_by_name(
    resolver, name_form, resolving_instance, resolving_admin, preserve
):
    preserve(OPENAI_CONFIG, on=resolving_instance)
    with serving_by_name(name_form) as listener, resolving_admin.client() as client:
        runner = serve_llama_cpp(listener, RUNNER_MODEL)
        index = connect_runner(client, runner)
        catalog = client.get(f"/openai/models/{index}/catalog")
        loaded = client.post(f"/openai/models/{index}/load", json={"model": RUNNER_MODEL})
        unloaded = client.post(f"/openai/models/{index}/unload", json={"model": RUNNER_MODEL})

    assert catalog.status_code == 200, catalog.text
    assert [model["id"] for model in catalog.json()["data"]] == [RUNNER_MODEL]
    assert loaded.json() == {"success": True}, loaded.text
    assert unloaded.json() == {"success": True}, unloaded.text
    assert runner.sent("/models/load") == [{"model": RUNNER_MODEL}]


@pytest.mark.parametrize("name_form", name_forms())
def test_a_responses_api_connection_answers_by_name(
    resolver, name_form, resolving_instance, resolving_admin, preserve
):
    preserve(OPENAI_CONFIG, on=resolving_instance)
    with serving_by_name(name_form) as listener, resolving_admin.client() as client:
        provider = connect_responses(client, listener)
        provider.answer(events_stream(*message("answered ", "by name"), completed()))
        _, stored = ask(client, "through the Responses API?", model=RESPONSES_MODEL)

    assert stored["content"] == REPLY, stored
    assert len(provider.sent()) == 1


def _add_unresolvable_openai(client) -> None:
    """An OpenAI connection by a name no resolver answers, its model named in its config."""
    connections = client.get(OPENAI_CONFIG[0]).json()
    index = str(len(connections["OPENAI_API_BASE_URLS"]))
    updated = {
        **connections,
        "OPENAI_API_BASE_URLS": [
            *connections["OPENAI_API_BASE_URLS"],
            f"http://{UNRESOLVABLE}:8000/v1",
        ],
        "OPENAI_API_KEYS": [*connections["OPENAI_API_KEYS"], "sk-gone"],
        "OPENAI_API_CONFIGS": {
            **connections["OPENAI_API_CONFIGS"],
            index: {"enable": True, "model_ids": [NAMED_MODEL]},
        },
    }
    client.post(OPENAI_CONFIG[1], json=updated).raise_for_status()


def _enable_unresolvable_ollama(client) -> None:
    current = client.get(OLLAMA_CONFIG[0]).json()
    connection = {**current, "ENABLE_OLLAMA_API": True, "OLLAMA_BASE_URLS": [UNRESOLVABLE_OLLAMA]}
    client.post(OLLAMA_CONFIG[1], json=connection).raise_for_status()


def test_an_unresolvable_provider_fails_verification_the_same_way(resolver, resolving_admin):
    with resolving_admin.client() as client:
        openai_verified, openai_seconds = timed(
            client.post,
            "/openai/verify",
            json={"url": f"{UNRESOLVABLE_OLLAMA}/v1", "key": "sk-gone"},
        )
        ollama_verified, ollama_seconds = timed(
            client.post, "/ollama/verify", json={"url": UNRESOLVABLE_OLLAMA, "key": ""}
        )

    assert (openai_verified.status_code, openai_verified.json()) == (
        500,
        {"detail": CONNECTION_ERROR},
    ), f"after {openai_seconds:.1f}s"
    assert (ollama_verified.status_code, ollama_verified.json()) == (
        500,
        {"detail": CONNECTION_ERROR},
    ), f"after {ollama_seconds:.1f}s"
    slowest = max(openai_seconds, ollama_seconds)
    assert slowest < FAILS_WITHIN, f"{resolver} took {slowest:.1f}s to refuse an unknown name"


def test_an_unresolvable_provider_leaves_the_model_list_and_fails_its_chats(
    resolver, resolving_instance, resolving_admin, preserve
):
    preserve(OPENAI_CONFIG, on=resolving_instance)
    request = {"model": NAMED_MODEL, "messages": [{"role": "user", "content": "anyone?"}]}
    with resolving_admin.client() as client:
        _add_unresolvable_openai(client)
        listed, listing_seconds = timed(client.get, "/api/models")
        whole, chat_seconds = timed(
            client.post, "/api/chat/completions", json={**request, "stream": False}
        )
        _, stored = ask(client, "anyone there?", model=NAMED_MODEL)

    assert listed.status_code == 200, listed.text
    listed_ids = [model["id"] for model in listed.json()["data"]]
    assert MOCK_MODEL_ID in listed_ids and NAMED_MODEL in listed_ids
    assert (whole.status_code, whole.json()) == (400, {"detail": CONNECTION_ERROR})
    assert stored["error"] == {"content": CONNECTION_ERROR}, stored
    slowest = max(listing_seconds, chat_seconds)
    assert slowest < FAILS_WITHIN, f"{resolver} took {slowest:.1f}s to refuse an unknown name"


def test_an_unresolvable_ollama_names_the_host_it_could_not_reach(
    resolver, resolving_instance, resolving_admin, preserve
):
    preserve(OLLAMA_CONFIG, on=resolving_instance)
    with resolving_admin.client() as client:
        _enable_unresolvable_ollama(client)
        pulled, seconds = timed(client.post, "/ollama/api/pull/0", json={"model": OLLAMA_MODEL})

    assert pulled.status_code == 500, pulled.text
    assert without_resolver_reason(pulled.json()["detail"]) == (
        f"Ollama: Cannot connect to host {UNRESOLVABLE}:11434 ssl:default [<resolver reason>]"
    )
    assert seconds < FAILS_WITHIN, f"{resolver} took {seconds:.1f}s to refuse an unknown name"


def test_the_flag_hands_name_resolution_to_c_ares(
    resolver, resolving_instance, resolving_admin, preserve
):
    """The one thing that tells the resolvers apart from outside: their wording of a failure."""
    preserve(OLLAMA_CONFIG, on=resolving_instance)
    with resolving_admin.client() as client:
        _enable_unresolvable_ollama(client)
        pulled = client.post("/ollama/api/pull/0", json={"model": OLLAMA_MODEL})

    reason = resolver_reason(pulled.json()["detail"])
    assert reason, f"the error names no resolver reason: {pulled.text}"
    if resolver == "c-ares":
        assert reason in C_ARES_REASONS, f"the flag is on but the OS resolver answered: {reason}"
    else:
        assert reason not in C_ARES_REASONS, f"the flag is off but c-ares answered: {reason}"
