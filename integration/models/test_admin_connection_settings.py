"""Journey: a connection's own settings decide which of its models are listed, and how.

What the admin sets on one OpenAI or Ollama connection in Admin Settings > Connections is read
back through `/api/models`: the model allowlist keeps only the listed models, the prefix goes in
front of each id, the tags and the connection type are carried on each model, and a connection
switched off or deleted lists nothing. A key changed on the connection is the one the provider
gets on the next chat. The providers are local stand-ins, and every test puts the connection
settings back afterwards.

Twin of e2e/admin/test_admin_connections.py.

Discriminates: passes on dev 176d31d1d; in a backend copy, the OpenAI model list ignoring
`model_ids` fails the OpenAI listing test (the model left off shows), ignoring `enable` fails the
switch-off test, the chat taking the first connection's key fails the key test (the provider gets
the instance's own key) and the Ollama model list ignoring `model_ids` fails the Ollama listing
test.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.listener import Listener, json_answer
from harness.ollama_provider import OLLAMA_CONFIG, serve_ollama
from harness.second_provider import OPENAI_CONFIG

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

HELLO = [{"role": "user", "content": "hello"}]


def listed_models(client: httpx.Client) -> dict[str, dict]:
    listed = client.get("/api/models", params={"refresh": True})
    assert listed.status_code == 200, listed.text
    return {model["id"]: model for model in listed.json()["data"]}


def serve_openai_models(listener: Listener, *names: str) -> str:
    served = [{"id": name, "object": "model"} for name in names]
    listener.route("GET", "/v1/models", json_answer({"object": "list", "data": served}))
    return f"{listener.base_url}/v1"


def save_openai(client: httpx.Client, urls: list[str], keys: list[str], configs: dict) -> None:
    current = client.get(OPENAI_CONFIG[0]).json()
    changed = {
        **current,
        "OPENAI_API_BASE_URLS": urls,
        "OPENAI_API_KEYS": keys,
        "OPENAI_API_CONFIGS": configs,
    }
    saved = client.post(OPENAI_CONFIG[1], json=changed)
    assert saved.status_code == 200, saved.text


def add_openai(client: httpx.Client, url: str, key: str, **config) -> None:
    current = client.get(OPENAI_CONFIG[0]).json()
    index = str(len(current["OPENAI_API_BASE_URLS"]))
    save_openai(
        client,
        [*current["OPENAI_API_BASE_URLS"], url],
        [*current["OPENAI_API_KEYS"], key],
        {**current["OPENAI_API_CONFIGS"], index: {"enable": True, **config}},
    )


def change_openai(client: httpx.Client, url: str, key: str | None = None, **config) -> None:
    current = client.get(OPENAI_CONFIG[0]).json()
    index = current["OPENAI_API_BASE_URLS"].index(url)
    keys = list(current["OPENAI_API_KEYS"])
    if key is not None:
        keys[index] = key
    configs = dict(current["OPENAI_API_CONFIGS"])
    configs[str(index)] = {**configs.get(str(index), {}), **config}
    save_openai(client, current["OPENAI_API_BASE_URLS"], keys, configs)


def delete_openai(client: httpx.Client, url: str) -> None:
    current = client.get(OPENAI_CONFIG[0]).json()
    urls = current["OPENAI_API_BASE_URLS"]
    kept = [position for position in range(len(urls)) if urls[position] != url]
    save_openai(
        client,
        [current["OPENAI_API_BASE_URLS"][position] for position in kept],
        [current["OPENAI_API_KEYS"][position] for position in kept],
        {
            str(new_index): current["OPENAI_API_CONFIGS"].get(str(position), {})
            for new_index, position in enumerate(kept)
        },
    )


@pytest.fixture
def prefix() -> str:
    return f"conn{uuid.uuid4().hex[:6]}"


def test_the_allowlist_prefix_tags_and_connection_type_shape_the_openai_models(
    admin, preserve, listener, prefix
):
    preserve(OPENAI_CONFIG)
    url = serve_openai_models(listener, "alpha", "beta", "gamma")
    tags = [{"name": f"team-{prefix}"}]
    with admin.client() as client:
        add_openai(
            client,
            url,
            "sk-listing",
            prefix_id=prefix,
            model_ids=["alpha", "beta"],
            tags=tags,
            connection_type="local",
        )
        listed = listed_models(client)

    ours = {model_id: model for model_id, model in listed.items() if prefix in model_id}
    assert sorted(ours) == [f"{prefix}.alpha", f"{prefix}.beta"]
    assert "gamma" not in listed
    for model in ours.values():
        assert model["tags"] == tags
        assert model["connection_type"] == "local"


def test_a_switched_off_or_deleted_openai_connection_lists_nothing(
    admin, preserve, listener, prefix
):
    preserve(OPENAI_CONFIG)
    url = serve_openai_models(listener, "alpha")
    with admin.client() as client:
        add_openai(client, url, "sk-listing", prefix_id=prefix)
        assert f"{prefix}.alpha" in listed_models(client)

        change_openai(client, url, enable=False)
        assert f"{prefix}.alpha" not in listed_models(client), "a switched-off connection listed"

        change_openai(client, url, enable=True)
        assert f"{prefix}.alpha" in listed_models(client)
        delete_openai(client, url)
        assert f"{prefix}.alpha" not in listed_models(client), "a deleted connection listed"


def test_a_changed_key_is_sent_on_the_next_chat(admin, preserve, listener, prefix):
    preserve(OPENAI_CONFIG)
    url = serve_openai_models(listener, "alpha")
    message = {"role": "assistant", "content": "done"}
    completion = {
        "id": "second",
        "object": "chat.completion",
        "model": "alpha",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }
    listener.route("POST", "/v1/chat/completions", json_answer(completion))
    body = {"model": f"{prefix}.alpha", "messages": HELLO, "stream": False}
    with admin.client() as client:
        add_openai(client, url, "sk-before", prefix_id=prefix)
        listed_models(client)
        client.post("/openai/chat/completions", json=body).raise_for_status()
        change_openai(client, url, key="sk-after")
        listed_models(client)
        client.post("/openai/chat/completions", json=body).raise_for_status()

    sent_keys = [
        request.headers.get("Authorization")
        for request in listener.requests_to("/v1/chat/completions")
    ]
    assert sent_keys == ["Bearer sk-before", "Bearer sk-after"]


def test_the_allowlist_prefix_and_tags_shape_the_ollama_models(admin, preserve, listener, prefix):
    preserve(OLLAMA_CONFIG)
    serve_ollama(listener, "llama3:latest", "qwen3:latest")
    tags = [{"name": f"team-{prefix}"}]
    with admin.client() as client:
        current = client.get(OLLAMA_CONFIG[0]).json()
        connected = {
            **current,
            "ENABLE_OLLAMA_API": True,
            "OLLAMA_BASE_URLS": [listener.base_url],
            "OLLAMA_API_CONFIGS": {
                "0": {"prefix_id": prefix, "model_ids": ["llama3:latest"], "tags": tags}
            },
        }
        client.post(OLLAMA_CONFIG[1], json=connected).raise_for_status()
        listed = listed_models(client)

    ours = {model_id: model for model_id, model in listed.items() if prefix in model_id}
    assert list(ours) == [f"{prefix}.llama3:latest"]
    assert "qwen3:latest" not in listed
    assert ours[f"{prefix}.llama3:latest"]["tags"] == tags
    assert ours[f"{prefix}.llama3:latest"]["connection_type"] == "local"
