"""Dependency smoke: how Open WebUI reaches its model providers, each library through its feature.

aiohttp carries every call to a provider, and its `ClientTimeout` is what cuts a connection
that is slow to list its models loose at `AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST`, so the model list
still answers with the others. aiocache's `@cached` keeps each account's provider model list for
`MODELS_CACHE_TTL` seconds (one by default) and asks the provider again once that runs out; the
per-account key is pinned in integration/security/test_cache_key_builder.py. With
`AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` on, aiohttp resolves host names through aiodns (c-ares)
instead of the threaded resolver, so that instance reaches a provider by the name `localhost`.
That aiodns stays off by default is pinned in unit/deps/test_aiodns.py, since no request shows
which resolver answered.

Discriminates: passes on dev ef67cc3fa; in a backend copy, `ttl=None` on the OpenAI model list
cache never asks the provider again, the model list's `ClientTimeout(total=None)` waits out the
slow connection, and an aiodns whose `getaddrinfo` fails every lookup leaves the provider
unreached by name on the async DNS instance.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.listener import json_answer
from harness.second_provider import OPENAI_CONFIG, attach, sse

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

MODEL_LIST_TIMEOUT = {"AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST": "1"}
ASYNC_DNS = {"AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER": "true"}
SLOW_MODEL = "slow-model"
SLOW_LISTING_SECONDS = 8
NAMED_MODEL = "named-model"


def _provider_listings_during(provider, client) -> int:
    before = len(provider.requests_to("/models"))
    listed = client.get("/api/models")
    assert listed.status_code == 200, listed.text
    return len(provider.requests_to("/models")) - before


def test_a_model_listing_is_cached_until_its_ttl_runs_out(make_user, upstream):
    account = make_user()
    with account.client() as client:
        first = _provider_listings_during(upstream, client)
        repeated = _provider_listings_during(upstream, client)
        time.sleep(2)  # past the default one-second MODELS_CACHE_TTL
        after_expiry = _provider_listings_during(upstream, client)

    assert first == 1, "the account's first model listing never reached the provider"
    assert repeated == 0, "a repeat listing inside the TTL was not served from the cache"
    assert after_expiry == 1, "the cached model list outlived its TTL"


def _add_connection(client, base_url: str) -> None:
    """Append a connection the way the admin's connection list saves it, without listing."""
    current = client.get(OPENAI_CONFIG[0])
    current.raise_for_status()
    connections = current.json()
    index = str(len(connections["OPENAI_API_BASE_URLS"]))
    updated = {
        **connections,
        "OPENAI_API_BASE_URLS": [*connections["OPENAI_API_BASE_URLS"], base_url],
        "OPENAI_API_KEYS": [*connections["OPENAI_API_KEYS"], "sk-slow"],
        "OPENAI_API_CONFIGS": {**connections["OPENAI_API_CONFIGS"], index: {"enable": True}},
    }
    client.post(OPENAI_CONFIG[1], json=updated).raise_for_status()


def _slow_model_list(_request):
    time.sleep(SLOW_LISTING_SECONDS)
    return json_answer({"object": "list", "data": [{"id": SLOW_MODEL}]})


@pytest.mark.slow
def test_a_slow_connection_is_cut_off_at_the_model_list_timeout(instance_with, preserve, listener):
    impatient = instance_with(MODEL_LIST_TIMEOUT)
    preserve(OPENAI_CONFIG, on=impatient)
    listener.route("GET", "/v1/models", _slow_model_list)

    with admin_of(impatient).client() as client:
        _add_connection(client, f"{listener.base_url}/v1")
        time.sleep(1.5)  # let the admin's cached model list expire
        started = time.monotonic()
        listed = client.get("/api/models")
        elapsed = time.monotonic() - started

    assert listed.status_code == 200, listed.text
    model_ids = {model["id"] for model in listed.json()["data"]}
    assert listener.requests_to("/v1/models"), "the slow connection was never asked"
    assert elapsed < SLOW_LISTING_SECONDS - 3, f"the model list waited {elapsed:.1f}s"
    assert reply.MOCK_MODEL_ID in model_ids
    assert SLOW_MODEL not in model_ids


@pytest.mark.slow
def test_with_async_dns_a_provider_is_reached_by_its_host_name(instance_with, preserve, listener):
    async_dns = instance_with(ASYNC_DNS)
    preserve(OPENAI_CONFIG, on=async_dns)
    by_name = dataclasses.replace(listener, base_url=f"http://localhost:{listener.port}")
    listener.route("POST", "/v1/chat/completions", sse({"content": "resolved by name"}))

    with admin_of(async_dns).client() as client:
        attach(client, by_name, NAMED_MODEL)
        _, message = ask(client, "hello?", model=NAMED_MODEL)

    assert listener.requests_to("/v1/chat/completions"), "the provider was never called"
    assert message["content"] == "resolved by name"
