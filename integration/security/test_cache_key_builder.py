"""Regression: the provider model lists are cached per user, not once for everyone.

`routers/openai.py` and `routers/ollama.py` shipped `@cached(key=lambda _, user: ...)` on
`get_all_models`. In aiocache 0.12 `key=` is used verbatim as the cache key and never called,
so every caller shared one entry for the whole TTL and one user's permission-filtered model
list could be served to another. The fix switched both to `key_builder=` (branch
fix/cached-key-builder-per-user-models). Those two are the only `@cached` lists in the backend.

With a five-minute cache, a second account's first model listing must reach the provider
itself, while a repeat listing by the same account is served from its own cache entry. The
OpenAI list is read from the scripted provider, the Ollama list from an Ollama stand-in.

Discriminates: passes on dev ef67cc3fa; with `key_builder=` turned back into `key=` in
routers/openai.py the second account's OpenAI listing is served from the first account's entry
and the provider is never asked, and the same edit in routers/ollama.py does that to the Ollama
listing.
"""

from __future__ import annotations

import pytest

from harness.actors import admin_of, create_user
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]


@pytest.fixture
def cached_models(instance_with):
    return instance_with({"MODELS_CACHE_TTL": "300"})


def _provider_listings_during(launched, client) -> int:
    before = len(launched.upstream.requests_to("/models"))
    client.get("/api/models").raise_for_status()
    return len(launched.upstream.requests_to("/models")) - before


def test_each_account_gets_its_own_cached_model_list(cached_models):
    with cached_models.client() as admin_client:
        _provider_listings_during(cached_models, admin_client)
        assert _provider_listings_during(cached_models, admin_client) == 0, (
            "the model list is not cached at all, so this test proves nothing"
        )

    with create_user(cached_models).client() as user_client:
        user_listings = _provider_listings_during(cached_models, user_client)
        repeat_listings = _provider_listings_during(cached_models, user_client)

    assert user_listings == 1, (
        "a second account's model listing was served from the first account's cache entry, "
        "so one user's filtered model list reaches another (static aiocache key)"
    )
    assert repeat_listings == 0


def test_each_account_gets_its_own_cached_ollama_model_list(cached_models, preserve, listener):
    preserve(OLLAMA_CONFIG, on=cached_models)
    serve_ollama(listener, "llama3:latest")
    with admin_of(cached_models).client() as admin_client:
        connect_ollama(admin_client, listener)

    def ollama_listings_during(client) -> int:
        before = len(listener.requests_to("/api/tags"))
        client.get("/api/models").raise_for_status()
        return len(listener.requests_to("/api/tags")) - before

    first, second = create_user(cached_models), create_user(cached_models)
    with first.client() as first_client, second.client() as second_client:
        first_listings = ollama_listings_during(first_client)
        repeat_listings = ollama_listings_during(first_client)
        second_listings = ollama_listings_during(second_client)

    assert first_listings == 1, "the Ollama server was never asked for its models"
    assert repeat_listings == 0, "the Ollama model list is not cached, so this proves nothing"
    assert second_listings == 1, (
        "a second account's Ollama model listing was served from the first account's cache "
        "entry, so one user's filtered model list reaches another (static aiocache key)"
    )
