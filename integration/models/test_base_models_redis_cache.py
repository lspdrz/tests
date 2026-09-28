"""The base model list moved into Redis, open-webui v0.11.2 (`6609918bf`).

With the base model cache on, `/api/models` kept the assembled provider list only in the
worker's memory, so every worker of a deployment rebuilt it on its own and a connection change
on one worker left the others serving the old list. The list is now shared through Redis: a
worker reads what another built, writes back what it builds, and a refresh or a connection
change on either provider deletes it.

Two instances share one database and one real Redis, with `ENABLE_BASE_MODELS_CACHE` on. The
provider is a `listener` added as one more OpenAI connection, whose model list the test changes
between calls. Every listing comes from a fresh admin, so the providers' per-account one-second
cache never answers for an earlier call.

Twin of unit/models/test_base_models_redis_cache.py.

Discriminates: passes on dev ef67cc3fa; with 6609918bf reverted (the Redis read, write and
deletes removed from `utils/models.py` and both provider routers) the shared-list, both
connection-change and the refresh tests fail, each second instance serving its own stale list,
and the nearby tests pass.
"""

from __future__ import annotations

import time
from typing import Iterator

import httpx
import pytest

from harness import backends
from harness.actors import create_user
from harness.listener import json_answer, listening
from harness.ollama_provider import OLLAMA_CONFIG
from harness.second_provider import OPENAI_CONFIG, attach

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

FIRST_MODEL = "first-model"
SECOND_MODEL = "second-model"
CACHE_ON = {"ENABLE_BASE_MODELS_CACHE": "true"}
CONNECTIONS_CONFIG = ("/api/v1/configs/connections", "/api/v1/configs/connections")


class Provider:
    """A listener answering `/v1/models` with whatever `serving` holds at the time."""

    def __init__(self, listener) -> None:
        self.listener = listener
        self.serving = [FIRST_MODEL]
        listener.route("GET", "/v1/models", lambda _request: self._models())

    def _models(self):
        return json_answer({"object": "list", "data": [{"id": name} for name in self.serving]})

    def attach_to(self, instance) -> None:
        """Add it as one more OpenAI connection, then let it answer from `serving` again."""
        with instance.client() as client:
            attach(client, self.listener, FIRST_MODEL)
        self.listener.route("GET", "/v1/models", lambda _request: self._models())


def listed_models(instance, refresh: bool = False) -> list[str]:
    """The provider models a fresh admin of `instance` is offered."""
    with create_user(instance, role="admin").client() as client:
        listed = client.get("/api/models", params={"refresh": "true"} if refresh else None)
    assert listed.status_code == 200, listed.text
    offered = [model["id"] for model in listed.json()["data"]]
    return [model_id for model_id in offered if model_id in (FIRST_MODEL, SECOND_MODEL)]


@pytest.fixture(scope="module")
def shared_redis() -> Iterator[str]:
    with backends.redis_server() as url:
        yield url


@pytest.fixture(scope="module")
def provider() -> Iterator[Provider]:
    with listening() as listener:
        yield Provider(listener)


@pytest.fixture(scope="module")
def workers(shared_redis, instance_with, provider) -> tuple:
    """Two instances on one database and one Redis, the provider attached through the first."""
    first = instance_with({**CACHE_ON, "REDIS_URL": shared_redis, "WEBUI_NAME": "worker-a"})
    second = instance_with(
        {
            **CACHE_ON,
            "REDIS_URL": shared_redis,
            "DATABASE_URL": first.database_url,
            "WEBUI_NAME": "worker-b",
        }
    )
    provider.attach_to(first)
    return first, second


@pytest.fixture
def first_model_everywhere(workers, provider) -> tuple:
    """The list built with `FIRST_MODEL` on the first instance, and listed on the second."""
    provider.serving = [FIRST_MODEL]
    first, second = workers
    assert listed_models(first, refresh=True) == [FIRST_MODEL]
    listed_models(second)
    return workers


def _post_current(client: httpx.Client, config: tuple[str, str]) -> None:
    current = client.get(config[0])
    current.raise_for_status()
    client.post(config[1], json=current.json()).raise_for_status()


# ---------------------------------------------------------------- narrow


def test_an_instance_serves_the_list_another_instance_built(first_model_everywhere, provider):
    first, second = first_model_everywhere
    provider.serving = [SECOND_MODEL]
    listed_models(first, refresh=True)
    provider.serving = [FIRST_MODEL, SECOND_MODEL]

    assert listed_models(second) == [SECOND_MODEL], (
        "the second instance rebuilt the list or kept its own, instead of taking the one the "
        "first instance put in Redis"
    )


@pytest.mark.parametrize("config", [OPENAI_CONFIG, OLLAMA_CONFIG], ids=["openai", "ollama"])
def test_a_connection_change_on_one_instance_reaches_the_other(
    first_model_everywhere, provider, config
):
    first, second = first_model_everywhere
    provider.serving = [SECOND_MODEL]

    with first.client() as client:
        _post_current(client, config)

    assert listed_models(second) == [SECOND_MODEL], (
        "a connection change on one instance left another serving the old model list"
    )


def test_a_refresh_on_one_instance_reaches_the_other(first_model_everywhere, provider):
    first, second = first_model_everywhere
    provider.serving = [SECOND_MODEL]

    assert listed_models(first, refresh=True) == [SECOND_MODEL]
    assert listed_models(second) == [SECOND_MODEL], (
        "a refresh on one instance left another serving the old model list"
    )


# ---------------------------------------------------------------- nearby


@pytest.fixture
def lone_instance(instance_with, provider):
    """One instance with the cache on and no Redis, the provider attached."""
    alone = instance_with({**CACHE_ON, "WEBUI_NAME": "alone"})
    provider.serving = [FIRST_MODEL]
    with alone.client() as client:
        current = client.get(OPENAI_CONFIG[0]).json()
    if not any(provider.listener.base_url in url for url in current["OPENAI_API_BASE_URLS"]):
        provider.attach_to(alone)
    return alone


def test_without_redis_an_instance_keeps_its_own_list(lone_instance, provider):
    assert listed_models(lone_instance, refresh=True) == [FIRST_MODEL]
    provider.serving = [SECOND_MODEL]

    assert listed_models(lone_instance) == [FIRST_MODEL]
    assert listed_models(lone_instance, refresh=True) == [SECOND_MODEL]


def test_with_the_cache_off_every_listing_asks_the_provider(workers, provider, preserve):
    first, second = workers
    preserve(CONNECTIONS_CONFIG, on=first)
    with first.client() as client:
        current = client.get(CONNECTIONS_CONFIG[0]).json()
        switched = client.post(
            CONNECTIONS_CONFIG[1], json={**current, "ENABLE_BASE_MODELS_CACHE": False}
        )
    assert switched.status_code == 200, switched.text
    provider.serving = [FIRST_MODEL]
    assert listed_models(second, refresh=True) == [FIRST_MODEL]

    provider.serving = [SECOND_MODEL]
    time.sleep(1.1)  # the providers' per-account cache
    assert listed_models(first) == [SECOND_MODEL]
    assert listed_models(second) == [SECOND_MODEL]


def test_providers_answering_nothing_keep_the_last_list(lone_instance, provider, preserve):
    preserve(CONNECTIONS_CONFIG, on=lone_instance)
    with lone_instance.client() as client:
        current = client.get(CONNECTIONS_CONFIG[0]).json()
        client.post(CONNECTIONS_CONFIG[1], json={**current, "ENABLE_BASE_MODELS_CACHE": False})
    assert listed_models(lone_instance) == [FIRST_MODEL]

    provider.serving = []
    lone_instance.upstream.models = []
    time.sleep(1.1)  # the providers' per-account cache

    assert listed_models(lone_instance) == [FIRST_MODEL], (
        "providers that answered with no models at all wiped the list the instance was serving"
    )
