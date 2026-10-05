"""Journey: unloading a model shows as unloaded in the model list straight away.

The model selector marks the models a runner has loaded. Unloading one, through the selector's
`/api/models/unload` for Ollama or llama.cpp or through Ollama's own `/ollama/api/unload`, now
drops the cached model list, so the next `/api/models` reads the runner again and shows the
model unloaded instead of repeating the cached answer.

Discriminates: in a backend copy with the cache clear taken out of the three unload paths, each
test reads `loaded: true` for a model the runner has already freed.
"""

from __future__ import annotations

import pytest

from harness.model_runners import connect_runner, serve_llama_cpp
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama
from harness.second_provider import OPENAI_CONFIG

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

OLLAMA_MODEL = "llama3:latest"
LLAMA_MODEL = "qwen3-0.6b"


def _loaded(client, model_id: str) -> bool | None:
    listed = client.get("/api/models")
    assert listed.status_code == 200, listed.text
    return {entry["id"]: entry.get("loaded") for entry in listed.json()["data"]}[model_id]


@pytest.fixture
def ollama(admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, OLLAMA_MODEL)
    server.loaded.append(OLLAMA_MODEL)
    with admin.client() as client:
        connect_ollama(client, listener)
        yield server, client


def test_the_selectors_unload_of_an_ollama_model_refreshes_the_list(ollama):
    server, client = ollama
    assert _loaded(client, OLLAMA_MODEL) is True

    unloaded = client.post("/api/models/unload", json={"model": OLLAMA_MODEL})

    assert unloaded.status_code == 200, unloaded.text
    assert server.loaded == []
    assert _loaded(client, OLLAMA_MODEL) is False, "the model list still showed it loaded"


def test_ollamas_own_unload_refreshes_the_list(ollama):
    server, client = ollama
    assert _loaded(client, OLLAMA_MODEL) is True

    unloaded = client.post("/ollama/api/unload", json={"model": OLLAMA_MODEL})

    assert unloaded.status_code == 200, unloaded.text
    assert server.loaded == []
    assert _loaded(client, OLLAMA_MODEL) is False, "the model list still showed it loaded"


def test_the_selectors_unload_of_a_llama_cpp_model_refreshes_the_list(admin, preserve, listener):
    preserve(OPENAI_CONFIG)
    runner = serve_llama_cpp(listener, LLAMA_MODEL)
    runner.loaded.append(LLAMA_MODEL)
    with admin.client() as client:
        connect_runner(client, runner)
        assert _loaded(client, LLAMA_MODEL) is True

        unloaded = client.post("/api/models/unload", json={"model": LLAMA_MODEL})

        assert unloaded.status_code == 200, unloaded.text
        assert runner.loaded == []
        assert _loaded(client, LLAMA_MODEL) is False, "the model list still showed it loaded"
