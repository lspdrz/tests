"""Regression: ejecting a model from the selector ignored the setting that relaxes TLS checks.

Issue open-webui/open-webui#31371, fix PR open-webui/open-webui#31391. A llama.cpp or Ollama
connection served over HTTPS with a certificate the system does not trust worked for chat on an
instance started with `AIOHTTP_CLIENT_SESSION_SSL=false`, but the unload the model selector sends
failed with a certificate error. Both runners here answer over HTTPS with a self-signed
certificate; the unload must reach them and free the model, and chat through the same connection
must still work.

Discriminates: passes on dev a5bc78300; fails with fab58bd35 reverted (the unload of a llama.cpp
model and of an Ollama model both answer 500 with a certificate error and the runner is never
called), the chat tests pass on both.
"""

from __future__ import annotations

import pytest

from harness import second_provider
from harness.chat import send_message, wait_for_reply
from harness.listener import listening
from harness.model_runners import connect_runner, serve_llama_cpp
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

MODEL = "tiny-model"


@pytest.fixture
def relaxed(instance_with):
    return instance_with({"AIOHTTP_CLIENT_SESSION_SSL": "false"})


@pytest.fixture
def https_listener():
    with listening(tls=True) as service:
        yield service


def _refresh_models(client) -> None:
    client.get("/api/models", params={"refresh": "true"}).raise_for_status()


def test_a_llama_cpp_model_is_unloaded_over_https_with_an_untrusted_certificate(
    relaxed, https_listener, preserve
):
    runner = serve_llama_cpp(https_listener, MODEL)
    runner.loaded.append(MODEL)
    preserve(second_provider.OPENAI_CONFIG, on=relaxed)
    with relaxed.client() as client:
        connect_runner(client, runner)
        _refresh_models(client)
        unloaded = client.post("/api/models/unload", json={"model": MODEL})

    assert unloaded.status_code == 200, unloaded.text
    assert runner.sent("/models/unload") == [{"model": MODEL}]
    assert runner.loaded == []


def test_an_ollama_model_is_unloaded_over_https_with_an_untrusted_certificate(
    relaxed, https_listener, preserve
):
    server = serve_ollama(https_listener, MODEL)
    server.loaded.append(MODEL)
    preserve(OLLAMA_CONFIG, on=relaxed)
    with relaxed.client() as client:
        connect_ollama(client, https_listener)
        _refresh_models(client)
        unloaded = client.post("/api/models/unload", json={"model": MODEL})

    assert unloaded.status_code == 200, unloaded.text
    assert [call["model"] for call in server.sent("/api/generate")] == [MODEL]
    assert server.loaded == []


def test_chat_reaches_a_llama_cpp_runner_over_https_with_an_untrusted_certificate(
    relaxed, https_listener, preserve
):
    runner = serve_llama_cpp(https_listener, MODEL)
    https_listener.route(
        "POST", "/v1/chat/completions", second_provider.sse({"content": "served over https"})
    )
    preserve(second_provider.OPENAI_CONFIG, on=relaxed)
    with relaxed.client() as client:
        connect_runner(client, runner)
        _refresh_models(client)
        reply = wait_for_reply(client, send_message(client, "hello", model=MODEL))

    assert "served over https" in reply["content"], reply


def test_chat_reaches_an_ollama_server_over_https_with_an_untrusted_certificate(
    relaxed, https_listener, preserve
):
    serve_ollama(https_listener, MODEL)
    preserve(OLLAMA_CONFIG, on=relaxed)
    with relaxed.client() as client:
        connect_ollama(client, https_listener)
        _refresh_models(client)
        reply = wait_for_reply(client, send_message(client, "hello", model=MODEL))

    assert "pong" in reply["content"], reply
