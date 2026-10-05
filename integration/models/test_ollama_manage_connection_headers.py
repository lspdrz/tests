"""Regression: managing the models of one Ollama connection ignored its headers and sign-in.

Fix commit `00a245b9f` (open-webui/open-webui#31489, issue open-webui/open-webui#31487). The
admin's Manage Ollama dialog works on a single connection: it lists that connection's models,
pulls, creates and deletes them, and the same per-connection routes check the version, copy and
unload. Those requests went out with the connection's key as a bearer token and nothing else, so
the custom headers a gateway such as Cloudflare Access asks for never arrived and a connection set
to authentication "none" still sent its key. Verifying and chatting had applied both since
`bc2416c5d` (open-webui/open-webui#29868, covered in test_ollama_connection_headers.py).

The Ollama stand-in records every request, so each test reads what the dialog's calls sent. The
dialog's model file upload and the model selector's unload were left out of that fix and followed
in `e8d6a8734` (open-webui/open-webui#31490).

Discriminates: passes on dev 176d31d1d; on dev 00a245b9f the two #31490 tests fail, and they pass
with that PR applied; with the fix reverted in `routers/ollama.py` every narrow row and the broad
test fail (no custom header, and the key goes out as a bearer token despite `auth_type: none`, or
no key at all on the version check); the other nearby tests pass on both.
"""

from __future__ import annotations

import hashlib

import pytest

from harness.listener import ReceivedRequest, json_answer
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, ndjson, serve_ollama

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

BASE_MODEL = "llama3:latest"
GATEWAY_HEADER = "CF-Access-Client-Id"
GATEWAY_VALUE = "gateway-client.access"
IGNORED_KEY = "sk-not-for-the-gateway"
GATED_WITHOUT_KEY = {
    "headers": {GATEWAY_HEADER: GATEWAY_VALUE},
    "auth_type": "none",
    "key": IGNORED_KEY,
}

# (method, Open WebUI route, body, path the Ollama server is asked for)
DIALOG_CALLS = {
    "list": ("GET", "/ollama/api/tags/0", None, "/api/tags"),
    "version": ("GET", "/ollama/api/version/0", None, "/api/version"),
    "pull": ("POST", "/ollama/api/pull/0", {"name": "qwen3:0.6b"}, "/api/pull"),
    "create": (
        "POST",
        "/ollama/api/create/0",
        {"model": "brief:latest", "from": BASE_MODEL, "system": "Answer in one line."},
        "/api/create",
    ),
    "copy": (
        "POST",
        "/ollama/api/copy/0",
        {"source": BASE_MODEL, "destination": "backup:latest"},
        "/api/copy",
    ),
    "unload": ("POST", "/ollama/api/unload", {"model": BASE_MODEL}, "/api/generate"),
    "delete": ("DELETE", "/ollama/api/delete/0", {"model": BASE_MODEL}, "/api/delete"),
}
ROUTES_WITH_A_KEY_BEFORE = [name for name in DIALOG_CALLS if name != "version"]


def header(request: ReceivedRequest, name: str) -> str | None:
    by_name = {key.lower(): value for key, value in request.headers.items()}
    return by_name.get(name.lower())


def last_request_to(server, path: str) -> ReceivedRequest:
    requests = server.listener.requests_to(path)
    assert requests, f"the Ollama server was never asked for {path}"
    return requests[-1]


def call(client, name: str):
    method, route, body, _ = DIALOG_CALLS[name]
    response = client.request(method, route, json=body)
    assert response.status_code == 200, f"{name}: {response.status_code} {response.text}"
    return response


@pytest.fixture
def operator(make_user):
    """A fresh admin, so no model list cached for another account answers in its place."""
    return make_user(role="admin")


@pytest.fixture
def ollama(preserve, listener):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, BASE_MODEL)
    server.loaded.append(BASE_MODEL)
    listener.route("POST", "/api/push", lambda _request: ndjson({"status": "success"}))
    listener.route("POST", "/api/embeddings", lambda _request: json_answer({"embedding": [0.1]}))
    return server


def connect(operator, server, **config) -> None:
    with operator.client() as client:
        connect_ollama(client, server.listener, **config)


# --- narrow: each call of the dialog follows the connection's settings ------------------------


@pytest.mark.parametrize("name", list(DIALOG_CALLS))
def test_a_dialog_call_sends_the_headers_and_no_key_with_auth_none(operator, ollama, name):
    connect(operator, ollama, **GATED_WITHOUT_KEY)

    with operator.client() as client:
        call(client, name)

    sent = last_request_to(ollama, DIALOG_CALLS[name][3])
    assert (header(sent, GATEWAY_HEADER), header(sent, "Authorization")) == (GATEWAY_VALUE, None)


def test_the_version_check_of_one_connection_sends_its_key(operator, ollama):
    connect(operator, ollama, key="sk-ollama", auth_type="bearer")

    with operator.client() as client:
        call(client, "version")

    sent = last_request_to(ollama, "/api/version")
    assert header(sent, "Authorization") == "Bearer sk-ollama"


# --- broad: a whole dialog session and the proxy calls, nothing ungated -----------------------


def test_managing_a_gated_connection_never_bypasses_the_gateway(operator, ollama):
    connect(operator, ollama, **GATED_WITHOUT_KEY)

    with operator.client() as client:
        for name in DIALOG_CALLS:
            call(client, name)
        proxied = [
            client.request("DELETE", "/ollama/api/push/0", json={"model": "backup:latest"}),
            client.post("/ollama/api/show", json={"name": "backup:latest"}),
            client.post("/ollama/api/embed", json={"model": "backup:latest", "input": "a"}),
            client.post("/ollama/api/embeddings", json={"model": "backup:latest", "prompt": "a"}),
            client.post(
                "/ollama/api/generate",
                json={"model": "backup:latest", "prompt": "hi", "stream": False},
            ),
            client.get("/ollama/v1/models/0"),
        ]
    assert [response.status_code for response in proxied] == [200] * len(proxied), [
        response.text for response in proxied
    ]

    leaking = [
        f"{request.method} {request.path}"
        for request in ollama.listener.received
        if header(request, GATEWAY_HEADER) != GATEWAY_VALUE
        or header(request, "Authorization") is not None
    ]
    assert not leaking, f"requests without the gateway header or with the key: {leaking}"


# --- nearby: bearer keys and plain connections keep working -----------------------------------


@pytest.mark.parametrize("name", ROUTES_WITH_A_KEY_BEFORE)
def test_a_dialog_call_still_sends_the_bearer_key(operator, ollama, name):
    connect(operator, ollama, key="sk-ollama", auth_type="bearer")

    with operator.client() as client:
        call(client, name)

    sent = last_request_to(ollama, DIALOG_CALLS[name][3])
    assert header(sent, "Authorization") == "Bearer sk-ollama"


def test_a_connection_without_settings_still_lists_and_pulls(operator, ollama):
    connect(operator, ollama)

    with operator.client() as client:
        listed = call(client, "list")
        call(client, "pull")

    assert [model["name"] for model in listed.json()["models"]] == [BASE_MODEL]
    assert "qwen3:0.6b" in ollama.models
    assert all(header(request, "Authorization") is None for request in ollama.listener.received)
    assert all(header(request, GATEWAY_HEADER) is None for request in ollama.listener.received)


# --- the two paths open-webui/open-webui#31490 fixes ----------------------------------------


def test_uploading_a_model_file_sends_the_headers(operator, ollama):
    connect(operator, ollama, **GATED_WITHOUT_KEY)
    content = b"GGUF tiny model"
    blob_path = f"/api/blobs/sha256:{hashlib.sha256(content).hexdigest()}"

    with operator.client() as client:
        uploaded = client.post(
            "/ollama/models/upload/0",
            files={"file": ("tiny.gguf", content, "application/octet-stream")},
        )
    assert uploaded.status_code == 200, uploaded.text
    assert "model_created" in uploaded.text, uploaded.text

    for path in (blob_path, "/api/create"):
        assert header(last_request_to(ollama, path), GATEWAY_HEADER) == GATEWAY_VALUE, path


def test_unloading_from_the_model_selector_follows_the_connection(operator, ollama):
    connect(operator, ollama, **GATED_WITHOUT_KEY)

    with operator.client() as client:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        unloaded = client.post("/api/models/unload", json={"model": BASE_MODEL})
    assert unloaded.status_code == 200, unloaded.text

    sent = last_request_to(ollama, "/api/generate")
    assert (header(sent, GATEWAY_HEADER), header(sent, "Authorization")) == (GATEWAY_VALUE, None)
