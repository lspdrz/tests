"""Journey: outbound calls behind an HTTP proxy, with the threaded and the c-ares resolver.

Open WebUI's aiohttp sessions honour `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY`. Behind a proxy
the only name the instance resolves is the proxy's own, so the proxy is named `localhost` here and
the resolver `AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` picks resolves it; the provider's name is one only
the proxy knows (under `.invalid`, which no local resolver answers), so a call that tried to resolve
it locally would fail. Under both resolvers an OpenAI and an Ollama connection behind the proxy
are verified, listed and chatted with through it, a name in `NO_PROXY` is reached directly by
each name form and never shown to the proxy, a name the proxy cannot resolve fails the same way,
and a proxy whose own name does not resolve fails every call the same way.

Discriminates: on dev 176d31d1d, a backend copy whose `env.py` installs a resolver that fails every
lookup when the flag is on turns every c-ares run but the unresolvable-proxy one red (the proxy's
own name no longer resolves) and leaves every threaded run green; the same resolver installed for
the flag off does the reverse. The unresolvable-proxy test stays green under a failing resolver by
design, and a resolver that takes 20 seconds to refuse an unknown name turns it red; its c-ares run
skips, naming the reason, on a machine whose DNS server keeps c-ares from refusing an unknown name
at once (a cached reply with a stale EDNS cookie, c-ares issues 1081 and 1271).
"""

from __future__ import annotations

import dataclasses

import pytest

from harness.actors import admin_of
from harness.chat import ask
from harness.host_names import (
    FAILS_WITHIN,
    RESOLVERS,
    UNRESOLVABLE,
    find_name_form,
    name_forms,
    serving_by_name,
    timed,
)
from harness.http_proxy import serving_proxy
from harness.listener import listening
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama
from harness.second_provider import OPENAI_CONFIG, attach, sse

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

BEHIND_PROXY = "models.behind-proxy.invalid"
UNKNOWN_TO_PROXY = "unknown.behind-proxy.invalid"
PROXIED_MODEL = "proxied-model"
OLLAMA_MODEL = "proxied-llama:1b"
REPLY = "answered through the proxy"
CONNECTION_ERROR = "Open WebUI: Server Connection Error"


def _direct_names() -> str:
    forms = [find_name_form(label) for label in ("ipv6-only", "hosts-file")]
    return ",".join(["127.0.0.1", "localhost", *(form.host for form in forms if form)])


@pytest.fixture(scope="module")
def proxy():
    with serving_proxy() as serving:
        yield serving


@pytest.fixture
def proxied_instance(resolver, proxy, instance_with):
    proxy_url = f"http://localhost:{proxy.port}"
    return instance_with(
        {
            **RESOLVERS[resolver],
            "HTTP_PROXY": proxy_url,
            "HTTPS_PROXY": proxy_url,
            "NO_PROXY": _direct_names(),
        }
    )


def test_a_provider_only_the_proxy_can_name_is_reached_through_it(
    resolver, proxy, proxied_instance, preserve
):
    preserve(OPENAI_CONFIG, OLLAMA_CONFIG, on=proxied_instance)
    with listening() as provider, listening() as ollama_listener:
        proxy.known[BEHIND_PROXY] = provider.port
        proxy.known[f"ollama.{BEHIND_PROXY}"] = ollama_listener.port
        provider.route("POST", "/v1/chat/completions", sse({"content": REPLY}))
        named = dataclasses.replace(provider, base_url=f"http://{BEHIND_PROXY}:8000")
        ollama = serve_ollama(
            dataclasses.replace(ollama_listener, base_url=f"http://ollama.{BEHIND_PROXY}:11434"),
            OLLAMA_MODEL,
        )
        with admin_of(proxied_instance).client() as client:
            attach(client, named, PROXIED_MODEL)
            verified = client.post(
                "/openai/verify", json={"url": f"{named.base_url}/v1", "key": "sk-proxied"}
            )
            _, stored = ask(client, "through the proxy?", model=PROXIED_MODEL)
            connect_ollama(client, ollama.listener)
            ollama_verified = client.post(
                "/ollama/verify", json={"url": ollama.listener.base_url, "key": ""}
            )
            tags = client.get("/ollama/api/tags")
            _, ollama_stored = ask(client, "and ollama?", model=OLLAMA_MODEL)

    assert verified.status_code == 200, verified.text
    assert stored["content"] == REPLY, stored
    assert ollama_verified.status_code == 200, ollama_verified.text
    assert [model["name"] for model in tags.json()["models"]] == [OLLAMA_MODEL]
    assert ollama_stored["content"] == "pong", ollama_stored
    assert "/v1/chat/completions" in proxy.requests_for(BEHIND_PROXY)
    assert "/api/chat" in proxy.requests_for(f"ollama.{BEHIND_PROXY}")


@pytest.mark.parametrize("label", name_forms())
def test_a_no_proxy_name_is_reached_directly(resolver, label, proxy, proxied_instance, preserve):
    preserve(OPENAI_CONFIG, on=proxied_instance)
    with serving_by_name(label) as provider, admin_of(proxied_instance).client() as client:
        host = provider.base_url.split("//")[1].rsplit(":", 1)[0]
        provider.route("POST", "/v1/chat/completions", sse({"content": REPLY}))
        attach(client, provider, PROXIED_MODEL)
        _, stored = ask(client, "straight there?", model=PROXIED_MODEL)

    assert stored["content"] == REPLY, stored
    assert provider.requests_to("/v1/chat/completions"), "the provider was never called"
    assert proxy.requests_for(host) == [], "a NO_PROXY name went through the proxy"


def test_a_name_the_proxy_cannot_resolve_fails_the_same_way(resolver, proxied_instance):
    with admin_of(proxied_instance).client() as client:
        verified, seconds = timed(
            client.post,
            "/openai/verify",
            json={"url": f"http://{UNKNOWN_TO_PROXY}:8000/v1", "key": "sk-proxied"},
        )

    assert (verified.status_code, verified.text) == (
        502,
        f"proxy cannot resolve {UNKNOWN_TO_PROXY}",
    )
    assert seconds < FAILS_WITHIN, f"the proxy's refusal took {seconds:.1f}s"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_proxy_whose_name_does_not_resolve_fails_every_call_the_same_way(
    resolver, instance_with, preserve
):
    unreachable_proxy = f"http://{UNRESOLVABLE}:3128"
    stranded = instance_with(
        {
            **RESOLVERS[resolver],
            "HTTP_PROXY": unreachable_proxy,
            "HTTPS_PROXY": unreachable_proxy,
            "NO_PROXY": "127.0.0.1",
        }
    )
    preserve(OPENAI_CONFIG, on=stranded)
    with admin_of(stranded).client() as client:
        verified, seconds = timed(
            client.post,
            "/openai/verify",
            json={"url": f"http://{BEHIND_PROXY}:8000/v1", "key": "sk-proxied"},
        )
        listed = client.get("/api/models")

    assert (verified.status_code, verified.json()) == (500, {"detail": CONNECTION_ERROR})
    assert listed.status_code == 200, listed.text
    assert seconds < FAILS_WITHIN, f"{resolver} took {seconds:.1f}s to refuse the proxy's name"
