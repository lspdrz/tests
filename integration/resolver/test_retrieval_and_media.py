"""Journey: web search, page loading and embeddings reached by host name, under both resolvers.

`AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` decides which resolver the shared aiohttp pool and the
SSRF-safe connector use. Each test drives one retrieval path by a host name (`localhost`, a
hosts-file name for `::1` alone and a hosts-file name for another local address) and expects the
same outcome under both resolvers: a search through SearXNG and through OpenSERP (the two engines
whose address an admin sets, the others being hardcoded public hosts), the pages a web search
loads through the default loader, a link attached to a chat (fetched by the SSRF-safe session
whether it is a page or a file), and a knowledge upload and query embedded through the OpenAI,
Ollama and Azure OpenAI engines. A name that does not resolve gives the same status and message
each time, the resolver's own wording aside. Left out as not touching aiohttp: the other search
engines (requests), the reranker (requests), the page loader behind `/process/web` (requests) and
`EXTERNAL_PWA_MANIFEST_URL` (only settable in the environment).

Discriminates: on dev 176d31d1d, a backend copy whose `env.py` installs a resolver that fails every
lookup when the flag is on turns every c-ares run of a by-name test red and leaves every threaded
run green; the same resolver installed for the flag off does the reverse. A failure test stays green
under a failing resolver by design, and a resolver that takes 20 seconds to refuse an unknown name
turns it red. Its c-ares runs skip, naming the reason, on a machine whose DNS server keeps c-ares
from refusing an unknown name at once (a cached reply with a stale EDNS cookie, c-ares issues 1081
and 1271).
"""

from __future__ import annotations

import json

import pytest

from harness.actors import admin_of
from harness.host_names import (
    FAILS_WITHIN,
    RESOLVERS,
    UNRESOLVABLE,
    name_forms,
    serving_by_name,
    timed,
    without_resolver_reason,
)
from harness.keyword_embeddings import keyword_vector
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.listener import json_answer, text_answer
from harness.web_retrieval import LOCAL_WEB_FETCH, save_web_settings, web_settings_restored

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

EMBEDDING_CONFIG = ("/api/v1/retrieval/embedding", "/api/v1/retrieval/embedding/update")
KEYWORDS = ["kestrel", "harbour"]
PAGE_TEXT = "Kestrels migrate south in the autumn"
UNRESOLVABLE_URL = f"http://{UNRESOLVABLE}:8000"
SEARCH_FAILED = "[ERROR: Something went wrong while searching the web.]"
# one attempt, then the loader's retries wait 2 s and 3 s before it gives up
LOADER_RETRY_SECONDS = 5.0
UNREACHABLE_EMBEDDINGS = (
    f"Cannot connect to host {UNRESOLVABLE}:8000 ssl:default [<resolver reason>]"
)


@pytest.fixture
def fetching_instance(resolver, package_instance_with):
    """The resolver's instance, allowed to fetch pages from local addresses."""
    return package_instance_with({**RESOLVERS[resolver], **LOCAL_WEB_FETCH})


@pytest.fixture
def fetching_admin(fetching_instance):
    return admin_of(fetching_instance)


@pytest.fixture
def web_admin(fetching_admin):
    with fetching_admin.client() as client, web_settings_restored(client):
        yield client


def _searxng_answering(listener, links: list[str]) -> dict:
    results = [
        {"url": link, "title": link, "content": f"snippet of {link}", "score": 1.0 - index / 10}
        for index, link in enumerate(links)
    ]
    listener.route("GET", "/search", json_answer({"results": results}))
    return {
        "ENABLE_WEB_SEARCH": True,
        "WEB_SEARCH_ENGINE": "searxng",
        "SEARXNG_QUERY_URL": f"{listener.base_url}/search",
        "WEB_SEARCH_RESULT_COUNT": len(links),
    }


def _openserp_answering(listener, links: list[str]) -> dict:
    results = [{"url": link, "title": link, "snippet": f"snippet of {link}"} for link in links]
    listener.route("GET", "/mega/search", json_answer({"results": results}))
    return {
        "ENABLE_WEB_SEARCH": True,
        "WEB_SEARCH_ENGINE": "openserp",
        "OPENSERP_BASE_URL": listener.base_url,
        "WEB_SEARCH_RESULT_COUNT": len(links),
    }


def _search(client, **loading):
    """A web search whose results are only read, or whose pages are loaded when asked to."""
    settings = {
        "BYPASS_WEB_SEARCH_WEB_LOADER": True,
        "BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL": True,
        "WEB_LOADER_ENGINE": "safe_web",
        **loading,
    }
    save_web_settings(client, **settings)
    return client.post("/api/v1/retrieval/process/web/search", json={"queries": ["kestrels"]})


@pytest.mark.parametrize("engine", ["searxng", "openserp"])
@pytest.mark.parametrize("name_form", name_forms())
def test_a_search_engine_answers_by_name(resolver, name_form, engine, web_admin):
    answering = {"searxng": _searxng_answering, "openserp": _openserp_answering}[engine]
    links = ["http://example.com/kestrels", "http://example.org/harbours"]
    with serving_by_name(name_form) as listener:
        save_web_settings(web_admin, **answering(listener, links))
        searched = _search(web_admin)

    assert searched.status_code == 200, searched.text
    assert searched.json()["filenames"] == links
    assert [doc["content"] for doc in searched.json()["docs"]] == [
        f"snippet of {link}" for link in links
    ]
    assert len(listener.requests_to("/search" if engine == "searxng" else "/mega/search")) == 1


@pytest.mark.parametrize("name_form", name_forms())
def test_the_pages_a_search_finds_are_loaded_by_name(resolver, name_form, web_admin):
    with serving_by_name(name_form) as listener:
        page = f"{listener.base_url}/kestrels"
        listener.route(
            "GET", "/kestrels", text_answer(f"<html><body><p>{PAGE_TEXT}</p></body></html>")
        )
        save_web_settings(web_admin, **_searxng_answering(listener, [page]))
        searched = _search(web_admin, BYPASS_WEB_SEARCH_WEB_LOADER=False)

    assert searched.status_code == 200, searched.text
    assert [doc["content"].strip() for doc in searched.json()["docs"]] == [PAGE_TEXT]
    assert len(listener.requests_to("/kestrels")) == 1


@pytest.mark.parametrize("name_form", name_forms())
def test_an_attached_page_and_an_attached_file_are_fetched_by_name(resolver, name_form, web_admin):
    with serving_by_name(name_form) as listener:
        listener.route(
            "GET", "/kestrels", text_answer(f"<html><body><p>{PAGE_TEXT}</p></body></html>")
        )
        listener.route("GET", "/notes.txt", text_answer("harbour notes", "text/plain"))
        page = web_admin.post(
            "/api/v1/retrieval/process/url", json={"url": f"{listener.base_url}/kestrels"}
        )
        notes = web_admin.post(
            "/api/v1/retrieval/process/url", json={"url": f"{listener.base_url}/notes.txt"}
        )

    assert page.status_code == 200, page.text
    assert page.json()["type"] == "web" and PAGE_TEXT in page.json()["content"]
    assert notes.status_code == 200, notes.text
    assert notes.json()["type"] == "file" and notes.json()["name"] == "notes.txt"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_an_unresolvable_search_engine_fails_the_same_way(resolver, web_admin):
    failures = {}
    for engine, settings in {
        "searxng": {
            "WEB_SEARCH_ENGINE": "searxng",
            "SEARXNG_QUERY_URL": f"{UNRESOLVABLE_URL}/search",
        },
        "openserp": {"WEB_SEARCH_ENGINE": "openserp", "OPENSERP_BASE_URL": UNRESOLVABLE_URL},
    }.items():
        save_web_settings(web_admin, ENABLE_WEB_SEARCH=True, **settings)
        failures[engine] = timed(_search, web_admin)

    for engine, (searched, seconds) in failures.items():
        assert (searched.status_code, searched.json()["detail"]) == (400, SEARCH_FAILED), engine
        assert seconds < FAILS_WITHIN, f"{engine} failed only after {seconds:.1f}s"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_an_unresolvable_page_in_the_results_loads_as_an_empty_page(resolver, web_admin):
    with serving_by_name("localhost") as listener:
        page = f"http://{UNRESOLVABLE}/kestrels"
        save_web_settings(web_admin, **_searxng_answering(listener, [page]))
        searched, seconds = timed(_search, web_admin, BYPASS_WEB_SEARCH_WEB_LOADER=False)

    assert searched.status_code == 200, searched.text
    assert [doc["content"] for doc in searched.json()["docs"]] == [""]
    assert seconds < LOADER_RETRY_SECONDS + FAILS_WITHIN, (
        f"the lookup failed only after {seconds:.1f}s"
    )


@pytest.mark.usefixtures("refuses_unknown_names")
def test_an_unresolvable_link_is_refused_when_attached(resolver, web_admin):
    refused, seconds = timed(
        web_admin.post,
        "/api/v1/retrieval/process/url",
        json={"url": f"http://{UNRESOLVABLE}/kestrels"},
    )

    assert refused.status_code == 400, refused.text
    assert (
        refused.json()["detail"]
        == f"[ERROR: Could not read content from http://{UNRESOLVABLE}/kestrels]"
    )
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


def _serve_embeddings(listener, engine: str) -> None:
    """Answer as an embedding engine, one keyword vector per text."""

    def openai(request):
        texts = request.json()["input"]
        return json_answer({"data": [{"embedding": keyword_vector(t, KEYWORDS)} for t in texts]})

    def ollama(request):
        texts = request.json()["input"]
        return json_answer({"embeddings": [keyword_vector(t, KEYWORDS) for t in texts]})

    routes = {
        "openai": ("/embeddings", openai),
        "ollama": ("/api/embed", ollama),
        "azure_openai": ("/openai/deployments/embedder/embeddings", openai),
    }
    listener.route("POST", *routes[engine])


def _embed_settings(engine: str, base_url: str) -> dict:
    connections = {
        "openai": ("openai_config", {"url": base_url, "key": "sk-e"}),
        "ollama": ("ollama_config", {"url": base_url, "key": ""}),
        "azure_openai": (
            "azure_openai_config",
            {"url": base_url, "key": "az-e", "version": "2024-02-01"},
        ),
    }
    section, connection = connections[engine]
    return {"RAG_EMBEDDING_ENGINE": engine, "RAG_EMBEDDING_MODEL": "embedder", section: connection}


def _embedded_texts(listener, engine: str) -> list[str]:
    path = {
        "openai": "/embeddings",
        "ollama": "/api/embed",
        "azure_openai": "/openai/deployments/embedder/embeddings",
    }[engine]
    return [text for call in listener.requests_to(path) for text in call.json()["input"]]


@pytest.mark.parametrize("engine", ["openai", "ollama", "azure_openai"])
@pytest.mark.parametrize("name_form", name_forms())
def test_an_embedding_engine_embeds_an_upload_and_a_query_by_name(
    resolver, name_form, engine, fetching_instance, fetching_admin, preserve
):
    preserve(EMBEDDING_CONFIG, on=fetching_instance)
    with serving_by_name(name_form) as listener, fetching_admin.client() as client:
        _serve_embeddings(listener, engine)
        updated = client.post(EMBEDDING_CONFIG[1], json=_embed_settings(engine, listener.base_url))
        with knowledge_base(client) as knowledge_id:
            add_text_file(client, knowledge_id, "kestrels.txt", PAGE_TEXT)
            queried = client.post(
                "/api/v1/retrieval/query/collection",
                json={"collection_names": [knowledge_id], "query": "kestrel", "k": 1},
            )

    assert updated.status_code == 200, updated.text
    assert queried.status_code == 200, queried.text
    assert queried.json()["documents"][0] == [PAGE_TEXT]
    embedded = _embedded_texts(listener, engine)
    assert PAGE_TEXT in embedded and embedded[-1] == "kestrel", embedded


def _processing_outcome(client, file_id: str) -> dict:
    """The last status event of a file's processing, its error without the resolver's wording."""
    streamed = client.get(f"/api/v1/files/{file_id}/process/status", params={"stream": "true"})
    last = [json.loads(line[len("data: ") :]) for line in streamed.text.splitlines() if line][-1]
    if "error" in last:
        last["error"] = without_resolver_reason(last["error"])
    return last


@pytest.mark.usefixtures("refuses_unknown_names")
@pytest.mark.parametrize("engine", ["openai", "ollama", "azure_openai"])
def test_an_unresolvable_embedding_engine_fails_an_upload_the_same_way(
    resolver, engine, fetching_instance, fetching_admin, preserve
):
    preserve(EMBEDDING_CONFIG, on=fetching_instance)
    with fetching_admin.client() as client:
        updated = client.post(EMBEDDING_CONFIG[1], json=_embed_settings(engine, UNRESOLVABLE_URL))
        with knowledge_base(client) as knowledge_id:
            uploaded, seconds = timed(
                client.post,
                "/api/v1/files/",
                params={"process": "true", "process_in_background": "false"},
                files={"file": ("kestrels.txt", PAGE_TEXT.encode(), "text/plain")},
            )
            processing = _processing_outcome(client, uploaded.json()["id"])
            queried, query_seconds = timed(
                client.post,
                "/api/v1/retrieval/query/collection",
                json={"collection_names": [knowledge_id], "query": "kestrel", "k": 1},
            )

    assert updated.status_code == 200, updated.text
    assert uploaded.status_code == 200, uploaded.text
    assert processing == {"status": "failed", "error": UNREACHABLE_EMBEDDINGS}, processing
    assert (queried.status_code, queried.json()) == (
        400,
        {"detail": "[ERROR: Error querying knowledge base]"},
    ), queried.text
    assert max(seconds, query_seconds) < FAILS_WITHIN, "the lookup failed slowly"
