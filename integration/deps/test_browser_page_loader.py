"""Dependency smoke: the Playwright web loader on a remote browser, the way a deployment runs it.

With the `playwright` loader engine and `PLAYWRIGHT_WS_URL` set, Open WebUI connects to a
browser server of its own (`BrowserType.connect`), opens a page with service workers blocked,
routes every request the page makes through its own fetch (`Page.route`, answered with
`Route.fulfill`), refuses the page's WebSockets (`Page.route_web_socket`), loads the link
(`Page.goto`) and reads the rendered HTML (`Page.content`). The synchronous API does this for
an attached link, the async API for the results of a web search. Here the browser server is the
one the playwright package ships (`harness.playwright_server`), and the pages write their text
with a script, so only a rendering browser finds it. The local-browser path (`launch`) is driven
in integration/retrieval/test_v0114_playwright_media_skip.py.

Twin of unit/deps/test_playwright.py, which keeps the element removal calls no setting reaches.

Discriminates: passes on dev ef67cc3fa; in a backend copy that launches a local browser in place
of connecting both tests fail on the connection count, and in one that no longer routes the
page's WebSockets both fail the socket check.
"""

from __future__ import annotations

import pytest

from harness.listener import text_answer
from harness.playwright_server import serving_playwright
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    save_web_settings,
    serve_search_results,
    web_settings_restored,
)

pytestmark = [
    pytest.mark.depcheck,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.requires_browser,
    pytest.mark.slow,
]

RENDERED = "Tide tables for the northern breakwater"
PAGE = """<html><body><div id="app">loading</div>
<script src="/render.js"></script></body></html>"""
RENDER_SCRIPT = f"""
new WebSocket("ws://" + location.host + "/socket");
document.getElementById("app").textContent = "{RENDERED} " + location.pathname;
"""


@pytest.fixture(scope="module")
def browser_server():
    with serving_playwright() as server:
        yield server


@pytest.fixture
def remote_browser_admin(instance_with, browser_server):
    instance = instance_with(LOCAL_WEB_FETCH)
    with instance.client() as client, web_settings_restored(client):
        save_web_settings(
            client,
            WEB_LOADER_ENGINE="playwright",
            PLAYWRIGHT_WS_URL=browser_server.ws_url,
            PLAYWRIGHT_TIMEOUT=10000,
        )
        yield client


def _serve_rendered_pages(listener, *paths: str) -> None:
    for path in paths:
        listener.route("GET", path, text_answer(PAGE))
    listener.route("GET", "/render.js", text_answer(RENDER_SCRIPT, "text/javascript"))


def test_an_attached_link_is_rendered_by_the_remote_browser(
    remote_browser_admin, browser_server, listener
):
    _serve_rendered_pages(listener, "/tides")
    connections_before = browser_server.connections

    loaded = remote_browser_admin.post(
        "/api/v1/retrieval/process/web?process=false", json={"url": f"{listener.base_url}/tides"}
    )

    assert loaded.status_code == 200, loaded.text
    assert f"{RENDERED} /tides" in loaded.json()["content"]
    assert "loading" not in loaded.json()["content"]
    assert browser_server.connections > connections_before, "the remote browser was not used"
    assert listener.requests_to("/socket") == [], "the page's WebSocket reached the server"


def test_web_search_results_are_rendered_by_the_remote_browser(
    remote_browser_admin, browser_server, listener
):
    pages = [f"{listener.base_url}/first", f"{listener.base_url}/second"]
    _serve_rendered_pages(listener, "/first", "/second")
    save_web_settings(
        remote_browser_admin,
        **serve_search_results(listener, pages),
        BYPASS_WEB_SEARCH_WEB_LOADER=False,
        BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True,
    )
    connections_before = browser_server.connections

    searched = remote_browser_admin.post(
        "/api/v1/retrieval/process/web/search", json={"queries": ["tides"]}
    )

    assert searched.status_code == 200, searched.text
    contents = sorted(document["content"] for document in searched.json()["docs"])
    assert [f"{RENDERED} /first", f"{RENDERED} /second"] == [text.strip() for text in contents]
    assert browser_server.connections > connections_before, "the remote browser was not used"
    assert listener.requests_to("/socket") == [], "a page's WebSocket reached the server"
