"""Regression: a web search where no page loaded blamed the embedding settings, #29327.

Fix 2e43d6698 (PR #31859). When every result page of a web search failed to load, the default
loader still handed back one document per page with no text, and those went on to be embedded.
The search then failed with "Failed to embed and store the retrieved web pages. Check the
embedding configuration", though embedding never had anything to do. A search where no page
loaded any text now stops before embedding with "None of the web search results could be loaded"
(a 404, like "No results found from web search").

A local search engine lists pages on a local listener: a page that answers 404 and one on a port
nothing listens on. Nearby: one page that loads is enough for the search to succeed, and with
embedding bypassed the failed pages are handed back as before.

Discriminates: passes on dev b859124f9, fails with 2e43d6698 reverted (the search answers 500 with
the embedding message).
"""

from __future__ import annotations

import pytest

from harness.instance import free_port
from harness.listener import text_answer
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    save_web_settings,
    serve_search_results,
    web_settings_restored,
)

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

PAGE_TEXT = "Kestrels hover over the verge"
NOTHING_LOADED = "None of the web search results could be loaded"


@pytest.fixture(scope="module")
def fetching_instance(instance_with):
    return instance_with(LOCAL_WEB_FETCH)


@pytest.fixture
def search(fetching_instance, listener):
    """`search(links, bypass_embedding=False)` lists those links as results and searches."""
    listener.route("GET", "/kestrels", text_answer(f"<p>{PAGE_TEXT}</p>"))
    with fetching_instance.client() as client, web_settings_restored(client):

        def run(links: list[str], bypass_embedding: bool = False):
            save_web_settings(
                client,
                **serve_search_results(listener, links),
                WEB_LOADER_ENGINE="",
                BYPASS_WEB_SEARCH_WEB_LOADER=False,
                BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=bypass_embedding,
            )
            return client.post(
                "/api/v1/retrieval/process/web/search", json={"queries": ["kestrels"]}
            )

        yield run


@pytest.fixture
def unreachable_links(listener) -> list[str]:
    return [f"{listener.base_url}/missing", f"http://127.0.0.1:{free_port()}/closed"]


def test_a_search_where_no_page_loaded_says_so(search, unreachable_links):
    searched = search(unreachable_links)

    assert searched.status_code == 404, searched.text
    assert NOTHING_LOADED in searched.json()["detail"]
    assert "embedding" not in searched.text


# ---------------------------------------------------------------- nearby


def test_one_page_that_loads_is_enough(search, listener, unreachable_links):
    searched = search([*unreachable_links, f"{listener.base_url}/kestrels"])

    assert searched.status_code == 200, searched.text
    assert searched.json()["collection_names"]


def test_with_embedding_bypassed_the_failed_pages_come_back_empty(search, unreachable_links):
    searched = search(unreachable_links, bypass_embedding=True)

    assert searched.status_code == 200, searched.text
    assert not any(doc["content"].strip() for doc in searched.json()["docs"])
