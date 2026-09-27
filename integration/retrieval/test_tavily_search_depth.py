"""Journey: the admin picks how deep Tavily web searches go, apart from the extract depth.

PR open-webui/open-webui#31308 (issue open-webui/open-webui#29891) adds a Tavily search depth
(ultra-fast, fast, basic or advanced) to the Tavily settings of the web search page and as
`TAVILY_SEARCH_DEPTH`, defaulting to basic. Every Tavily search sends it; the existing extract
depth still goes only to the Tavily page loader. `TAVILY_API_BASE_URL` points the instance at a
local stand-in for the Tavily API, so the depth is read off the search request itself.

Discriminates: passes on dev ef67cc3fa; in a backend copy, sending no `search_depth` in the
search request fails the four depth cases, the env default search and both independence cases;
saving the extract depth into the search depth fails the read-back, independence and
extract-change cases; ignoring the environment variable fails the three cases that expect it.
"""

from __future__ import annotations

import pytest

from harness.actors import create_user
from harness.listener import json_answer, listening
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    RETRIEVAL_CONFIG,
    save_web_settings,
    web_settings_restored,
)

pytestmark = [
    pytest.mark.journey,
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

WEB_SEARCH = "/api/v1/retrieval/process/web/search"
DEPTHS = ["ultra-fast", "fast", "basic", "advanced"]
RESULT_LINK = "http://127.0.0.1:9/ospreys"
PAGE_TEXT = "Ospreys return in March"


@pytest.fixture(scope="module")
def tavily_api():
    """A stand-in for the Tavily search and extract endpoints."""
    with listening() as service:
        result = {"url": RESULT_LINK, "title": "Ospreys", "content": PAGE_TEXT}
        service.route("POST", "/search", json_answer({"results": [result]}))
        extracted = {"url": RESULT_LINK, "raw_content": PAGE_TEXT}
        service.route("POST", "/extract", json_answer({"results": [extracted]}))
        yield service


@pytest.fixture(scope="module")
def tavily_instance(instance_with, tavily_api):
    return instance_with(
        {
            **LOCAL_WEB_FETCH,
            "TAVILY_API_BASE_URL": tavily_api.base_url,
            "TAVILY_SEARCH_DEPTH": "advanced",
        }
    )


@pytest.fixture
def tavily_admin(tavily_instance):
    with tavily_instance.client() as client, web_settings_restored(client):
        yield client


def web_settings(client) -> dict:
    current = client.get(RETRIEVAL_CONFIG[0])
    current.raise_for_status()
    return current.json()["web"]


def search_with_tavily(client, tavily_api, load_pages: bool = False, **settings):
    """Run one Tavily search with `settings`; returns what the Tavily API was sent."""
    save_web_settings(
        client,
        ENABLE_WEB_SEARCH=True,
        WEB_SEARCH_ENGINE="tavily",
        TAVILY_API_KEY="tavily-key",
        WEB_SEARCH_RESULT_COUNT=1,
        WEB_SEARCH_DOMAIN_FILTER_LIST=[],
        WEB_LOADER_ENGINE="tavily",
        BYPASS_WEB_SEARCH_WEB_LOADER=not load_pages,
        BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True,
        **settings,
    )
    searches_before = len(tavily_api.requests_to("/search"))
    extracts_before = len(tavily_api.requests_to("/extract"))
    searched = client.post(WEB_SEARCH, json={"queries": ["ospreys"]})
    assert searched.status_code == 200, searched.text
    assert PAGE_TEXT in searched.json()["docs"][0]["content"]
    searches = tavily_api.requests_to("/search")[searches_before:]
    extracts = tavily_api.requests_to("/extract")[extracts_before:]
    assert len(searches) == 1, "the search never reached the Tavily API"
    return searches[0].json(), [extract.json() for extract in extracts]


def test_the_environment_sets_the_default_depth(tavily_admin):
    web = web_settings(tavily_admin)

    assert web["TAVILY_SEARCH_DEPTH"] == "advanced"
    assert web["TAVILY_EXTRACT_DEPTH"] == "basic"


def test_without_the_environment_variable_the_depth_is_basic(admin):
    with admin.client() as client:
        assert web_settings(client)["TAVILY_SEARCH_DEPTH"] == "basic"


@pytest.mark.parametrize("depth", DEPTHS)
def test_a_saved_depth_is_read_back_and_sent_with_the_search(tavily_admin, tavily_api, depth):
    search, _ = search_with_tavily(tavily_admin, tavily_api, TAVILY_SEARCH_DEPTH=depth)

    assert web_settings(tavily_admin)["TAVILY_SEARCH_DEPTH"] == depth
    assert search["search_depth"] == depth
    assert search["query"] == "ospreys"


def test_the_env_default_is_sent_until_the_admin_changes_it(tavily_admin, tavily_api):
    search, _ = search_with_tavily(tavily_admin, tavily_api)

    assert search["search_depth"] == "advanced"


@pytest.mark.parametrize(
    ("search_depth", "extract_depth"),
    [("advanced", "basic"), ("ultra-fast", "advanced")],
    ids=["deep-search-shallow-extract", "fast-search-deep-extract"],
)
def test_search_and_extract_depths_are_sent_apart(
    tavily_admin, tavily_api, search_depth, extract_depth
):
    search, extracts = search_with_tavily(
        tavily_admin,
        tavily_api,
        load_pages=True,
        TAVILY_SEARCH_DEPTH=search_depth,
        TAVILY_EXTRACT_DEPTH=extract_depth,
    )

    web = web_settings(tavily_admin)
    assert (web["TAVILY_SEARCH_DEPTH"], web["TAVILY_EXTRACT_DEPTH"]) == (
        search_depth,
        extract_depth,
    )
    assert search["search_depth"] == search_depth
    assert [extract["extract_depth"] for extract in extracts] == [extract_depth]


def test_changing_the_extract_depth_leaves_the_search_depth(tavily_admin):
    save_web_settings(tavily_admin, TAVILY_SEARCH_DEPTH="fast")
    save_web_settings(tavily_admin, TAVILY_EXTRACT_DEPTH="advanced")

    web = web_settings(tavily_admin)
    assert (web["TAVILY_SEARCH_DEPTH"], web["TAVILY_EXTRACT_DEPTH"]) == ("fast", "advanced")


def test_a_user_cannot_change_the_depth(tavily_instance, tavily_admin):
    with create_user(tavily_instance).client() as client:
        refused = client.post(RETRIEVAL_CONFIG[1], json={"web": {"TAVILY_SEARCH_DEPTH": "fast"}})

    assert refused.status_code in (401, 403), refused.text
    assert web_settings(tavily_admin)["TAVILY_SEARCH_DEPTH"] == "advanced"
