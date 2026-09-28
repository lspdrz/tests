"""Dependency smoke: web search on DuckDuckGo, the engine that needs no API key.

The `duckduckgo` engine is the ddgs library: Open WebUI opens a `DDGS` with the proxy the
environment names, sets its thread count from the admin's concurrent requests and calls `text`
with the query, the result count and the ddgs backend the admin chose. ddgs's DuckDuckGo backend
posts the query to DuckDuckGo's HTML page over httpx, under a browser User-Agent that
fake-useragent draws from its bundled list, and reads links, titles and snippets out of the page.
DuckDuckGo is played by `harness/duckduckgo.py`, a proxy the instance trusts; its environment is
the instance's own.

Discriminates: passes on dev ef67cc3fa; in a backend copy, opening `DDGS` with a `proxi` keyword
(a bump dropping `proxy`) fails every test, `text` given `max_result` in place of `max_results`
returns more results than the admin allows and reading a result's `url` in place of `href` fails
the three that find something. A `fake_useragent` whose `random` is empty, placed ahead of the
real one (a bump that stops drawing agents), fails only the User-Agent test.
"""

from __future__ import annotations

import pytest

from harness.duckduckgo import (
    Result,
    duckduckgo_env,
    duckduckgo_settings,
    serving_duckduckgo,
)
from harness.web_retrieval import save_web_settings, web_settings_restored

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

RESULTS = [
    Result(
        "https://example.org/herons",
        "Herons of the estuary",
        "Grey herons wait motionless in the shallows.",
    ),
    Result("https://example.org/egrets", "Egrets in winter", "Little egrets gather at dusk."),
    Result("https://example.org/bitterns", "The booming bittern", "Bitterns boom in the reeds."),
]
SEARCH = "/api/v1/retrieval/process/web/search"


@pytest.fixture(scope="module")
def duckduckgo():
    with serving_duckduckgo() as fake:
        yield fake


@pytest.fixture
def searching(instance_with, duckduckgo):
    """A client of an instance searching DuckDuckGo, and the fake with fresh results."""
    duckduckgo.results = list(RESULTS)
    duckduckgo.rate_limited = False
    searcher = instance_with(duckduckgo_env(duckduckgo))
    with searcher.client() as client, web_settings_restored(client):
        save_web_settings(client, **duckduckgo_settings(result_count=2))
        yield client


def test_a_search_lists_what_duckduckgo_found(searching, duckduckgo):
    searched = searching.post(SEARCH, json={"queries": ["herons of the estuary"]})

    assert searched.status_code == 200, searched.text
    assert duckduckgo.queries()[-1] == "herons of the estuary"
    found = {item["link"]: item for item in searched.json()["items"]}
    assert found["https://example.org/herons"]["title"] == "Herons of the estuary"
    assert found["https://example.org/herons"]["snippet"] == RESULTS[0].snippet
    contents = [document["content"] for document in searched.json()["docs"]]
    assert RESULTS[0].snippet in contents


def test_the_admin_result_count_limits_the_results(searching):
    searched = searching.post(SEARCH, json={"queries": ["estuary birds"]})

    assert searched.status_code == 200, searched.text
    assert len(searched.json()["items"]) == 2, searched.json()["items"]


def test_the_search_goes_out_under_a_browser_user_agent(searching, duckduckgo):
    searching.post(SEARCH, json={"queries": ["egrets in winter"]}).raise_for_status()

    agent = duckduckgo.searches[-1].headers.get("User-Agent", "")
    assert agent.startswith("Mozilla/5.0 ("), f"not a browser's User-Agent: {agent!r}"
    assert "python" not in agent.lower(), agent


def test_a_rate_limited_search_fails_with_a_web_search_error(searching, duckduckgo):
    duckduckgo.rate_limited = True
    before = len(duckduckgo.searches)

    searched = searching.post(SEARCH, json={"queries": ["bitterns"]})

    assert len(duckduckgo.searches) > before, "the search never reached DuckDuckGo"
    assert searched.status_code == 400, searched.text
