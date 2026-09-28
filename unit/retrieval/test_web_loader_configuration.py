"""Regression: Playwright web loader cleanup that no request can reach, fixed in v0.11.0.

Leaked Playwright sessions (PR #27526, commit 94b1b7e6b, issue #25880): `SafePlaywrightURLLoader`
never closed a page and closed the browser only after the URL loop finished, so a page timeout
or an abandoned search left the remote browser session open until a restart. A web search, which
reads every page through the async loader and tolerates failures, is driven in a real browser by
integration/retrieval/test_web_loader_configuration.py. What stays here no caller reaches: the
synchronous loader with more than one URL (an attached link is always one), a failure raised out
of the loader (every caller tolerates failures) and a search abandoned half way (every caller
reads to the end). The unit lane has no browser, so the loader runs against specced Playwright
classes (`unit/specced_playwright.py`).

The urllib3-future socket options (#26791) moved to integration/retrieval/test_web_loaders.py,
and the admin loader settings (#26747), embedding failures (#26883) and per-URL parser (#27367)
to integration/retrieval/test_web_loader_configuration.py.

Discriminates: passes on dev bbfa876af; opening the page outside a `with` block while closing
the browser only after the loop fails all four cleanup cases.
"""

from __future__ import annotations

import playwright.async_api
import playwright.sync_api
import pytest

from harness.listener import listening, text_answer
from unit.specced_playwright import released, specced_browser

pytestmark = pytest.mark.regression


@pytest.fixture
def pages(retrieval_web_utils_module, monkeypatch):
    """Two local pages the loader may fetch."""
    monkeypatch.setattr(retrieval_web_utils_module, "ENABLE_LOCAL_WEB_FETCH", True)
    with listening() as page_host:
        for name in ("a", "b"):
            page_host.route("GET", f"/{name}", text_answer(f"<p>page {name}</p>"))
        yield [f"{page_host.base_url}/a", f"{page_host.base_url}/b"]


def playwright_loader(module, urls):
    return module.SafePlaywrightURLLoader(
        web_paths=urls, verify_ssl=False, continue_on_failure=False
    )


async def load_async(documents):
    return [document async for document in documents]


def test_a_page_timeout_releases_the_page_and_the_browser(retrieval_web_utils_module, pages):
    loader = playwright_loader(retrieval_web_utils_module, pages[:1])

    with specced_browser(failing=(pages[0],)) as browsing:
        with pytest.raises(playwright.sync_api.TimeoutError):
            list(loader.lazy_load())

    assert [released(page) for page in browsing.pages] == [True]
    assert released(browsing.browser)


def test_every_page_is_released_as_soon_as_it_is_loaded(retrieval_web_utils_module, pages):
    loader = playwright_loader(retrieval_web_utils_module, pages)

    with specced_browser() as browsing:
        documents = list(loader.lazy_load())

    assert [document.metadata["source"] for document in documents] == pages
    assert [released(page) for page in browsing.pages] == [True, True]
    assert released(browsing.browser)


@pytest.mark.asyncio
async def test_an_abandoned_search_releases_the_browser(retrieval_web_utils_module, pages):
    documents = playwright_loader(retrieval_web_utils_module, pages).alazy_load()

    with specced_browser(asynchronous=True) as browsing:
        first = await documents.__anext__()
        await documents.aclose()

    assert first.metadata["source"] == pages[0]
    assert [released(page) for page in browsing.pages] == [True]
    assert released(browsing.browser)


@pytest.mark.asyncio
async def test_an_async_page_timeout_releases_the_page_and_the_browser(
    retrieval_web_utils_module, pages
):
    loader = playwright_loader(retrieval_web_utils_module, pages[:1])

    with specced_browser(asynchronous=True, failing=(pages[0],)) as browsing:
        with pytest.raises(playwright.async_api.TimeoutError):
            await load_async(loader.alazy_load())

    assert [released(page) for page in browsing.pages] == [True]
    assert released(browsing.browser)
