"""Dependency contract: the Playwright calls behind element removal, which no setting reaches.

Open WebUI's ``playwright`` web-loader engine (``SafePlaywrightURLLoader`` in
``retrieval/web/utils.py``) connects to a remote browser or launches one, routes and fulfils
every request, refuses WebSockets, loads the page and reads its HTML, on the sync API for an
attached link and the async API for web search. All of that is driven end to end in
integration/deps/test_browser_page_loader.py (remote) and
integration/retrieval/test_v0114_playwright_media_skip.py (local). After loading, the loader
removes the elements its ``remove_selectors`` name::

    for element in page.locator(selector).all():
        if element.is_visible():
            element.evaluate('element => element.remove()')

``get_web_loader`` never passes ``remove_selectors``, so no request runs that loop, and those
calls are pinned here the way the loader makes them: the positional arguments bound against
each method's signature, awaited on the async API and called directly on the sync one. Nothing
launches a browser or starts the driver.

Discriminates: a playwright whose ``Locator.evaluate`` takes its expression by keyword only, or
whose async ``Locator.all`` stops being a coroutine, fails here (each patched into the installed
package in process).
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "playwright"
APIS = ["sync_api", "async_api"]


# (class, method, positional arguments, keywords) exactly as the loader calls them.
LOADER_CALLS = [
    ("Page", "locator", ("nav",), {}),
    ("Locator", "all", (), {}),
    ("Locator", "is_visible", (), {}),
    ("Locator", "evaluate", ("element => element.remove()",), {}),
]
# Awaited in alazy_load, called directly in lazy_load.
AWAITED = {"all", "is_visible", "evaluate"}


def _api(depcheck, name: str):
    return depcheck.resolve(depcheck.load(IMPORT_NAME), name)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    ("owner", "method", "args", "kwargs"),
    LOADER_CALLS,
    ids=[f"{owner}.{method}" for owner, method, _, _ in LOADER_CALLS],
)
def test_the_loader_calls_still_bind(depcheck, api, owner, method, args, kwargs):
    target = getattr(getattr(_api(depcheck, api), owner), method, None)
    assert callable(target), f"{api}.{owner}.{method} is gone"
    try:
        inspect.signature(target).bind(None, *args, **kwargs)
    except TypeError as error:
        pytest.fail(f"the loader's {owner}.{method}(...) call no longer binds on {api}: {error}")
    if method in AWAITED:
        assert inspect.iscoroutinefunction(target) == (api == "async_api"), (
            f"{api}.{owner}.{method} is awaited in alazy_load and called directly in lazy_load"
        )
