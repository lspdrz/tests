"""Dependency contract: ddgs, the part no Open WebUI request reaches.

Open WebUI's DuckDuckGo web search (`retrieval/web/duckduckgo.py`) is driven from outside in
integration/deps/test_web_search_stack.py and e2e/retrieval/test_duckduckgo_web_search.py:
`DDGS(proxy=...)` as a context manager, its `threads`, and `text(query, safesearch=,
max_results=, backend=)` with the links, titles and snippets it reads out of DuckDuckGo's page.

Kept as a unit contract: the rate-limit error. The loader caught
`ddgs.exceptions.RatelimitException` until #28943 removed that handler, so no request reaches it
now; it is pinned here only so a bump that drops it is noticed. Uses the `depcheck` fixture from
unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "ddgs"
DIST_NAME = "ddgs"


def test_exceptions_submodule_importable(depcheck):
    """The exceptions submodule the rate-limit error lives in must import."""
    depcheck.load(IMPORT_NAME)
    exc_mod = depcheck.try_load("ddgs.exceptions")
    assert exc_mod is not None, "ddgs.exceptions no longer importable"


# --------------------------------------------------------------------------- #
# ddgs.exceptions.RatelimitException — the catchable rate-limit error
# --------------------------------------------------------------------------- #
def test_ratelimit_exception_exists(depcheck):
    """``RatelimitException`` must exist in ddgs.exceptions."""
    depcheck.load(IMPORT_NAME)
    exc_mod = depcheck.load("ddgs.exceptions")
    assert hasattr(exc_mod, "RatelimitException"), "ddgs.exceptions.RatelimitException missing"


def test_ratelimit_exception_is_exception_subclass(depcheck):
    """RatelimitException must subclass Exception, so a broad handler catches it."""
    depcheck.load(IMPORT_NAME)
    exc_mod = depcheck.load("ddgs.exceptions")
    assert issubclass(exc_mod.RatelimitException, Exception)


def test_ddgs_exception_hierarchy(depcheck):
    """ddgs groups its errors under a base ``DDGSException``; RatelimitException
    (and TimeoutException) subclass it. Pin that base + the rate-limit member so
    a future move of the rate-limit type stays catchable via the base too."""
    depcheck.load(IMPORT_NAME)
    exc_mod = depcheck.load("ddgs.exceptions")
    assert hasattr(exc_mod, "DDGSException"), "ddgs.exceptions.DDGSException missing"
    base = exc_mod.DDGSException
    assert issubclass(exc_mod.RatelimitException, base), (
        "RatelimitException no longer subclasses DDGSException"
    )
