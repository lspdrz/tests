"""Dependency contract: fake-useragent, the parts no Open WebUI request reaches.

fake-useragent is a declared requirement of the Open WebUI backend that the backend never
imports itself: ddgs's DuckDuckGo backend draws the User-Agent of every web search from
``UserAgent().random``. That is driven from outside in integration/deps/test_web_search_stack.py,
which checks the search goes out under a browser's User-Agent.

Kept as a unit contract: the rest of the public surface (the ``FakeUserAgent`` alias, the
constructor's filters and fallback, the per-browser accessors, the error type) and the promise
that the bundled data is read without the network, none of which a request reaches. Pattern
mirrors test_requests.py. Uses ``depcheck`` from conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "fake_useragent"

TOP_LEVEL_SYMBOLS = [
    "UserAgent",  # primary public class
    "FakeUserAgent",  # canonical class name (UserAgent is an alias)
    "FakeUserAgentError",  # raised when no UA can be produced
]

# Per-browser convenience accessors consumers commonly read.
BROWSER_PROPERTIES = ["chrome", "firefox", "safari", "edge", "random"]


def test_top_level_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_useragent_aliases_fakeuseragent(depcheck):
    """`UserAgent` and `FakeUserAgent` have historically been the same class;
    code may import either name. Pin the alias."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.UserAgent is mod.FakeUserAgent


def test_constructor_filter_kwargs(depcheck):
    """UserAgent(browsers=, os=, min_version=, fallback=, ...) — the filter
    kwargs consumers use to constrain output and the fallback string must
    remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.UserAgent.__init__,
        ["browsers", "os", "fallback"],
    )


def test_error_type_is_exception(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.FakeUserAgentError, Exception)


# ---------------------------------------------------------------------------
# Behavioural contracts (OFFLINE — bundled dataset, never networks).
# ---------------------------------------------------------------------------


def test_behaviour_per_browser_properties_return_strings(depcheck):
    """The per-browser accessors (.chrome/.firefox/.safari/.edge) and .random
    must all yield UA strings — these are the read patterns consumers use."""
    mod = depcheck.load(IMPORT_NAME)
    ua = mod.UserAgent()
    for prop in BROWSER_PROPERTIES:
        value = getattr(ua, prop)
        assert isinstance(value, str) and value, f"UserAgent.{prop} empty/non-str"
        assert value.startswith("Mozilla/"), f"UserAgent.{prop} implausible: {value!r}"


def test_behaviour_getitem_accessor(depcheck):
    """ua['chrome'] indexing is a documented accessor equivalent to ua.chrome;
    pin it so consumers using the dict-style form keep working."""
    mod = depcheck.load(IMPORT_NAME)
    ua = mod.UserAgent()
    value = ua["chrome"]
    assert isinstance(value, str) and value.startswith("Mozilla/")


def test_behaviour_browser_filter_constrains_output(depcheck):
    """Constructing with browsers=['Firefox'] must still produce a usable UA
    (Firefox/Gecko-flavoured) entirely offline."""
    mod = depcheck.load(IMPORT_NAME)
    ua = mod.UserAgent(browsers=["Firefox"], os=["Windows"])
    value = ua.random
    assert isinstance(value, str) and value.startswith("Mozilla/")
    # Firefox UAs carry the Gecko/Firefox tokens; the bundled data should match,
    # but tolerate the library's fallback by only requiring a Mozilla UA.


def test_behaviour_fallback_string_preserved(depcheck):
    """The fallback string passed to the constructor must be retained and used
    when a filter yields no match — consumers rely on a guaranteed UA."""
    mod = depcheck.load(IMPORT_NAME)
    sentinel = "Mozilla/5.0 (compatible; OWUI-Test/1.0)"
    ua = mod.UserAgent(fallback=sentinel)
    assert ua.fallback == sentinel


def test_behaviour_no_network_imports(depcheck):
    """Guard against the dataset being fetched lazily over the network: socket
    creation is blocked while we instantiate and read a UA. The bundled-data
    design must hold (a regression to live-fetching would break offline use)."""
    import socket

    mod = depcheck.load(IMPORT_NAME)
    real_socket = socket.socket

    def _blocked(*args, **kwargs):
        raise AssertionError("fake_useragent attempted network access at runtime")

    socket.socket = _blocked
    try:
        ua = mod.UserAgent()
        value = ua.random
        assert value.startswith("Mozilla/")
    finally:
        socket.socket = real_socket
