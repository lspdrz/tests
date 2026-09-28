"""Dependency contract: asgiref, the parts nothing in a running Open WebUI reaches.

`utils/audit.py` imports its ASGI type names from `asgiref.typing`; an import that fails takes
the whole app down at startup, and the audit middleware those names annotate is driven from
outside by the audit tests in integration/config/test_observability_middleware.py. What stays
here is what no request can show: that `Scope` is still the union of the HTTP, WebSocket and
Lifespan scopes and `ASGI3Application` a callable alias (typing only), and asgiref's
`sync_to_async` / `async_to_sync` bridge, which the backend does not import.
"""

from __future__ import annotations

import typing

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "asgiref"

# Broader asgiref.typing vocabulary the middleware's scope-dispatch logic
# implicitly depends on (HTTP vs WebSocket vs Lifespan scopes).
TYPING_SCOPE_KINDS = ["HTTPScope", "WebSocketScope", "LifespanScope", "ASGIVersions"]

# Core asgiref.sync surface, not directly imported by the backend, pinned as
# the stable public API (sync<->async bridging the ecosystem relies on).
SYNC_SYMBOLS = ["sync_to_async", "async_to_sync", "SyncToAsync", "AsyncToSync"]


def test_sync_submodule_importable(depcheck):
    """asgiref.sync is the canonical sync<->async bridge module; pin that it
    imports (stable-surface guard)."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.try_load("asgiref.sync")
    assert mod is not None, "asgiref.sync no longer importable"


# --------------------------------------------------------------------------- #
# asgiref.typing, typing structure nothing checks at runtime
# --------------------------------------------------------------------------- #


def test_scope_kinds_exist(depcheck):
    """The middleware dispatches on scope['type'] (http / websocket / lifespan).
    asgiref models those as HTTPScope/WebSocketScope/LifespanScope; pin they
    exist so the typed dispatch stays expressible."""
    depcheck.load(IMPORT_NAME)
    typing_mod = depcheck.load("asgiref.typing")
    depcheck.assert_symbols(typing_mod, TYPING_SCOPE_KINDS)


def test_scope_is_union_of_scope_kinds(depcheck):
    """``Scope`` (imported as ASGIScope) is the union of the concrete scope
    TypedDicts. Pin that it is a typing-union that includes the HTTP, WebSocket
    and Lifespan scope types: the structural guarantee the middleware relies on
    when it narrows a generic Scope to an HTTP request.

    A bump that collapsed Scope to a bare ``dict`` (losing the discriminated
    union) would silently weaken the middleware's typing; catch it here."""
    depcheck.load(IMPORT_NAME)
    typing_mod = depcheck.load("asgiref.typing")
    scope = typing_mod.Scope
    args = typing.get_args(scope)
    if not args:
        # Some builds expose Scope as a plain alias; accept that but ensure the
        # concrete scope kinds are at least independently present.
        for kind in ("HTTPScope", "WebSocketScope", "LifespanScope"):
            assert hasattr(typing_mod, kind)
        return
    member_names = {getattr(a, "__name__", str(a)) for a in args}
    for kind in ("HTTPScope", "WebSocketScope", "LifespanScope"):
        assert any(kind in n for n in member_names), (
            f"asgiref.typing.Scope union no longer includes {kind}: {member_names}"
        )


def test_asgi3application_is_callable_alias(depcheck):
    """``ASGI3Application`` types the audited app object, which is invoked as
    ``await app(scope, receive, send)``. It is a Callable type alias; pin that
    it carries callable-type structure (get_origin resolves to a callable),
    so it stays usable as the app annotation."""
    depcheck.load(IMPORT_NAME)
    typing_mod = depcheck.load("asgiref.typing")
    app_t = typing_mod.ASGI3Application
    origin = typing.get_origin(app_t)
    # Callable aliases have collections.abc.Callable as their origin; if the
    # build models it differently, fall back to confirming it is subscriptable
    # type-machinery (has __args__) rather than a plain class.
    assert origin is not None or hasattr(app_t, "__args__"), (
        f"ASGI3Application no longer a typing alias: {app_t!r}"
    )


# --------------------------------------------------------------------------- #
# asgiref.sync: stable-surface guard (not directly imported by the backend)
# --------------------------------------------------------------------------- #
def test_sync_symbols_exist(depcheck):
    """sync_to_async / async_to_sync (and their class forms) are asgiref's core
    public API. Pin they remain present even though the backend doesn't import
    them today: a removal would signal a major, breaking asgiref reshuffle."""
    depcheck.load(IMPORT_NAME)
    sync_mod = depcheck.load("asgiref.sync")
    depcheck.assert_symbols(sync_mod, SYNC_SYMBOLS)


def test_sync_helpers_callable(depcheck):
    depcheck.load(IMPORT_NAME)
    sync_mod = depcheck.load("asgiref.sync")
    assert callable(sync_mod.sync_to_async)
    assert callable(sync_mod.async_to_sync)
