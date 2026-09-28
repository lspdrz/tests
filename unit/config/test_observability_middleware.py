"""Regression: every HTTP middleware is pure ASGI, so none re-buffers a response.

open-webui 0.11.0 (#26924, issue #26922): `SecurityHeadersMiddleware` was a `BaseHTTPMiddleware`,
which re-buffers every response through an anyio memory stream and cut streamed audio short. The
fix made every HTTP middleware pure ASGI. This audit sweeps the app's registered middleware, so it
also catches one upstream adds later; what each response carries today, streamed replies
included, is pinned over HTTP in integration/config/test_observability_middleware.py.

Discriminates: passes on bbfa876af; registering an `@app.middleware('http')` fails the audit.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression


def test_no_registered_middleware_rebuffers_responses(owui_module):
    from starlette.middleware.base import BaseHTTPMiddleware

    registered = owui_module("open_webui.main").app.user_middleware
    assert registered, "open_webui.main.app registers no middleware; retarget this audit"

    rebuffering = [
        str(middleware)
        for middleware in registered
        if isinstance(middleware.cls, type) and issubclass(middleware.cls, BaseHTTPMiddleware)
    ]
    assert rebuffering == [], (
        "a BaseHTTPMiddleware re-buffers every response through a memory stream, which cut "
        f"streamed replies short (#26922): {rebuffering}"
    )
