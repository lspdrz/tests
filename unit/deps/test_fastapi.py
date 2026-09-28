"""Dependency contract: fastapi, the parts no Open WebUI request reaches.

FastAPI is how every request reaches Open WebUI, and what the backend relies on is driven from
outside in integration/deps/test_web_framework.py: `Depends` with `HTTPBearer` and the cookie,
`HTTPException` statuses, details and headers, query, body, form and upload validation, response
models, `BackgroundTasks`, streaming, plain-text and JSON responses, middleware, CORS, static
files and the Swagger page. The terminal WebSocket route is driven in
integration/security/test_terminal_connection_gating.py.

Kept as a unit contract: the exception handler Open WebUI registers only fires on a recurrence
evaluation timeout outside the validated automation routes, which no request reaches reliably;
`dependency_overrides` and `fastapi.__version__` are surface Open WebUI never uses. The
behavioural checks run a tiny in-process app under `fastapi.testclient.TestClient`, so there is
no network. Uses the `depcheck` fixture from unit/deps/conftest.py.
"""

import pytest

# NOTE: deliberately no `from __future__ import annotations` here: the routes below are live
# FastAPI routes whose annotations FastAPI evaluates at runtime.

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "fastapi"


def _app_and_client(mod):
    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover - depends on env (needs httpx)
        pytest.skip(f"fastapi.testclient.TestClient unavailable: {e}")
    app = mod.FastAPI(title="depcheck", docs_url=None, openapi_url=None)
    return app, TestClient


def test_has_version_attr(depcheck):
    """fastapi exposes __version__ for tooling that reports it."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.__version__, str)
    assert mod.__version__


def test_exception_handler_registration(depcheck):
    """@app.exception_handler must let a handler turn an exception into a response; main.py
    registers one for RecurrenceEvaluationTimeout."""
    mod = depcheck.load(IMPORT_NAME)
    app, TestClient = _app_and_client(mod)

    class MyError(Exception):
        pass

    @app.exception_handler(MyError)
    async def handle(request, exc):
        return mod.responses.JSONResponse(
            status_code=mod.status.HTTP_400_BAD_REQUEST, content={"error": str(exc)}
        )

    @app.get("/err")
    def err():
        raise MyError("bad")

    with TestClient(app) as client:
        resp = client.get("/err")
        assert resp.status_code == 400
        assert resp.json() == {"error": "bad"}


def test_dependency_override(depcheck):
    """app.dependency_overrides, the standard test seam, must replace a dependency's value."""
    mod = depcheck.load(IMPORT_NAME)
    app, TestClient = _app_and_client(mod)

    def get_user():
        return {"role": "user"}  # pragma: no cover - overridden below

    @app.get("/role")
    def role(user=mod.Depends(get_user)):
        return {"role": user["role"]}

    app.dependency_overrides[get_user] = lambda: {"role": "admin"}

    with TestClient(app) as client:
        assert client.get("/role").json() == {"role": "admin"}
