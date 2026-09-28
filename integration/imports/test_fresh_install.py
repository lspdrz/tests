"""Regression: a fresh install and a manual migration died on a circular import (issue #29280,
fix 8c0c7b3b6, shipped in v0.11.3).

Importing the config module runs the migrations, and Alembic's `env.py` imports the calendar
model. The calendar model's import chain reached back into the config module while it was still
loading, so every boot stopped before the first table existed, and so did the documented manual
`alembic upgrade head` an operator runs from `backend/open_webui`. The fix moved that import
into the function that needs it.

The server boots on an empty data directory, where the first account to sign up becomes the
admin and reads the admin settings back; it then restarts on that same data. The manual
upgrade runs as the migration guide shows it, and the server then starts on the database it
built.

Twin of unit/imports/test_config_imports_cleanly.py.
Discriminates: passes on dev ef67cc3fa; a module-scope `from open_webui.config import
ENABLE_SIGNUP` in a copy's `models/calendar.py` (the #29280 cycle) fails all three tests.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path

import httpx
import pytest

from harness.instance import isolated_env, resolve_backend
from harness.prepared_data import RunningBackend, serving

pytestmark = [
    pytest.mark.regression,
    pytest.mark.slow,
    pytest.mark.api,
    pytest.mark.requires_source,
]

PASSWORD = "first-admin-password-123"


@pytest.fixture
def backend() -> Path:
    found = resolve_backend()
    if found is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")
    return found


def _sign_up(server: RunningBackend) -> dict:
    email = f"first-{uuid.uuid4().hex[:8]}@example.com"
    with server.client() as client:
        signed_up = client.post(
            "/api/v1/auths/signup", json={"name": "First", "email": email, "password": PASSWORD}
        )
    assert signed_up.status_code == 200, f"HTTP {signed_up.status_code}: {signed_up.text}"
    return signed_up.json()


def _admin_settings(server: RunningBackend, token: str) -> httpx.Response:
    with server.client(token) as client:
        return client.get("/api/v1/auths/admin/config")


def _alembic(backend: Path, data_dir: Path, *arguments: str) -> subprocess.CompletedProcess:
    """`alembic <arguments>` from `backend/open_webui`, as the manual migration guide runs it."""
    (data_dir / "static").mkdir(parents=True, exist_ok=True)
    environment = isolated_env(
        {
            "DATA_DIR": str(data_dir),
            "STATIC_DIR": str(data_dir / "static"),
            "WEBUI_SECRET_KEY": "integration-secret-key",
            "PYTHONPATH": str(backend),
        }
    )
    return subprocess.run(
        [sys.executable, "-m", "alembic", *arguments],
        cwd=backend / "open_webui",
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_a_fresh_install_starts_and_makes_the_first_account_admin(tmp_path):
    with serving(tmp_path) as server:
        first = _sign_up(server)
        settings = _admin_settings(server, first["token"])

    assert first["role"] == "admin", f"the first account on a fresh install got {first['role']}"
    assert settings.status_code == 200, f"HTTP {settings.status_code}: {settings.text}"
    assert "ENABLE_SIGNUP" in settings.json(), settings.json()


def test_a_fresh_install_starts_again_on_its_own_data(tmp_path):
    with serving(tmp_path) as server:
        email = _sign_up(server)["email"]

    with serving(tmp_path) as server, server.client() as client:
        signed_in = client.post("/api/v1/auths/signin", json={"email": email, "password": PASSWORD})

    assert signed_in.status_code == 200, f"HTTP {signed_in.status_code}: {signed_in.text}"
    assert signed_in.json()["role"] == "admin"


def test_the_manual_alembic_upgrade_builds_a_database_the_server_starts_on(backend, tmp_path):
    upgraded = _alembic(backend, tmp_path, "upgrade", "head")
    assert upgraded.returncode == 0, f"alembic upgrade head failed:\n{upgraded.stderr[-3000:]}"

    heads = _alembic(backend, tmp_path, "heads").stdout.split()
    current = _alembic(backend, tmp_path, "current").stdout
    assert heads, "alembic heads named no revision"
    assert heads[0] in current, f"the database is not at head {heads[0]}: {current!r}"

    with serving(tmp_path) as server:
        first = _sign_up(server)
    assert first["role"] == "admin"
