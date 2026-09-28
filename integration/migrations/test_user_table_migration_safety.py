"""Regression: upgrading an install from before the user-table migration broke its accounts.

The migration `b10670c03dd5` (0.6.x to 0.9.6) turns the text columns `user.settings` and
`user.info` into JSON columns, moves the single SSO link `user.oauth_sub` into the `user.oauth`
object and the `user.api_key` column into its own table. On a SQLite database that had accounts,
issue #26403 (fix c416c6cad): a populated `info` crashed the upgrade, and populated `settings`
were written serialized a second time, so they came back as a string the app cannot read, while
an empty `{}` was dropped to NULL. The companion fix bd8378f643 (#28101, v0.11.1) stopped the
same migration writing the new `oauth` object as a JSON string.

The server boots on a database built as such an install left it
(`harness.legacy_accounts`), which runs the migration over accounts saved in the old shapes.
Every account then signs in with its password and reads its settings and profile info back
intact, the SSO links from `oauth_sub` still sign their owners in through the OIDC stand-in and
the old API key still authenticates. SQLite and Postgres both run it.

Twin of unit/migrations/test_user_table_migration_safety.py.
Discriminates: passes on dev ef67cc3fa on both engines; with the migration's
`.values({...: parsed})` back to `json.dumps(parsed)` in a copy (#26403) every SQLite test
fails (sign-in answers 500 for each migrated account); with the `oauth_sub` conversion dropped,
or writing `json.dumps({provider: {'sub': sub}})` together with the v0.11.1 repair
`6d09d1bf1f23` emptied, the SSO tests fail on both engines. That conversion mutation alone stays
green: the repair later in the same upgrade decodes the rows before anyone signs in.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import httpx
import pytest

from harness import backends
from harness.legacy_accounts import (
    LEGACY_API_KEY,
    LEGACY_PASSWORD,
    RICH_INFO,
    RICH_SETTINGS,
    legacy_email,
    legacy_id,
    seed_legacy_accounts,
)
from harness.oidc_provider import session_user, shared_provider, sign_in, sso_env
from harness.prepared_data import RunningBackend, serving

pytestmark = [
    pytest.mark.regression,
    pytest.mark.slow,
    pytest.mark.api,
    pytest.mark.requires_source,
]


@pytest.fixture(scope="module")
def idp():
    return shared_provider()


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("sqlite", id="sqlite"),
        pytest.param("postgres", id="postgres", marks=pytest.mark.requires_postgres),
    ],
)
def upgraded(request, idp, tmp_path_factory) -> Iterator[RunningBackend]:
    """The checkout started on the pre-migration database, signing in through `idp`."""
    data_dir = tmp_path_factory.mktemp(f"legacy-{request.param}")
    with contextlib.ExitStack() as stack:
        settings = {**sso_env(idp), "ENABLE_API_KEYS": "true"}
        if request.param == "postgres":
            settings["DATABASE_URL"] = stack.enter_context(backends.postgres_database())
        seed_legacy_accounts(data_dir, settings.get("DATABASE_URL"))
        yield stack.enter_context(serving(data_dir, settings))


def _signed_in(backend: RunningBackend, who: str) -> httpx.Client:
    with backend.client() as client:
        answer = client.post(
            "/api/v1/auths/signin",
            json={"email": legacy_email(who), "password": LEGACY_PASSWORD},
        )
    assert answer.status_code == 200, f"{who} cannot sign in after the upgrade: {answer.text}"
    return backend.client(answer.json()["token"])


def _settings(backend: RunningBackend, who: str):
    with _signed_in(backend, who) as client:
        answer = client.get("/api/v1/users/user/settings")
    assert answer.status_code == 200, answer.text
    return answer.json()


def test_saved_settings_come_back_as_saved(upgraded):
    assert _settings(upgraded, "alice") == RICH_SETTINGS, (
        "user settings came back altered after the user-table migration (#26403)"
    )


def test_empty_settings_stay_empty(upgraded):
    assert _settings(upgraded, "bob") == {"ui": {}}, "empty settings were dropped by the upgrade"


def test_accounts_without_settings_still_sign_in(upgraded):
    assert _settings(upgraded, "carol") is None


def test_profile_info_comes_back_as_saved(upgraded):
    with _signed_in(upgraded, "alice") as client:
        info = client.get("/api/v1/users/user/info")
    assert info.status_code == 200, info.text
    assert info.json() == RICH_INFO


@pytest.mark.parametrize("who", ["dana", "erin"])
def test_an_old_sso_link_still_signs_its_owner_in(upgraded, idp, who):
    """`oidc@sub` and a bare sub, the two shapes `oauth_sub` held."""
    sub = "legacy-dana-sub" if who == "dana" else "legacy-erin-sub"
    idp.sign_in_as(sub=sub, email=legacy_email(who), name=who.title())

    result = sign_in(upgraded)

    assert result.token, f"{who}'s SSO sign-in was refused after the upgrade: {result.error}"
    assert session_user(upgraded, result.token)["id"] == legacy_id(who), (
        f"{who}'s SSO sign-in landed on another account"
    )


def test_the_old_api_key_still_authenticates(upgraded):
    with upgraded.client(LEGACY_API_KEY) as client:
        answer = client.get("/api/v1/auths/")
    assert answer.status_code == 200, (
        f"the API key saved before the upgrade stopped working: {answer.text}"
    )
    assert answer.json()["id"] == legacy_id("admin")
