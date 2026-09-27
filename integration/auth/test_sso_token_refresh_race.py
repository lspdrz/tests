"""Regression: two requests refreshing one expiring SSO token at once logged the OAuth session out.

Fixes `9db1a518d` (open-webui/open-webui#30426) and `4caf25538` (open-webui/open-webui#30450),
issue open-webui/open-webui#30416. Open WebUI refreshes the provider token it forwards to
`system_oauth` connections within five minutes of its expiry. Two requests arriving together both
spent the same refresh token; a provider that rotates refresh tokens refused the second with
`invalid_grant`, and Open WebUI then deleted the session, so every later request went without a
token until the user signed in again. #30426 made one request refresh while the other waits
and takes the stored result; #30450 moved that lock into Redis when there is one, so requests
on two workers or replicas are serialised too.

The provider holds each refresh answer back a second after spending the token, so the second
request always arrives while the first refresh is in flight. The cross-process test runs two
backends on one database and one Redis, the way replicas share them.

Discriminates: passes on dev efe63bd34; with both fixes reverted in a backend copy the
one-process test fails (two refreshes, the second request forwards nothing and the session is
gone), and with only 4caf25538 reverted the two-process test fails the same way while the
one-process test passes. The nearby test passes on both.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from harness.backends import redis_server
from harness.listener import json_answer
from harness.oidc_provider import oauth_settings, shared_provider, sign_in, sso_env
from harness.prepared_data import serving

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

EXPIRING_SOON = 60  # inside the five minutes before expiry in which Open WebUI refreshes
REFRESH_DELAY = 1.0
TOOL_SPEC = {"openapi": "3.0.0", "info": {"title": "SSO tools", "version": "1"}, "paths": {}}


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def tool_server(listener):
    listener.route("GET", "/openapi.json", json_answer(TOOL_SPEC))
    return listener


def _signed_in_admin(instance, idp) -> httpx.Client:
    """An SSO admin's browser, holding the session and the `oauth_session_id` cookie."""
    with oauth_settings(instance, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        idp.sign_in_as(roles=["admin"])
        result = sign_in(instance)
    assert result.token, f"the sign-in failed: {result.error}"
    return result.browser


def _browser_on(base_url: str, signed_in: httpx.Client) -> httpx.Client:
    """The same signed-in browser, sending its requests to another replica."""
    return httpx.Client(
        base_url=base_url,
        headers=dict(signed_in.headers),
        cookies=signed_in.cookies,
        timeout=60.0,
    )


def _forwarded_token(browser: httpx.Client, tool_server) -> str | None:
    """Verify a `system_oauth` tool server; returns the bearer token it was sent."""
    before = len(tool_server.requests_to("/openapi.json"))
    browser.post(
        "/api/v1/configs/tool_servers/verify",
        json={
            "url": tool_server.base_url,
            "path": "openapi.json",
            "type": "openapi",
            "auth_type": "system_oauth",
            "key": "",
            "config": {},
        },
    )
    fetches = tool_server.requests_to("/openapi.json")
    if len(fetches) == before:
        return None
    authorization = fetches[-1].headers.get("Authorization")
    return authorization.removeprefix("Bearer ") if authorization else None


def _forward_together(browsers: list[httpx.Client], tool_server) -> None:
    with ThreadPoolExecutor(len(browsers)) as pool:
        list(pool.map(lambda browser: _forwarded_token(browser, tool_server), browsers))


def _refresh_grants(idp) -> list[dict[str, str]]:
    forms = [entry.form for entry in idp.requests_to("/token")]
    return [form for form in forms if form.get("grant_type") == "refresh_token"]


def _sent_tokens(tool_server) -> list[str | None]:
    fetches = tool_server.requests_to("/openapi.json")
    headers = [fetch.headers.get("Authorization") for fetch in fetches]
    return [header.removeprefix("Bearer ") if header else None for header in headers]


def test_two_requests_on_one_worker_refresh_the_token_once(instance_with, idp, tool_server):
    sso = instance_with(sso_env(idp))
    idp.token_lifetime = EXPIRING_SOON
    browser = _signed_in_admin(sso, idp)
    idp.token_lifetime = 3600
    idp.refresh_delay = REFRESH_DELAY

    _forward_together([browser, browser], tool_server)
    idp.refresh_delay = 0.0

    refreshes = _refresh_grants(idp)
    refreshed = idp.issued[-1]["access_token"]
    assert len(refreshes) == 1, (
        f"two requests refreshed the same token {len(refreshes)} times; a provider that rotates "
        "refresh tokens refuses the second (#30416)"
    )
    assert _sent_tokens(tool_server) == [refreshed, refreshed], (
        "a request that raced the refresh forwarded no token or a stale one (#30416)"
    )
    assert _forwarded_token(browser, tool_server) == refreshed, (
        "the OAuth session was logged out by the raced refresh (#30416)"
    )


def test_two_requests_on_two_replicas_refresh_the_token_once(instance_with, idp, tool_server):
    with redis_server() as redis_url:
        sso = instance_with({**sso_env(idp), "REDIS_URL": redis_url})
        idp.token_lifetime = EXPIRING_SOON
        browser = _signed_in_admin(sso, idp)
        idp.token_lifetime = 3600
        replica_env = {**sso_env(idp), "REDIS_URL": redis_url, "DATABASE_URL": sso.database_url}
        with (
            serving(sso.data_dir, settings=replica_env) as replica,
            _browser_on(replica.base_url, browser) as replica_browser,
        ):
            idp.refresh_delay = REFRESH_DELAY
            _forward_together([browser, replica_browser], tool_server)
            idp.refresh_delay = 0.0

            refreshes = _refresh_grants(idp)
            refreshed = idp.issued[-1]["access_token"]
            assert len(refreshes) == 1, (
                f"two replicas refreshed the same token {len(refreshes)} times; a provider that "
                "rotates refresh tokens refuses the second (#30450)"
            )
            assert _sent_tokens(tool_server) == [refreshed, refreshed], (
                "the replica that raced the refresh forwarded no token or a stale one (#30450)"
            )
            assert _forwarded_token(replica_browser, tool_server) == refreshed, (
                "the OAuth session was logged out by the raced refresh (#30450)"
            )


def test_requests_one_after_another_refresh_once_and_reuse_the_new_token(
    instance_with, idp, tool_server
):
    sso = instance_with(sso_env(idp))
    idp.token_lifetime = EXPIRING_SOON
    browser = _signed_in_admin(sso, idp)
    idp.token_lifetime = 3600

    first = _forwarded_token(browser, tool_server)
    second = _forwarded_token(browser, tool_server)

    assert len(_refresh_grants(idp)) == 1
    assert first == second == idp.issued[-1]["access_token"]
