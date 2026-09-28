"""Regression: an outage of Redis locked every signed-in user out, and an SSO session expired early.

open-webui 0.11.4 fixes `c1615bec2` and `6a85abb3f`: `is_valid_token` read the revocation
markers from Redis with no error handling, so a Redis outage turned every authenticated request
into a 500. It now accepts the token when Redis raises, with one rate-limited warning, so a
sign-out may not take effect until Redis is back. While Redis answers, a signed-out token is
still refused.

open-webui 0.11.4 fix `aaaf26fb8` (following c055203f2, #27520): the stored expiry of an SSO
session was capped at the ID token's own `exp`, so an ID token shorter than the access token
renewed the session early. The stored expiry now tracks the access token, and the ID token's
expiry only counts when Open WebUI decides whether to refresh before it hands the tokens to an
SSO integration (here a `system_oauth` tool server). The provider signs the ID token with a
lifetime of its own; the admin's view of the user's sessions shows the stored expiry.

Discriminates: passes on dev ef67cc3fa; with c1615bec2 reverted in a copy the session request
after Redis vanished answers 500, with aaaf26fb8 reverted the stored expiry of the shorter ID
token is capped at its own, and with the ID token check dropped from `get_oauth_token` the
session whose ID token is about to expire is forwarded without a refresh.
"""

from __future__ import annotations

import socket
import time

import pytest

from harness.actors import create_user
from harness.listener import json_answer
from harness.oidc_provider import (
    oauth_settings,
    session_user,
    shared_provider,
    sign_in,
    sso_env,
)
from integration.stateful_redis import StatefulRedis

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]


class VanishingRedis(StatefulRedis):
    """The stateful fake Redis, plus `vanish()`: stop listening and cut every connection."""

    def __init__(self) -> None:
        self.connections: list[socket.socket] = []
        super().__init__()

    def _handler(self):
        base = super()._handler()
        connections = self.connections

        class Handler(base):
            def setup(self) -> None:
                super().setup()
                connections.append(self.connection)

        return Handler

    def vanish(self) -> None:
        self.close()
        for connection in self.connections:
            connection.shutdown(socket.SHUT_RDWR)


@pytest.fixture
def redis():
    redis = VanishingRedis()
    yield redis
    redis.close()


def test_sign_out_revokes_while_redis_answers_and_an_outage_locks_nobody_out(instance_with, redis):
    """Nearby: with Redis up a signed-out token is refused. Narrow: once Redis is gone, a
    signed-in request still answers instead of failing."""
    backend = instance_with({"REDIS_URL": redis.url})
    signed_out = create_user(backend)
    with signed_out.client() as client:
        assert client.post("/api/v1/auths/signout").status_code == 200
        assert client.get("/api/v1/auths/").status_code == 401, "sign-out did not revoke"

    with backend.client() as admin:
        assert admin.get("/api/v1/auths/").status_code == 200
        redis.vanish()
        after_outage = admin.get("/api/v1/auths/")
    assert after_outage.status_code == 200, f"HTTP {after_outage.status_code} {after_outage.text}"


TOOL_SPEC = {"openapi": "3.0.0", "info": {"title": "SSO tools", "version": "1"}, "paths": {}}


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with(sso_env(idp))


def sign_in_with_lifetimes(sso, idp, *, access_token: int, id_token: int):
    """An SSO admin's browser, signed in with tokens that live as long as asked."""
    idp.token_lifetime = access_token
    idp.id_token_lifetime = id_token
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        idp.sign_in_as(roles=["admin"])
        result = sign_in(sso)
    assert result.token, f"the sign-in failed: {result.error}"
    return result


def stored_expiry(sso, signed_in) -> int:
    """The session's expiry as the admin's view of the user's SSO sessions shows it."""
    user_id = session_user(sso, signed_in.token)["id"]
    with sso.client() as admin:
        sessions = admin.get(f"/api/v1/users/{user_id}/oauth/sessions")
    assert sessions.status_code == 200, sessions.text
    [session] = sessions.json()
    return session["expires_at"]


@pytest.mark.parametrize(
    ("access_token_ttl", "id_token_ttl"),
    [(7200, 600), (600, 7200)],
    ids=["shorter-id-token", "longer-id-token"],
)
def test_the_stored_expiry_is_the_access_tokens_own(sso, idp, access_token_ttl, id_token_ttl):
    """Narrow (shorter): the ID token no longer caps the stored expiry. Nearby (longer): nor
    does it ever extend it."""
    signed_in = sign_in_with_lifetimes(
        sso, idp, access_token=access_token_ttl, id_token=id_token_ttl
    )

    assert stored_expiry(sso, signed_in) == pytest.approx(time.time() + access_token_ttl, abs=30)


def forwarded_token(browser, tool_server) -> str | None:
    """Verify a `system_oauth` tool server; returns the bearer token it was sent."""
    tool_server.route("GET", "/openapi.json", json_answer(TOOL_SPEC))
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
    [fetch] = tool_server.requests_to("/openapi.json")
    authorization = fetch.headers.get("Authorization")
    return authorization.removeprefix("Bearer ") if authorization else None


@pytest.mark.parametrize(
    ("access_token_ttl", "id_token_ttl", "refreshed"),
    [(7200, 60, True), (7200, 600, False), (60, 7200, True)],
    ids=["id-token-about-to-expire", "both-live", "access-token-about-to-expire"],
)
def test_a_session_is_refreshed_exactly_when_a_token_it_hands_out_is_expiring(
    sso, idp, listener, access_token_ttl, id_token_ttl, refreshed
):
    """Narrow (id-token-about-to-expire): the ID token's expiry counts at hand-out time, so an
    SSO integration never gets a dying ID token. Nearby: a live pair is handed out as it is, and
    the access token's own expiry still refreshes."""
    signed_in = sign_in_with_lifetimes(
        sso, idp, access_token=access_token_ttl, id_token=id_token_ttl
    )
    issued_at_sign_in = idp.issued[-1]

    forwarded = forwarded_token(signed_in.browser, listener)

    refreshes = [
        entry
        for entry in idp.requests_to("/token")
        if entry.form.get("grant_type") == "refresh_token"
    ]
    assert len(refreshes) == int(refreshed), (
        f"{len(refreshes)} refreshes, expected {int(refreshed)}"
    )
    expected = idp.issued[-1] if refreshed else issued_at_sign_in
    assert forwarded == expected["access_token"]
