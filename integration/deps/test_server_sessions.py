"""Dependency smoke: SSO sign-in state kept in Redis through starsessions.

With `ENABLE_STAR_SESSIONS_MIDDLEWARE` on, Open WebUI keeps the session of an SSO sign-in in
Redis: starsessions' `SessionMiddleware` with a `RedisStore` under `<REDIS_KEY_PREFIX>:session:`,
loaded on every request by `SessionAutoloadMiddleware`. The browser holds only the session id in
the `owui-session` cookie, and the OAuth state authlib saves on the way to the provider is read
back from Redis when the provider sends the browser to the callback. The provider is
`harness/oidc_provider.py`; the Redis is a `redis-server` of the module's own (twin of
unit/deps/test_starsessions.py).

Discriminates: passes on dev ef67cc3fa. In a backend copy that adds Starlette's cookie
`SessionMiddleware` in place of starsessions', or whose `RedisStore` writes under another prefix,
the state test finds no session in Redis; a `RedisStore` whose `read` finds nothing (patched in at
import) fails the callback test. The stranger test pins that the state is tied to the session at
all.
"""

from __future__ import annotations

import json

import pytest
import redis

from harness import backends
from harness.oidc_provider import browser_for, session_user, shared_provider, sign_in, sso_env

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

SESSION_PREFIX = "open-webui:session:"


@pytest.fixture(scope="module")
def redis_url():
    with backends.redis_server() as url:
        yield url


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sessions(instance_with, idp, redis_url):
    return instance_with(
        {**sso_env(idp), "ENABLE_STAR_SESSIONS_MIDDLEWARE": "true", "REDIS_URL": redis_url}
    )


def _stored_session(redis_url: str, session_id: str) -> dict:
    stored = redis.Redis.from_url(redis_url).get(f"{SESSION_PREFIX}{session_id}")
    assert stored is not None, f"no session {session_id} in Redis"
    return json.loads(stored)


def test_the_way_to_the_provider_keeps_the_oauth_state_in_redis(sessions, idp, redis_url):
    idp.sign_in_as(sub="harbour-keeper", email="keeper@harbour.example", name="Harbour Keeper")
    with browser_for(sessions) as browser:
        started = browser.get("/oauth/oidc/login")
        assert started.status_code == 302, started.text
        session_id = browser.cookies.get("owui-session")

    assert session_id, "no session cookie was set"
    stored = _stored_session(redis_url, session_id)
    states = [key for key in stored if key.startswith("_state_oidc_")]
    assert states, f"the OAuth state is not in the stored session: {stored}"
    assert states[0] not in session_id, "the state travels in the cookie"
    cookie = started.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie, cookie


def test_the_callback_reads_the_state_back_and_signs_in(sessions, idp):
    idp.sign_in_as(sub="ferry-master", email="ferry@harbour.example", name="Ferry Master")

    signed_in = sign_in(sessions)

    assert signed_in.error is None, signed_in.error
    assert session_user(sessions, signed_in.token)["email"] == "ferry@harbour.example"


def test_a_callback_from_a_browser_without_the_session_is_refused(sessions, idp):
    idp.sign_in_as(sub="gull", email="gull@harbour.example", name="Gull")
    with browser_for(sessions) as browser, browser_for(sessions) as stranger:
        started = browser.get("/oauth/oidc/login")
        approved = browser.get(started.headers["location"])
        callback = stranger.get(approved.headers["location"])

    assert "token" not in callback.cookies, "a browser without the session was signed in"
