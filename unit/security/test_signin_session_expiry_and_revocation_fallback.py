"""Regression: an SSO session expired with whichever of its two tokens ran out first.

open-webui 0.11.4 fix `aaaf26fb8` (following c055203f2, #27520): `get_oauth_token` now applies
the ID token's own expiry when it decides whether to refresh before handing the tokens to an SSO
integration. The stored expiry and every refresh decision a sign-in can produce are pinned over
HTTP in integration/security/test_signin_session_expiry_and_revocation_fallback.py.

This keeps the fallback for an ID token whose expiry cannot be read, which stays unit because a
sign-in validates the ID token, so no session reached over HTTP can hold an unreadable one. It
runs against a real OAuth session in an in-memory database and a local token endpoint.

Discriminates: passes on dev ef67cc3fa; letting the ID token decode error escape the expiry
check makes the unreadable session come back as no token at all.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from unittest.mock import patch

import jwt
import pytest
from fastapi import FastAPI

from harness.listener import json_answer, listening
from unit.security.memory_db import memory_database

pytestmark = pytest.mark.regression

ALICE = "alice-user-id"


def id_token_expiring_in(seconds: int) -> str:
    claims = {"sub": "u1", "exp": int(time.time()) + seconds}
    return jwt.encode(claims, "unit-test-key-of-a-decent-length-0123456789", algorithm="HS256")


@pytest.fixture
def token_endpoint():
    """The provider's discovery document and token endpoint, recording every refresh."""
    with listening() as provider:
        discovery = {"issuer": provider.base_url, "token_endpoint": f"{provider.base_url}/token"}
        provider.route("GET", "/.well-known/openid-configuration", json_answer(discovery))
        provider.route(
            "POST", "/token", json_answer({"access_token": "refreshed", "expires_in": 3600})
        )
        yield provider


@asynccontextmanager
async def signed_in_session(owui_module, provider, *, expires_in: int, id_token: str):
    """A real OAuthManager and one stored SSO session whose tokens live as long as asked."""
    oauth = owui_module("open_webui.utils.oauth")
    oauth_sessions = owui_module("open_webui.models.oauth_sessions")
    config = owui_module("open_webui.models.config").Config

    def register(registry):
        discovery_url = f"{provider.base_url}/.well-known/openid-configuration"
        return registry.register(
            name="oidc", client_id="owui", client_secret="secret", server_metadata_url=discovery_url
        )

    app = FastAPI()
    app.state.redis = None
    token = {
        "access_token": "stored",
        "refresh_token": "r",
        "id_token": id_token,
        "expires_at": int(time.time()) + expires_in,
    }
    async with memory_database(owui_module, oauth_sessions.OAuthSession, config):
        with patch.dict(oauth.OAUTH_PROVIDERS, {"oidc": {"register": register}}, clear=True):
            manager = oauth.OAuthManager(app=app)
            stored = await oauth_sessions.OAuthSessions.create_session(
                user_id=ALICE, provider="oidc", token=token
            )
            yield manager, stored


@pytest.mark.asyncio
async def test_a_session_with_an_unreadable_id_token_is_handed_out_as_it_is(
    owui_module, token_endpoint
):
    async with signed_in_session(
        owui_module, token_endpoint, expires_in=7200, id_token="not-a-jwt"
    ) as (manager, stored):
        handed_out = await manager.get_oauth_token(user_id=ALICE, session_id=stored.id)

    assert handed_out is not None, "an unreadable ID token lost the whole session"
    assert handed_out["access_token"] == "stored"
    assert token_endpoint.requests_to("/token") == []
