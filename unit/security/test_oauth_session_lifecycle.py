"""Regression: an MCP authorization without a state must never send the person to the provider.

The OAuth sign-in lifecycle is pinned from outside: the switch, key rotation and MCP flows in
integration/security/test_oauth_session_lifecycle.py (with the token expiry), the profile picture
fetch in test_oauth_picture_fetch.py, the sign-up race in test_first_sign_in_race.py and the
seeding of SSO settings in integration/auth/test_oauth_settings_from_env.py.

What stays here is c2107e5bb's guard in the MCP authorize step: the flow is bound to the account
that started it through the state, so a client that produced none must not send the person to
the provider. authlib always makes a state, so no request reaches this branch; only a stand-in
client that returns none does.

Discriminates: passes on dev ef67cc3fa, fails with c2107e5bb reverted (the stateless authorize
saves its data and redirects).
"""

from __future__ import annotations

from unittest.mock import create_autospec

import pytest
from authlib.integrations.starlette_client import StarletteOAuth2App
from fastapi import FastAPI, HTTPException

pytestmark = pytest.mark.regression


@pytest.fixture
def oauth(owui_module):
    return owui_module("open_webui.utils.oauth")


@pytest.mark.asyncio
async def test_an_mcp_authorization_without_a_state_never_leaves(oauth):
    """Narrow: with no state to bind the initiator to, nothing is saved and nobody is sent."""
    client = create_autospec(StarletteOAuth2App, instance=True)
    client.create_authorization_url.return_value = {"url": "https://mcp.example/authorize"}
    manager = oauth.OAuthClientManager(app=FastAPI())
    client_info = oauth.OAuthClientInformationFull(
        client_id="mcp-client", redirect_uris=["https://owui.example/oauth/clients/mcp:x/callback"]
    )
    manager.clients["mcp:x"] = {"client": client, "client_info": client_info}

    with pytest.raises(HTTPException) as refused:
        await manager.handle_authorize(request=None, client_id="mcp:x", user_id="alice")

    assert refused.value.status_code == 500
    client.save_authorize_data.assert_not_awaited()
