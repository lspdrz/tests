"""Dependency contract: authlib, the parts of it no sign-in the suite can drive reaches.

authlib runs every SSO sign-in and every MCP OAuth connection, and those are driven from
outside: the OIDC sign-in, a forged session cookie and a refused code exchange in
integration/deps/test_auth_stack.py, key rotation, unknown providers and the MCP flows in
integration/security/test_oauth_session_lifecycle.py, userinfo and group claims in
integration/security/test_oauth_identity.py, token refresh in
integration/auth/test_sso_token_refresh_race.py, and PKCE, refresh and re-registration of an MCP
server's client in integration/tools/test_mcp_oauth_connection.py.

Two contracts stay here. The callback error message reads `error` and `description` off an
`OAuth2Error`, but authlib 1.7's clients raise `OAuthError` for a refused exchange, so no
request reaches that branch. And sign-out reads the registered client's `_server_metadata_url`
only for providers whose metadata lives on a fixed public host (Google, Microsoft); an OIDC
provider falls back to `OPENID_PROVIDER_URL`, which is the same address.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck


def test_oauth2_error_is_exception(depcheck):
    """`_build_oauth_callback_error_message` does `isinstance(e, OAuth2Error)`."""
    mod = depcheck.load("authlib")
    oauth2_error = depcheck.resolve(mod, "oauth2.rfc6749.errors.OAuth2Error")
    assert issubclass(oauth2_error, Exception)


def test_oauth2_error_exposes_error_and_description(depcheck):
    """The callback error message is built from `e.error` and `e.description`."""
    mod = depcheck.load("authlib")
    oauth2_error = depcheck.resolve(mod, "oauth2.rfc6749.errors.OAuth2Error")
    error = oauth2_error(error="invalid_grant", description="token expired")
    assert error.error == "invalid_grant"
    assert error.description == "token expired"


def test_client_records_server_metadata_url(depcheck):
    """`get_server_metadata_url` reads `client._server_metadata_url` for the sign-out."""
    mod = depcheck.load("authlib")
    oauth = depcheck.resolve(mod, "integrations.starlette_client.OAuth")()
    client = oauth.register(
        name="oidc",
        client_id="cid",
        client_secret="csec",
        server_metadata_url="https://idp.test/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email"},
    )
    assert (
        getattr(client, "_server_metadata_url", None)
        == "https://idp.test/.well-known/openid-configuration"
    )
