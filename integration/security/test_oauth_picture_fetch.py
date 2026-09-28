"""Regression: the profile picture an SSO provider names could be fetched from an internal host.

open-webui 0.11.1 fix `5dcca59ae` (#26699): on the first sign-in Open WebUI downloads the picture
the provider's `picture` claim points at. It vetted the URL once and then fetched it with a plain
HTTP session, so whatever that session dialled next went unchecked: a host that re-resolved to an
internal address between check and fetch, or a redirect to a host the operator listed in
`WEB_FETCH_FILTER_LIST`. The fetch now goes through the session that checks every request and
every address it connects to.

The picture is served by a local service that redirects to a listed host, `localhost`, and to an
IPv6 spelling of the listed `127.0.0.2`, whose name passes and whose address does not; loopback
is allowed otherwise (`ENABLE_LOCAL_WEB_FETCH`), and redirects are followed as an operator can
switch on. A resolver that answers differently the second time is out of a test's reach; the
listed-address hop is the same connection-time check that stops it.

Twin of unit/security/test_oauth_session_lifecycle.py (its profile picture part).

Discriminates: passes on dev ef67cc3fa, fails with the picture fetched through a plain
`aiohttp.ClientSession` (both listed hops are fetched and stored as the account's picture).
"""

from __future__ import annotations

import base64

import pytest

from harness.listener import listening
from harness.oidc_provider import shared_provider, sign_in, sso_env

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

LISTED_ADDRESS = "127.0.0.2"
LOCAL_FETCH = {
    "ENABLE_LOCAL_WEB_FETCH": "true",
    "WEB_FETCH_FILTER_LIST": f"!{LISTED_ADDRESS}/32,!localhost",
    "AIOHTTP_CLIENT_ALLOW_REDIRECTS": "true",
}
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_ANSWER = (200, {"Content-Type": "image/png"}, PNG)
DEFAULT_PICTURE = "/user.png"


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with({**sso_env(idp), **LOCAL_FETCH})


def redirect_to(location: str):
    return 302, {"Location": location}, b""


def stored_picture(sso, email: str) -> str:
    """The account's picture as the admin's user list stores it."""
    with sso.client() as admin:
        listed = admin.get("/api/v1/users/", params={"query": email}).json()["users"]
    (account,) = [user for user in listed if user["email"] == email]
    return account["profile_image_url"]


def first_sign_in_with_picture(sso, idp, picture_url: str | None) -> str:
    """A new person signs in with `picture_url` as their picture claim; returns their email."""
    person = idp.sign_in_as(**({"picture": picture_url} if picture_url else {}))
    result = sign_in(sso)
    assert result.token, f"the sign-in failed: {result.error}"
    return person["email"]


def test_a_picture_redirecting_to_a_listed_host_is_never_fetched(sso, idp, listener):
    """Narrow: the hop to `localhost`, which the filter list names, is refused."""
    listener.route("GET", "/avatar.png", redirect_to(f"http://localhost:{listener.port}/inner"))
    listener.route("GET", "/inner", PNG_ANSWER)

    email = first_sign_in_with_picture(sso, idp, f"{listener.base_url}/avatar.png")

    assert listener.requests_to("/inner") == [], (
        "the profile picture fetch followed a redirect to a host the filter list names, so an "
        "SSO provider can make the server read internal addresses (#26699)"
    )
    assert stored_picture(sso, email) == DEFAULT_PICTURE


def test_a_picture_hop_resolving_to_a_listed_address_is_never_dialled(sso, idp, listener):
    """Narrow: the name passes the check, so only the address it connects to can refuse it."""
    with listening(host=LISTED_ADDRESS) as internal:
        internal.route("GET", "/inner", PNG_ANSWER)
        hop = f"http://[::ffff:{LISTED_ADDRESS}]:{internal.port}/inner"
        listener.route("GET", "/avatar.png", redirect_to(hop))

        email = first_sign_in_with_picture(sso, idp, f"{listener.base_url}/avatar.png")
        reached = list(internal.received)

    assert reached == [], (
        f"the profile picture fetch connected to {LISTED_ADDRESS} although the filter list "
        "names it; the address a fetch dials was never checked (#26699)"
    )
    assert stored_picture(sso, email) == DEFAULT_PICTURE


def test_a_picture_on_an_allowed_host_becomes_the_profile_picture(sso, idp, listener):
    """Nearby: an ordinary picture, redirect included, is still fetched and stored."""
    listener.route("GET", "/avatar.png", redirect_to(f"{listener.base_url}/moved.png"))
    listener.route("GET", "/moved.png", PNG_ANSWER)

    email = first_sign_in_with_picture(sso, idp, f"{listener.base_url}/avatar.png")

    expected = f"data:image/png;base64,{base64.b64encode(PNG).decode()}"
    assert stored_picture(sso, email) == expected


def test_a_provider_without_a_picture_leaves_the_default(sso, idp, listener):
    """Nearby: no picture claim fetches nothing and keeps the default picture."""
    email = first_sign_in_with_picture(sso, idp, None)

    assert listener.received == []
    assert stored_picture(sso, email) == DEFAULT_PICTURE
