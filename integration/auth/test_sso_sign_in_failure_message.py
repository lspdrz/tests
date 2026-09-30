"""Regression: a failed SSO sign-in tells the person the identity provider failed.

`6a2aad92f` (#31629, issue #31627): when signing in through an OAuth/OIDC provider failed (the
provider denied access, the token exchange failed, the account had no email, the userinfo had no
sub or the email domain was not allowed) the sign-in page said the email or password was wrong,
though the person typed neither. The callback now sends every such failure back to the sign-in
page with one message that asks them to contact their administrator, and the exact reason still
goes to the server log as a warning.

Twin of e2e/auth/test_sso_sign_in_failure_message.py.

Discriminates: passes on dev a5bc78300; with the fix reverted in a backend copy (the failures
carry the email/password message again) every narrow test fails on the message. The good
sign-in and the wrong password on the email form pass on both and are the controls.
"""

from __future__ import annotations

import urllib.parse

import httpx
import pytest

from harness.actors import create_user
from harness.oidc_provider import (
    browser_for,
    oauth_settings,
    session_user,
    shared_provider,
    sign_in,
    sso_env,
)

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

PROVIDER_FAILED = (
    "Sign-in with your identity provider failed. Please contact your administrator for assistance."
)
EMAIL_OR_PASSWORD = "The email or password provided is incorrect."


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with(sso_env(idp))


def callback_error(sso, *, tamper=None) -> str | None:
    """Walk the sign-in and return the error the callback sends the sign-in page to show."""
    browser = browser_for(sso)
    start = browser.get("/oauth/oidc/login")
    approved = browser.get(start.headers["location"])
    callback_url = approved.headers["location"]
    if tamper is not None:
        callback_url = tamper(callback_url)
    callback = browser.get(callback_url)
    assert callback.status_code in (302, 307), f"callback answered {callback.status_code}"
    assert "token" not in callback.cookies, "the failed sign-in was given a session"
    landing = urllib.parse.urlsplit(callback.headers["location"])
    return dict(urllib.parse.parse_qsl(landing.query)).get("error")


def with_query(url: str, **replacements) -> str:
    parts = urllib.parse.urlsplit(url)
    query = dict(urllib.parse.parse_qsl(parts.query))
    query = {key: value for key, value in {**query, **replacements}.items() if value is not None}
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def assert_provider_failure(error: str | None) -> None:
    assert error == PROVIDER_FAILED, f"the sign-in page was sent to show: {error}"


def test_an_account_without_an_email_gets_the_provider_message(sso, idp):
    """Narrow (#31629): no email and no email fallback fails as a provider failure."""
    idp.sign_in_as(email=None)
    offset = sso.log_size()

    assert_provider_failure(callback_error(sso))

    assert "email is missing" in sso.log_since(offset)


def test_an_email_domain_outside_the_allowed_list_gets_the_provider_message(sso, idp):
    """Narrow (#31629): a domain that is not in the allowed list fails as a provider failure."""
    idp.sign_in_as(email="someone@blocked.test")
    offset = sso.log_size()

    with oauth_settings(sso, OAUTH_ALLOWED_DOMAINS="example.org"):
        error = callback_error(sso)

    assert_provider_failure(error)
    assert "not in the list of allowed domains" in sso.log_since(offset)


def test_an_empty_sub_gets_the_provider_message(sso, idp):
    """Narrow (#31629): an empty sub fails as a provider failure."""
    idp.sign_in_as(sub="")
    offset = sso.log_size()

    assert_provider_failure(callback_error(sso))

    assert "sub is missing" in sso.log_since(offset)


def test_a_token_endpoint_refusal_gets_the_provider_message(sso, idp):
    """Narrow (#31629): a code the provider refuses to exchange fails as a provider failure."""
    idp.sign_in_as()
    offset = sso.log_size()

    assert_provider_failure(callback_error(sso, tamper=lambda url: with_query(url, code="spent")))

    assert "authorize_access_token" in sso.log_since(offset)


def test_a_provider_that_denies_access_gets_the_provider_message(sso, idp):
    """Narrow (#31629): the provider answering the redirect with access_denied fails likewise."""
    idp.sign_in_as()
    offset = sso.log_size()

    def denied(url: str) -> str:
        return with_query(url, code=None, error="access_denied")

    assert_provider_failure(callback_error(sso, tamper=denied))

    assert "access_denied" in sso.log_since(offset)


def test_no_failure_cause_shows_the_email_or_password_message(sso, idp):
    """Broad (#31629): whichever check fails, the message never blames the email or password."""
    errors = []
    idp.sign_in_as(email=None)
    errors.append(callback_error(sso))
    idp.sign_in_as(sub="")
    errors.append(callback_error(sso))
    idp.sign_in_as()
    errors.append(callback_error(sso, tamper=lambda url: with_query(url, code="spent")))

    assert len(set(errors)) == 1
    assert EMAIL_OR_PASSWORD not in errors[0]


def test_a_good_sso_sign_in_still_signs_in(sso, idp):
    """Nearby: a working sign-in gets a session and no error."""
    person = idp.sign_in_as()
    result = sign_in(sso)
    assert result.error is None
    assert result.token, "the sign-in failed"
    assert session_user(sso, result.token)["email"] == person["email"]


def test_a_wrong_password_on_the_email_form_still_says_email_or_password(sso):
    """Nearby: the email form keeps its own message."""
    account = create_user(sso)
    answer = httpx.post(
        f"{sso.base_url}/api/v1/auths/signin",
        json={"email": account.email, "password": "not-the-password"},
        timeout=60.0,
    )
    assert answer.status_code == 400
    assert EMAIL_OR_PASSWORD in answer.json()["detail"]
