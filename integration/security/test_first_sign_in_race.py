"""Regression: two first sign-ins of one person at once made two accounts or a broken session.

open-webui fix #27571 (issue #27117, commits b190dcf and 50e050e). A first sign-in checks that no
account has the address, hashes a password and then inserts the account; two requests for the
same new person both passed the check while the other was hashing, and both inserted. Nothing in
the database stopped a second account for one address, and with
`DATABASE_ENABLE_SESSION_SHARING` on, the losing insert left the request's shared database
session in a failed transaction, so that sign-in errored instead of finding the winner. A unique
index on the lower-cased address (migration f0bd01a18a3d) now refuses the second account, and a
failed insert is rolled back, so the sign-in that lost the race carries on as the account that
won it.

Several first sign-ins of one new person are fired at once, through an authenticating proxy's
header and through SSO callbacks, and the admin then counts the accounts that have the address.

Twin of unit/security/test_oauth_session_lifecycle.py (its duplicate-account part).

Discriminates: passes on dev ef67cc3fa; with the unique index left out of the migration both
races leave more than one account for the address, and with the rollback removed from the
account insert the proxy sign-ins that lost the race answer 500 on the shared session.
"""

from __future__ import annotations

import secrets
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from harness.oidc_provider import browser_for, session_user, shared_provider, sign_in, sso_env

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

RACERS = 8
EMAIL_HEADER = "X-Forwarded-Email"
PROXIED_SHARED_SESSION = {
    "WEBUI_AUTH_TRUSTED_EMAIL_HEADER": EMAIL_HEADER,
    "DATABASE_ENABLE_SESSION_SHARING": "true",
}


def accounts_with(instance, email: str) -> list[dict]:
    with instance.client() as admin:
        listed = admin.get("/api/v1/users/", params={"query": email})
    listed.raise_for_status()
    return [user for user in listed.json()["users"] if user["email"] == email]


def all_at_once(requests: list) -> list:
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        return list(pool.map(lambda send: send(), requests))


# ── behind an authenticating proxy, with a shared database session ────────


@pytest.fixture
def proxied(instance_with):
    return instance_with(PROXIED_SHARED_SESSION)


def proxy_sign_in(instance, email: str) -> httpx.Response:
    return httpx.post(
        f"{instance.base_url}/api/v1/auths/signin",
        json={"email": "", "password": ""},
        headers={EMAIL_HEADER: email},
        timeout=60.0,
    )


def test_simultaneous_first_sign_ins_behind_a_proxy_become_one_account(proxied):
    """Narrow: every request signs in, all as the one account the address now has."""
    email = f"racer-{secrets.token_hex(4)}@example.com"

    answers = all_at_once([lambda: proxy_sign_in(proxied, email)] * RACERS)

    statuses = [answer.status_code for answer in answers]
    assert statuses == [200] * RACERS, (
        f"first sign-ins racing for one address answered {statuses}: the losing insert left "
        "the shared database session unusable (#27571)"
    )
    assert len({answer.json()["id"] for answer in answers}) == 1
    assert len(accounts_with(proxied, email)) == 1, "the race made more than one account"


def test_a_proxy_sign_in_in_other_case_is_the_same_account(proxied):
    """Nearby: the address is one account whatever its case."""
    email = f"racer-{secrets.token_hex(4)}@example.com"
    first = proxy_sign_in(proxied, email)
    again = proxy_sign_in(proxied, email.upper())

    assert first.status_code == again.status_code == 200, (first.text, again.text)
    assert again.json()["id"] == first.json()["id"]
    assert len(accounts_with(proxied, email)) == 1


# ── SSO callbacks ──────────────────────────────────────────────────────────


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with(sso_env(idp))


def approved_callback(sso) -> tuple[httpx.Client, str]:
    """A browser that went to the provider and holds the callback it was sent back with."""
    browser = browser_for(sso)
    start = browser.get("/oauth/oidc/login")
    approved = browser.get(start.headers["location"])
    return browser, approved.headers["location"]


def test_simultaneous_first_sso_sign_ins_make_one_account(sso, idp):
    """Narrow: however the race ends for each callback, the address has one account."""
    person = idp.sign_in_as()
    callbacks = [approved_callback(sso) for _ in range(RACERS)]

    answers = all_at_once([lambda pair=pair: pair[0].get(pair[1]) for pair in callbacks])

    for browser, _ in callbacks:
        browser.close()
    assert any(answer.cookies.get("token") for answer in answers), "no callback signed in"
    assert len(accounts_with(sso, person["email"])) == 1, (
        "first SSO sign-ins racing for one address made more than one account (#27571)"
    )


def test_after_the_race_the_person_signs_in_to_their_account(sso, idp):
    """Nearby: the account the race left is the one every later sign-in finds."""
    person = idp.sign_in_as()
    callbacks = [approved_callback(sso) for _ in range(RACERS)]
    all_at_once([lambda pair=pair: pair[0].get(pair[1]) for pair in callbacks])
    for browser, _ in callbacks:
        browser.close()

    idp.sign_in_as(**person)
    later = sign_in(sso)

    assert later.token, f"signing in after the race failed: {later.error}"
    (account,) = accounts_with(sso, person["email"])
    assert session_user(sso, later.token)["id"] == account["id"]
