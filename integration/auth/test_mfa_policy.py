"""Journey: the admin's multi-factor switches and what each change does to everyone's sessions.

Switching multi-factor sign-in on signs every account out of every device, the admin included,
and the next password sign-in asks for an authenticator. Saving the settings without touching
the switches signs nobody out. With SSO exempted, an SSO sign-in lands with a session at once
while a password sign-in still needs its code; taking the exemption away sends SSO back to the
second step. Switching MFA off signs everyone out again and the password alone signs in, but the
authenticators stay: an account still shows its authenticator, cannot manage it while the switch
is off, and is asked for a code from that same app once MFA is back on. A sign-in caught half way
by the switch going off has to start again.

Discriminates: in a backend copy, making `is_mfa_required` always answer False turns every test
red; making `update_mfa_config` skip `revoke_all_sessions` turns the switch-off test red (the
sessions from before keep working; switching on is also enforced by the sessions lacking a
verified second step), and making `is_mfa_required` ignore `MFA_ALLOW_OAUTH_BYPASS` turns the SSO
exemption test red.
"""

from __future__ import annotations

import urllib.parse

import pytest

from harness.actors import create_user
from harness.mfa import (
    MFA,
    add_account,
    enroll,
    enrolled_account,
    mfa_step,
    require_mfa,
    sign_in,
    verified,
)
from harness.oidc_provider import browser_for, shared_provider, sso_env

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

ALL_OFF = {"ENABLE_MFA": False, "MFA_ALLOW_OAUTH_BYPASS": False}


@pytest.fixture(scope="module")
def idp():
    return shared_provider()


@pytest.fixture(scope="module")
def switchable(instance_with, idp):
    return instance_with({**sso_env(idp), "DEFAULT_USER_ROLE": "user"})


def session_status(instance, token: str) -> int:
    with instance.client(token) as client:
        return client.get("/api/v1/auths/").status_code


def sso_landing(instance) -> tuple[dict, str | None]:
    """Sign in through SSO; returns the landing page's query and the session cookie, if any."""
    with browser_for(instance) as browser:
        start = browser.get("/oauth/oidc/login")
        approved = browser.get(start.headers["location"])
        callback = browser.get(approved.headers["location"])
    landing = urllib.parse.urlsplit(callback.headers["location"])
    return dict(urllib.parse.parse_qsl(landing.query)), callback.cookies.get("token")


def test_switching_mfa_on_signs_everyone_out_and_asks_for_an_authenticator(switchable):
    require_mfa(switchable, **ALL_OFF)
    member = create_user(switchable)
    admin_session = switchable.admin_token
    assert session_status(switchable, member.token) == 200

    saved = require_mfa(switchable, ENABLE_MFA=True)

    assert saved["sessions_revoked"] is True and saved["ENABLE_MFA"] is True
    assert session_status(switchable, member.token) == 401
    assert session_status(switchable, admin_session) == 401
    answer = sign_in(switchable, member.email, member.password)
    assert answer.json()["next_step"] == "enroll", answer.json()


def test_saving_without_touching_the_switches_signs_nobody_out(switchable):
    require_mfa(switchable, ENABLE_MFA=True)
    account = enrolled_account(switchable)

    saved = require_mfa(switchable, ENABLE_MFA=True, ENABLE_MESSAGE_RATING=True)

    assert saved["sessions_revoked"] is False
    assert session_status(switchable, account.token) == 200


def test_the_sso_exemption_skips_the_code_for_sso_alone(switchable, idp):
    require_mfa(switchable, ENABLE_MFA=True, MFA_ALLOW_OAUTH_BYPASS=True)
    idp.sign_in_as()
    query, session = sso_landing(switchable)
    assert "mfa" not in query and session, f"an exempted SSO sign-in got no session: {query}"
    with switchable.client(session) as client:
        assert client.get(f"{MFA}/status").json()["required"] is False

    account = enrolled_account(switchable)
    assert sign_in(switchable, account.email, account.password).json()["next_step"] == "verify"

    require_mfa(switchable, ENABLE_MFA=True, MFA_ALLOW_OAUTH_BYPASS=False)
    assert session_status(switchable, session) == 401
    query, session = sso_landing(switchable)
    assert query == {"mfa": "1"} and session is None


def test_switching_mfa_off_keeps_the_authenticator_for_next_time(switchable):
    require_mfa(switchable, ENABLE_MFA=True)
    account = enrolled_account(switchable)

    saved = require_mfa(switchable, ENABLE_MFA=False)

    assert saved["sessions_revoked"] is True
    assert session_status(switchable, account.token) == 401
    answer = sign_in(switchable, account.email, account.password)
    assert answer.status_code == 200 and "token" in answer.json(), answer.json()
    account.token = answer.json()["token"]
    with account.client() as client:
        status = client.get(f"{MFA}/status").json()
        regenerate = client.post(
            f"{MFA}/recovery/codes", json={"code": account.authenticator.code()}
        )
    assert status == {"enabled": True, "required": False, "recovery_codes_remaining": 10}
    assert regenerate.status_code == 403, regenerate.text

    require_mfa(switchable, ENABLE_MFA=True)
    assert verified(switchable, account)


def test_a_sign_in_caught_by_the_switch_going_off_starts_again(switchable):
    require_mfa(switchable, ENABLE_MFA=True)
    email, password, _ = add_account(switchable)
    started = sign_in(switchable, email, password)
    assert started.json()["next_step"] == "enroll"

    require_mfa(switchable, ENABLE_MFA=False)
    setup = mfa_step(
        switchable, "enroll/start", {"challenge_token": started.json()["challenge_token"]}
    )

    assert setup.status_code in (401, 403), setup.text
    require_mfa(switchable, ENABLE_MFA=True)
    assert enroll(switchable, email, password).token
