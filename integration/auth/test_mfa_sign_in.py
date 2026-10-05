"""Journey: signing in with an authenticator app once the admin requires one for everybody.

With multi-factor sign-in on, a correct password no longer hands out a session. A first sign-in
answers with a setup step: the account scans a key (also shown as a QR code), confirms it with a
code and gets its session together with ten recovery codes, shown that once. Every later sign-in
answers with a verify step that takes a current code or one of those recovery codes, each used once;
wrong codes are refused and a sign-in ends after five of them. The challenge token itself opens
nothing. Ten password sign-ins hold an account and a hundred requests hold an address. A signed-in
account can generate new recovery codes or replace its authenticator, both with a fresh code and
both signing it out everywhere. LDAP and SSO sign-ins take the same second step (the SSO callback
hands the browser a challenge cookie instead of a session), API keys keep working without a code but
cannot reach the MFA routes or the switches, an account the admin adds gets no session, a pending
account waits without a challenge, the admin can sign one account out everywhere, and an operator's
`open-webui mfa reset` lets a locked-out account set up a new authenticator with a one-time token.

Discriminates: in a backend copy, making `is_mfa_required` always answer False turns every
second-step test red (the password alone signs in); dropping the `last_step` check from
`matching_step` turns the reused-code test red, leaving the used hash in place in
`consume_factor` turns the recovery-code test red, the users router's sessions route returning
without revoking turns the admin sign-out test red, and lifting the per-account and per-address
limits (`limit_account` and the MFA router's address limiter) turns the two limit tests red.
"""

from __future__ import annotations

import urllib.parse

import httpx
import pytest

from harness.ldap_server import LDAP_CONFIG, save_ldap_settings, serve_directory
from harness.mfa import (
    MFA,
    Authenticator,
    add_account,
    admin_account,
    anonymous,
    enrolled_account,
    finish_enrollment,
    mfa_step,
    new_address,
    operator_reset,
    require_mfa,
    sign_in,
    verified,
)
from harness.oidc_provider import browser_for, shared_provider, sso_env

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]


@pytest.fixture(scope="module")
def idp():
    return shared_provider()


@pytest.fixture(scope="module")
def secured(instance_with, idp):
    """An instance whose admin switched MFA on, with SSO and API keys available."""
    instance = instance_with(
        {**sso_env(idp), "ENABLE_API_KEYS": "true", "DEFAULT_USER_ROLE": "user"}
    )
    require_mfa(instance, ENABLE_MFA=True)
    return instance


def session_status(instance, token: str) -> int:
    with instance.client(token) as client:
        return client.get("/api/v1/auths/").status_code


def mfa_status(account) -> dict:
    with account.client() as client:
        answer = client.get(f"{MFA}/status")
    assert answer.status_code == 200, answer.text
    return answer.json()


def challenge_for(secured, account) -> str:
    answer = sign_in(secured, account.email, account.password, account.address)
    assert answer.status_code == 200, answer.text
    assert answer.json()["next_step"] == "verify", answer.json()
    return answer.json()["challenge_token"]


def verify(secured, account, challenge_token: str, code: str, recovery: bool = False):
    body = {"challenge_token": challenge_token, "code": code, "recovery": recovery}
    return mfa_step(secured, "verify", body, account.address)


# --------------------------------------------------------------------------- first sign-in


def test_a_first_sign_in_sets_up_the_authenticator_before_any_session(secured):
    email, password, _ = add_account(secured)
    with anonymous(secured) as browser:
        answer = browser.post("/api/v1/auths/signin", json={"email": email, "password": password})
        assert answer.status_code == 200, answer.text
        assert answer.json()["next_step"] == "enroll"
        assert "token" not in answer.json() and "token" not in browser.cookies
        challenge_token = answer.json()["challenge_token"]

        assert session_status(secured, challenge_token) == 401
        setup = browser.post(f"{MFA}/enroll/start", json={"challenge_token": challenge_token})
        assert setup.status_code == 200, setup.text
        assert setup.json()["qr_code"].startswith("data:image/svg+xml;base64,")
        reloaded = browser.post(f"{MFA}/enroll/start", json={"challenge_token": challenge_token})
        assert reloaded.json()["manual_key"] == setup.json()["manual_key"]

        authenticator = Authenticator(setup.json()["manual_key"])
        wrong = browser.post(
            f"{MFA}/enroll/confirm",
            json={"challenge_token": challenge_token, "code": authenticator.wrong_code()},
        )
        assert wrong.status_code == 401, wrong.text
        confirmed = browser.post(
            f"{MFA}/enroll/confirm",
            json={"challenge_token": challenge_token, "code": authenticator.code()},
        )
    assert confirmed.status_code == 200, confirmed.text
    session = confirmed.json()
    assert session["email"] == email
    assert len(session["recovery_codes"]) == 10 and len(set(session["recovery_codes"])) == 10
    with secured.client(session["token"]) as client:
        status = client.get(f"{MFA}/status").json()
    assert status == {"enabled": True, "required": True, "recovery_codes_remaining": 10}


def test_an_added_account_gets_no_session_while_mfa_is_on(secured):
    _, _, added = add_account(secured)
    assert added.get("token") is None, added


def test_a_pending_account_is_told_to_wait_without_a_challenge(secured):
    email, password, _ = add_account(secured, role="pending")
    answer = sign_in(secured, email, password)
    assert answer.status_code == 200, answer.text
    assert answer.json() == {"next_step": "pending"}


def test_a_wrong_password_never_reaches_the_second_step(secured):
    email, _, _ = add_account(secured)
    answer = sign_in(secured, email, "not-the-password")
    assert answer.status_code == 400, answer.text
    assert "challenge_token" not in answer.text


# --------------------------------------------------------------------------- later sign-ins


def test_a_later_sign_in_needs_a_current_code(secured):
    account = enrolled_account(secured)
    challenge_token = challenge_for(secured, account)
    assert session_status(secured, challenge_token) == 401

    answer = verify(secured, account, challenge_token, account.authenticator.code())

    assert answer.status_code == 200, answer.text
    assert session_status(secured, answer.json()["token"]) == 200


def test_a_wrong_code_is_refused_and_a_used_code_is_not_taken_again(secured):
    account = enrolled_account(secured)
    challenge_token = challenge_for(secured, account)
    wrong = verify(secured, account, challenge_token, account.authenticator.wrong_code())
    assert wrong.status_code == 401, wrong.text

    code = account.authenticator.code()
    assert verify(secured, account, challenge_token, code).status_code == 200

    replayed = verify(secured, account, challenge_for(secured, account), code)
    assert replayed.status_code == 401, f"a used code signed in again: {replayed.text}"


def test_a_challenge_is_spent_by_its_sign_in(secured):
    account = enrolled_account(secured)
    challenge_token = challenge_for(secured, account)
    assert (
        verify(secured, account, challenge_token, account.authenticator.code()).status_code == 200
    )

    again = verify(secured, account, challenge_token, account.authenticator.code())

    assert again.status_code == 401, again.text


def test_five_wrong_codes_end_the_sign_in(secured):
    account = enrolled_account(secured)
    challenge_token = challenge_for(secured, account)
    for _ in range(5):
        wrong = verify(secured, account, challenge_token, account.authenticator.wrong_code())
        assert wrong.status_code == 401, wrong.text

    right = verify(secured, account, challenge_token, account.authenticator.code())

    assert right.status_code == 429, right.text
    assert verified(secured, account)


def test_ten_password_sign_ins_hold_the_account_for_a_while(secured):
    account = enrolled_account(secured)
    for _ in range(9):
        refused = sign_in(secured, account.email, "not-the-password", account.address)
        assert refused.status_code == 400, refused.text

    held = sign_in(secured, account.email, account.password, account.address)

    # the enrollment's sign-in was the first of the ten
    assert held.status_code == 429, held.text
    assert int(held.headers["Retry-After"]) > 0


def test_one_address_is_held_after_a_hundred_sign_in_requests(secured):
    address = new_address()
    answers = [
        sign_in(secured, f"nobody-{attempt}@example.com", "not-the-password", address)
        for attempt in range(101)
    ]
    assert [answer.status_code for answer in answers[:100]] == [400] * 100
    assert answers[100].status_code == 429, answers[100].text
    elsewhere = sign_in(secured, "nobody@example.com", "not-the-password")
    assert elsewhere.status_code == 400, elsewhere.text


# --------------------------------------------------------------------------- recovery codes


def test_a_recovery_code_signs_in_once(secured):
    account = enrolled_account(secured)
    recovery_code = account.recovery_codes[0]

    # typed with spaces and in capitals, as people copy them
    used = verify(
        secured, account, challenge_for(secured, account), f" {recovery_code.upper()} ", True
    )
    assert used.status_code == 200, used.text
    account.token = used.json()["token"]
    assert mfa_status(account)["recovery_codes_remaining"] == 9

    reused = verify(secured, account, challenge_for(secured, account), recovery_code, True)
    assert reused.status_code == 401, f"a used recovery code signed in again: {reused.text}"


def test_a_recovery_code_is_no_authenticator_code_and_back(secured):
    account = enrolled_account(secured)
    challenge_token = challenge_for(secured, account)
    as_code = verify(secured, account, challenge_token, account.recovery_codes[0])
    assert as_code.status_code == 401, as_code.text
    as_recovery = verify(secured, account, challenge_token, account.authenticator.code(), True)
    assert as_recovery.status_code == 401, as_recovery.text


def test_new_recovery_codes_retire_the_old_ones_and_sign_out_everywhere(secured):
    account = enrolled_account(secured)
    old_session = account.token
    with account.client() as client:
        regenerated = client.post(
            f"{MFA}/recovery/codes", json={"code": account.authenticator.code()}
        )
    assert regenerated.status_code == 200, regenerated.text
    new_codes = regenerated.json()["recovery_codes"]
    assert len(new_codes) == 10 and not set(new_codes) & set(account.recovery_codes)
    assert session_status(secured, old_session) == 401

    retired = verify(
        secured, account, challenge_for(secured, account), account.recovery_codes[1], True
    )
    assert retired.status_code == 401, retired.text
    fresh = verify(secured, account, challenge_for(secured, account), new_codes[0], True)
    assert fresh.status_code == 200, fresh.text


def test_managing_the_authenticator_needs_a_valid_code(secured):
    account = enrolled_account(secured)
    with account.client() as client:
        refused = client.post(
            f"{MFA}/recovery/codes", json={"code": account.authenticator.wrong_code()}
        )
    assert refused.status_code == 401, refused.text
    assert session_status(secured, account.token) == 200


# --------------------------------------------------------------------------- replacing it


def test_replacing_the_authenticator_moves_sign_in_to_the_new_one(secured):
    account = enrolled_account(secured)
    old_authenticator = account.authenticator
    with account.client() as client:
        started = client.post(f"{MFA}/replace", json={"code": old_authenticator.code()})
    assert started.status_code == 200, started.text
    assert started.json()["next_step"] == "enroll"

    new_authenticator, confirmed = finish_enrollment(
        secured, started.json()["challenge_token"], account.address
    )

    assert new_authenticator.secret != old_authenticator.secret
    assert set(confirmed) == {"recovery_codes"}, "replacing should not hand out a session"
    assert session_status(secured, account.token) == 401
    old_code = verify(secured, account, challenge_for(secured, account), old_authenticator.code())
    assert old_code.status_code == 401, old_code.text
    account.authenticator = new_authenticator
    assert verified(secured, account)


# --------------------------------------------------------------------------- other sign-ins


@pytest.fixture
def directory(secured, preserve):
    preserve(LDAP_CONFIG, on=secured)
    with serve_directory() as served, secured.client() as client:
        save_ldap_settings(client, served)
        yield served


def test_an_ldap_sign_in_takes_the_same_second_step(secured, directory):
    directory.add_person("mfa-ldap-person", "directory-pass-1", cn="Directory Person")
    with anonymous(secured) as browser:
        answer = browser.post(
            "/api/v1/auths/ldap", json={"user": "mfa-ldap-person", "password": "directory-pass-1"}
        )
    assert answer.status_code == 200, answer.text
    assert answer.json()["next_step"] == "enroll", answer.json()
    assert "token" not in answer.json()

    _, session = finish_enrollment(secured, answer.json()["challenge_token"])
    assert session["email"] == "mfa-ldap-person@example.org"


def sso_callback(secured, browser) -> httpx.Response:
    start = browser.get("/oauth/oidc/login")
    approved = browser.get(start.headers["location"])
    callback = browser.get(approved.headers["location"])
    assert callback.status_code in (302, 307), callback.text
    return callback


def test_an_sso_sign_in_lands_on_the_second_step_with_a_challenge_cookie(secured, idp):
    idp.sign_in_as()
    with browser_for(secured) as browser:
        callback = sso_callback(secured, browser)
        landing = urllib.parse.urlsplit(callback.headers["location"])
        assert dict(urllib.parse.parse_qsl(landing.query)) == {"mfa": "1"}
        assert "token" not in callback.cookies

        cross_site = browser.post(f"{MFA}/challenge", headers={"Origin": "https://elsewhere.test"})
        assert cross_site.status_code == 403, cross_site.text
        challenge = browser.post(f"{MFA}/challenge", headers={"Origin": secured.base_url})
        assert challenge.status_code == 200, challenge.text
        assert challenge.json()["next_step"] == "enroll"
        spent = browser.post(f"{MFA}/challenge", headers={"Origin": secured.base_url})
        assert spent.status_code == 401, "the challenge cookie should be good for one read"

    _, session = finish_enrollment(secured, challenge.json()["challenge_token"])
    with secured.client(session["token"]) as client:
        assert client.get(f"{MFA}/status").json()["required"] is True


def test_the_sso_token_exchange_answers_with_the_second_step(secured, idp):
    person = idp.sign_in_as()
    with browser_for(secured) as browser:
        sso_callback(secured, browser)
        challenge = browser.post(f"{MFA}/challenge", headers={"Origin": secured.base_url})
    finish_enrollment(secured, challenge.json()["challenge_token"])

    access_token = idp.issue_access_token(person)
    with anonymous(secured) as client:
        answer = client.post(
            "/api/v1/auths/oauth/oidc/token/exchange", json={"token": access_token}
        )

    assert answer.status_code == 200, answer.text
    assert answer.json()["next_step"] == "verify" and "token" not in answer.json()


# --------------------------------------------------------------------------- API keys


def test_an_api_key_works_without_a_code_but_not_on_the_mfa_routes(secured):
    account = enrolled_account(secured, role="admin")
    with account.client() as client:
        generated = client.post("/api/v1/auths/api_key")
    assert generated.status_code == 200, generated.text
    api_key = generated.json()["api_key"]

    with secured.client(api_key) as client:
        assert client.get("/api/models").status_code == 200
        assert client.get(f"{MFA}/status").status_code == 403
        current = client.get("/api/v1/auths/admin/config")
        assert current.status_code == 200, current.text
        switched = client.post(
            "/api/v1/auths/admin/config", json={**current.json(), "ENABLE_MFA": False}
        )
    assert switched.status_code == 403, f"an API key changed the MFA switches: {switched.text}"
    with secured.client() as client:
        assert client.get("/api/v1/auths/admin/config").json()["ENABLE_MFA"] is True


# --------------------------------------------------------------------------- the admin


def test_the_admin_signs_one_account_out_everywhere(secured):
    target, bystander = enrolled_account(secured), enrolled_account(secured)
    with secured.client() as client:
        revoked = client.post(f"/api/v1/users/{target.id}/sessions/revoke")
    assert revoked.status_code == 200 and revoked.json() is True, revoked.text

    assert session_status(secured, target.token) == 401
    assert session_status(secured, bystander.token) == 200
    assert verified(secured, target)


def test_only_an_admin_signs_accounts_out_and_never_the_first_admin(secured):
    member, second_admin = enrolled_account(secured), enrolled_account(secured, role="admin")
    with member.client() as client:
        assert client.post(f"/api/v1/users/{second_admin.id}/sessions/revoke").status_code == 401
    with second_admin.client() as client:
        refused = client.post(f"/api/v1/users/{admin_account(secured).id}/sessions/revoke")
    assert refused.status_code == 403, refused.text
    assert session_status(secured, secured.admin_token) == 200


# --------------------------------------------------------------------------- operator reset


def test_an_operator_reset_lets_a_locked_out_account_set_up_again(secured):
    account = enrolled_account(secured)
    reset = operator_reset(secured, account.email, "lost phone, identity checked")
    assert reset.returncode == 0, reset.stderr[-3000:]
    reset_token = reset.stdout.strip().splitlines()[-1]
    assert session_status(secured, account.token) == 401

    answer = sign_in(secured, account.email, account.password, account.address)
    assert answer.json()["next_step"] == "recover", answer.json()
    challenge_token = answer.json()["challenge_token"]
    wrong = mfa_step(
        secured, "recover", {"challenge_token": challenge_token, "reset_token": "x" * 40}
    )
    assert wrong.status_code == 401, wrong.text
    recovered = mfa_step(
        secured, "recover", {"challenge_token": challenge_token, "reset_token": reset_token}
    )
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["next_step"] == "enroll"

    _, session = finish_enrollment(secured, recovered.json()["challenge_token"])
    assert len(session["recovery_codes"]) == 10
    old_recovery = verify(
        secured, account, challenge_for(secured, account), account.recovery_codes[0], True
    )
    assert old_recovery.status_code == 401, old_recovery.text


def test_an_operator_reset_needs_a_reason(secured):
    account = enrolled_account(secured)
    refused = operator_reset(secured, account.email, "   ")
    assert refused.returncode != 0
    assert session_status(secured, account.token) == 200
