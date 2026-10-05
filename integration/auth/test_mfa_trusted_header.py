"""Journey: multi-factor sign-in behind a reverse proxy that names the person in a header.

With MFA on, a sign-in the proxy vouches for (`WEBUI_AUTH_TRUSTED_EMAIL_HEADER`) still answers
with the authenticator step, setup the first time and a code after that. With trusted-header
sign-ins exempted, for a proxy that enforces its own second factor, the same sign-in gets its
session at once and the account's MFA status says no authenticator is required.

Discriminates: in a backend copy, making `is_mfa_required` answer False for `trusted_header`
turns the second-step test red, and making it ignore `MFA_ALLOW_TRUSTED_HEADER_BYPASS` turns the
exemption test red.
"""

from __future__ import annotations

import secrets

import pytest

from harness.instance import ADMIN_EMAIL
from harness.mfa import MFA, finish_enrollment, mfa_step, require_mfa, sign_in

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

EMAIL_HEADER = "X-Forwarded-Email"


@pytest.fixture(scope="module")
def proxied(instance_with):
    instance = instance_with(
        {"WEBUI_AUTH_TRUSTED_EMAIL_HEADER": EMAIL_HEADER, "DEFAULT_USER_ROLE": "user"}
    )
    require_mfa(instance, sign_in_headers={EMAIL_HEADER: ADMIN_EMAIL}, ENABLE_MFA=True)
    return instance


def proxy_sign_in(instance, email: str) -> dict:
    """The web client's empty sign-in form, with the proxy's header naming the person."""
    answer = sign_in(instance, "", "", headers={EMAIL_HEADER: email})
    assert answer.status_code == 200, answer.text
    return answer.json()


def new_email() -> str:
    return f"proxied-{secrets.token_hex(4)}@example.com"


def test_a_proxied_sign_in_still_asks_for_the_authenticator(proxied):
    email = new_email()
    first = proxy_sign_in(proxied, email)
    assert first["next_step"] == "enroll" and "token" not in first, first
    authenticator, _ = finish_enrollment(proxied, first["challenge_token"])

    later = proxy_sign_in(proxied, email)
    assert later["next_step"] == "verify", later
    session = mfa_step(
        proxied,
        "verify",
        {"challenge_token": later["challenge_token"], "code": authenticator.code()},
    )
    assert session.status_code == 200, session.text


def test_exempting_proxied_sign_ins_hands_out_the_session_at_once(proxied):
    admin_headers = {EMAIL_HEADER: ADMIN_EMAIL}
    require_mfa(proxied, sign_in_headers=admin_headers, MFA_ALLOW_TRUSTED_HEADER_BYPASS=True)
    try:
        email = new_email()
        answer = proxy_sign_in(proxied, email)
        assert "token" in answer, f"an exempted proxied sign-in got no session: {answer}"
        with proxied.client(answer["token"]) as client:
            status = client.get(f"{MFA}/status", headers={EMAIL_HEADER: email})
        assert status.json()["required"] is False, status.text
    finally:
        require_mfa(proxied, sign_in_headers=admin_headers, MFA_ALLOW_TRUSTED_HEADER_BYPASS=False)

    assert proxy_sign_in(proxied, email)["next_step"] == "enroll"
