"""Journey: the environment variables behind multi-factor sign-in, and the boots MFA refuses.

`ENABLE_MFA`, `MFA_ALLOW_OAUTH_BYPASS` and `MFA_ALLOW_TRUSTED_HEADER_BYPASS` set the starting
values of the admin's switches on a first boot, so the very first account signs up into the
authenticator setup; after that the saved value wins over the variable. While MFA is on the
instance refuses to start without persistent configuration, with an `MFA_ENCRYPTION_KEY` that is
no key, or when the key (derived from `WEBUI_SECRET_KEY` unless `MFA_ENCRYPTION_KEY` is set)
can no longer read a stored authenticator; with the old key back it starts again and the
authenticator still works.

Discriminates: in a backend copy, dropping `validate_mfa_configuration` from startup turns the
refused-boot tests red (the instance starts anyway), and seeding `auth.mfa.enable` from a
hard-coded False turns every test that boots with `ENABLE_MFA=true` red (the first account signs
up without any second step).
"""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from harness.instance import ADMIN_EMAIL, ADMIN_PASSWORD
from harness.mfa import MFA, Authenticator, anonymous
from harness.prepared_data import boot_until_settled, serving

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

MFA_ON = {"ENABLE_MFA": "true"}


def sign_up_admin(backend) -> dict:
    """The first visitor's sign-up, which makes them the admin."""
    with anonymous(backend) as browser:
        answer = browser.post(
            "/api/v1/auths/signup",
            json={"name": "Admin", "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
        )
    assert answer.status_code == 200, answer.text
    return answer.json()


def enroll_with_challenge(backend, challenge_token: str) -> tuple[Authenticator, dict]:
    with anonymous(backend) as browser:
        setup = browser.post(f"{MFA}/enroll/start", json={"challenge_token": challenge_token})
        assert setup.status_code == 200, setup.text
        authenticator = Authenticator(setup.json()["manual_key"])
        confirmed = browser.post(
            f"{MFA}/enroll/confirm",
            json={"challenge_token": challenge_token, "code": authenticator.code()},
        )
    assert confirmed.status_code == 200, confirmed.text
    return authenticator, confirmed.json()


def admin_sign_in(backend) -> dict:
    with anonymous(backend) as browser:
        answer = browser.post(
            "/api/v1/auths/signin", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
        )
    assert answer.status_code == 200, answer.text
    return answer.json()


def enrolled_admin(data_dir, settings: dict) -> Authenticator:
    """Boot on `data_dir`, sign the admin up through setup, stop again."""
    with serving(data_dir, settings=settings) as backend:
        signed_up = sign_up_admin(backend)
        assert signed_up["next_step"] == "enroll", signed_up
        authenticator, _ = enroll_with_challenge(backend, signed_up["challenge_token"])
    return authenticator


def test_the_variables_set_the_switches_on_a_first_boot(tmp_path):
    settings = {
        **MFA_ON,
        "MFA_ALLOW_OAUTH_BYPASS": "true",
        "MFA_ALLOW_TRUSTED_HEADER_BYPASS": "true",
    }
    with serving(tmp_path, settings=settings) as backend:
        signed_up = sign_up_admin(backend)
        assert signed_up["next_step"] == "enroll" and "token" not in signed_up, signed_up
        _, session = enroll_with_challenge(backend, signed_up["challenge_token"])
        with backend.client(session["token"]) as client:
            switches = client.get("/api/v1/auths/admin/config").json()
    assert switches["ENABLE_MFA"] is True
    assert switches["MFA_ALLOW_OAUTH_BYPASS"] is True
    assert switches["MFA_ALLOW_TRUSTED_HEADER_BYPASS"] is True


def test_the_saved_switch_wins_over_the_variable_on_later_boots(tmp_path):
    with serving(tmp_path, settings=MFA_ON) as backend:
        signed_up = sign_up_admin(backend)
        _, session = enroll_with_challenge(backend, signed_up["challenge_token"])
        with backend.client(session["token"]) as client:
            current = client.get("/api/v1/auths/admin/config").json()
            saved = client.post("/api/v1/auths/admin/config", json={**current, "ENABLE_MFA": False})
        assert saved.status_code == 200, saved.text

    with serving(tmp_path, settings=MFA_ON) as backend:
        answer = admin_sign_in(backend)
    assert "token" in answer, f"ENABLE_MFA=true overrode the saved switch: {answer}"


@pytest.mark.parametrize(
    "settings",
    [
        {"ENABLE_PERSISTENT_CONFIG": "false"},
        {"MFA_ENCRYPTION_KEY": "not-a-fernet-key"},
    ],
    ids=["without-persistent-config", "with-a-key-that-is-no-key"],
)
def test_mfa_refuses_to_boot_without_what_it_needs(tmp_path, settings):
    outcome = boot_until_settled(tmp_path, settings={**MFA_ON, **settings})
    assert not outcome.healthy, f"the instance started with MFA on and {settings}"
    assert outcome.exit_code not in (None, 0)


def test_mfa_off_boots_whatever_the_key(tmp_path):
    outcome = boot_until_settled(tmp_path, settings={"MFA_ENCRYPTION_KEY": "not-a-fernet-key"})
    assert outcome.healthy, outcome.log[-3000:]


def test_a_changed_secret_key_cannot_read_the_authenticators_and_the_old_one_can(tmp_path):
    enrolled_admin(tmp_path, MFA_ON)

    changed = boot_until_settled(tmp_path, settings={**MFA_ON, "WEBUI_SECRET_KEY": "another-key"})
    assert not changed.healthy, "the instance started with authenticators it cannot read"

    with serving(tmp_path, settings=MFA_ON) as backend:
        assert admin_sign_in(backend)["next_step"] == "verify"


def test_an_encryption_key_of_its_own_keeps_the_authenticators(tmp_path):
    key = Fernet.generate_key().decode()
    authenticator = enrolled_admin(tmp_path, {**MFA_ON, "MFA_ENCRYPTION_KEY": key})

    without_key = boot_until_settled(tmp_path, settings=MFA_ON)
    assert not without_key.healthy, "the instance started without the key its authenticators need"

    with serving(tmp_path, settings={**MFA_ON, "MFA_ENCRYPTION_KEY": key}) as backend:
        challenge = admin_sign_in(backend)
        with anonymous(backend) as browser:
            session = browser.post(
                f"{MFA}/verify",
                json={
                    "challenge_token": challenge["challenge_token"],
                    "code": authenticator.code(),
                },
            )
    assert session.status_code == 200, session.text
