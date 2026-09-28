"""Dependency smoke: the sign-in stack, each library driven through the feature that uses it.

bcrypt (the default) and argon2-cffi hash the password at sign-up and check it at sign-in.
bcrypt stores each password as a salted `$2b$` hash at its default cost of 12, as the database
file shows, and takes at most 72 bytes: a longer new password is refused and a sign-in compares
only the first 72 bytes, which bcrypt 5 would otherwise refuse with an error. A changed password
is hashed afresh and the old one stops working.
Switching `PASSWORD_HASH_ALGORITHM` only changes how new passwords are hashed: every stored
hash names its algorithm, so accounts keep signing in across a switch in either direction. Two
accounts with one password get different argon2 hashes (a salt each), and an argon2 hash that
cannot be parsed refuses the sign-in like a wrong password.
PyJWT signs and checks the session token, whose `iat` and `exp` come from pytz's UTC clock.
authlib builds the SSO redirect and completes the code exchange, while itsdangerous signs the
`owui-session` cookie that carries its state from one to the other. The completed SSO sign-in
also verifies the provider's RS256 ID token and encrypts the stored OAuth session, which is
what cryptography does on this path (the OAuth twins under integration/security complete many
more). When the provider refuses the code exchange, authlib's `OAuthError` carries its
reason into the server log. A bump that breaks one of them fails a sign-in here, not only an
API check in unit/deps. Twin of unit/deps/test_argon2_cffi.py and unit/deps/test_authlib.py.

The argon2 instance runs in a zone far from UTC, so a clock that is not UTC shows in the token.

Discriminates: passes on dev bbfa876af; in a backend copy, `bcrypt.checkpw` answering True lets
the wrong password in, hashing with `bcrypt.gensalt(4, prefix=b"2a")` fails the stored-hash test,
a password update that stores nothing keeps the old password working, dropping the 72-byte cut
fails the long sign-in, argon2 verification answering True does the same on its instance,
`jwt.decode` without signature and expiry checks accepts the flipped and the expired token, a
naive `datetime.now()` for `exp` stretches the lifetime by the zone's 5 h 45 min and dropping
`SessionMiddleware` fails the SSO sign-in at its first step. On dev ef67cc3fa,
`verify_password` sending every hash to bcrypt fails the switch test, argon2's
`InvalidHashError` left uncaught answers a hash of an unknown argon2 variant with a 500 and a
callback error reported by its class name alone drops `invalid_grant` from the log.
"""

from __future__ import annotations

import base64
import contextlib
import json
import sqlite3
import time
import urllib.parse
import uuid

import httpx
import pytest
import sqlalchemy

from harness.actors import create_user, sign_in
from harness.backends import write_rows
from harness.instance import ADMIN_EMAIL, ADMIN_PASSWORD, LaunchedInstance
from harness.oidc_provider import browser_for, session_user, shared_provider, sso_env
from harness.prepared_data import RunningBackend, serving

pytestmark = [
    pytest.mark.depcheck,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
# a zone off UTC by hours and minutes, so a naive clock cannot pass for UTC
ARGON2_OFF_UTC = {"PASSWORD_HASH_ALGORITHM": "argon2", "TZ": "Asia/Kathmandu"}
# past bcrypt's 72 bytes, so only a hash of the whole password tells the last byte apart
LONG_PASSWORD = "argon2-" + "x" * 72 + "!"
FOUR_WEEKS = 4 * 7 * 24 * 3600
ARGON2 = {"PASSWORD_HASH_ALGORITHM": "argon2"}
FIRST_PASSWORD = "bcrypt-first-password"
# argon2 raises InvalidHashError for the unknown variant and VerificationError for the rest
DAMAGED_ARGON2_HASHES = {
    "unknown-variant": "$argon2x$v=19$m=65536,t=3,p=4$c2FsdHNhbHQ$aGFzaGhhc2g",
    "undecodable": "$argon2id$v=19$m=65536,t=3,p=4$not-a-salt$not-a-hash",
}


@pytest.fixture
def argon2_instance(instance_with):
    return instance_with(ARGON2_OFF_UTC)


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with(sso_env(idp))


def _sign_in(instance: LaunchedInstance, email: str, password: str) -> httpx.Response:
    return httpx.post(
        f"{instance.base_url}/api/v1/auths/signin",
        json={"email": email, "password": password},
        timeout=60.0,
    )


def _session_status(instance: LaunchedInstance, token: str) -> int:
    # not /api/v1/auths/, which checks `exp` itself and would hide PyJWT's own check
    with instance.client(token) as client:
        return client.get("/api/v1/users/user/settings").status_code


def _claims(token: str) -> dict:
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def _flip_middle(segment: str) -> str:
    """One base64url character changed mid-segment, where every bit counts."""
    middle = len(segment) // 2
    replacement = "A" if segment[middle] != "A" else "B"
    return segment[:middle] + replacement + segment[middle + 1 :]


def _with_flipped_signature(signed: str) -> str:
    *signed_part, signature = signed.split(".")
    return ".".join([*signed_part, _flip_middle(signature)])


def _save_admin_config(instance: LaunchedInstance, **changes) -> None:
    with instance.client() as client:
        current = client.get(ADMIN_CONFIG)
        current.raise_for_status()
        saved = client.post(ADMIN_CONFIG, json={**current.json(), **changes})
    assert saved.status_code == 200, saved.text


def test_bcrypt_checks_the_password_at_sign_in(instance, make_user):
    account = make_user(password="bcrypt-password-1")

    assert _sign_in(instance, account.email, "bcrypt-password-1").status_code == 200
    assert _sign_in(instance, account.email, "bcrypt-password-2").status_code == 400


def _stored_hashes(instance: LaunchedInstance, *emails: str) -> list[str]:
    """The password column of each account, read from the instance's database file."""
    if not instance.database_url.startswith("sqlite"):
        pytest.skip("reads the SQLite database file")
    location = f"file:{instance.data_dir / 'webui.db'}?mode=ro"
    with contextlib.closing(sqlite3.connect(location, uri=True)) as database:
        return [
            database.execute("SELECT password FROM auth WHERE email = ?", (email,)).fetchone()[0]
            for email in emails
        ]


def test_bcrypt_stores_every_password_salted_at_cost_12(instance, make_user):
    first = make_user(password="the-same-password")
    second = make_user(password="the-same-password")

    hashes = _stored_hashes(instance, first.email, second.email)

    assert all(stored.startswith("$2b$12$") and len(stored) == 60 for stored in hashes), hashes
    assert hashes[0] != hashes[1], "two accounts with one password share a hash"


def test_a_password_past_72_bytes_is_refused_and_a_sign_in_compares_72(instance, make_user):
    longest = "b" * 71 + "!"
    account = make_user(password=longest)
    with instance.client() as client:
        too_long = client.post(
            "/api/v1/auths/add",
            json={
                "name": "Long",
                "email": f"long-{uuid.uuid4().hex[:8]}@example.com",
                "password": longest + "x",
                "role": "user",
            },
        )

    assert too_long.status_code == 400, too_long.text
    assert _sign_in(instance, account.email, longest + " and more").status_code == 200
    assert _sign_in(instance, account.email, "b" * 71 + "?").status_code == 400


def test_a_changed_password_replaces_the_old_one(instance, make_user):
    account = make_user(password="old-harbour-password")
    with account.client() as client:
        changed = client.post(
            "/api/v1/auths/update/password",
            json={"password": "old-harbour-password", "new_password": "new-harbour-password"},
        )

    assert changed.status_code == 200, changed.text
    assert _sign_in(instance, account.email, "new-harbour-password").status_code == 200
    assert _sign_in(instance, account.email, "old-harbour-password").status_code == 400


def test_argon2_hashes_the_whole_password_at_sign_up(argon2_instance, preserve):
    preserve("admin_config", on=argon2_instance)
    _save_admin_config(argon2_instance, ENABLE_SIGNUP=True)
    email = f"argon2-{uuid.uuid4().hex[:8]}@example.com"

    signed_up = httpx.post(
        f"{argon2_instance.base_url}/api/v1/auths/signup",
        json={"name": "Argon", "email": email, "password": LONG_PASSWORD},
        timeout=60.0,
    )

    assert signed_up.status_code == 200, signed_up.text
    assert _sign_in(argon2_instance, email, LONG_PASSWORD).status_code == 200
    assert _sign_in(argon2_instance, email, LONG_PASSWORD[:-1] + "?").status_code == 400


def test_a_session_token_with_a_flipped_signature_is_refused(instance, make_user):
    account = make_user()

    assert _session_status(instance, account.token) == 200
    assert _session_status(instance, _with_flipped_signature(account.token)) == 401


def test_a_session_token_stops_working_once_it_expires(instance, make_user, preserve):
    account = make_user()
    preserve("admin_config")
    _save_admin_config(instance, JWT_EXPIRES_IN="2s")

    token = sign_in(instance, account.email, account.password)
    claims = _claims(token)

    assert claims["exp"] - claims["iat"] in (1, 2)
    assert _session_status(instance, token) == 200
    time.sleep(3)  # the token's own lifetime: nothing to wait on but the clock
    assert _session_status(instance, token) == 401


def test_the_session_token_runs_four_weeks_from_now_in_utc(argon2_instance):
    signed_in = _sign_in(argon2_instance, ADMIN_EMAIL, ADMIN_PASSWORD)
    signed_in.raise_for_status()
    claims = _claims(signed_in.json()["token"])

    assert abs(claims["iat"] - time.time()) < 60, "the token's iat is not the current UTC time"
    assert abs(claims["exp"] - claims["iat"] - FOUR_WEEKS) <= 1


def test_sso_starts_at_the_provider_with_a_signed_session_cookie(sso, idp):
    with browser_for(sso) as browser:
        start = browser.get("/oauth/oidc/login")
        session_cookie = browser.cookies.get("owui-session")

    assert start.status_code == 302, start.text
    authorize = urllib.parse.urlsplit(start.headers["location"])
    query = dict(urllib.parse.parse_qsl(authorize.query))
    assert f"{authorize.scheme}://{authorize.netloc}{authorize.path}" == f"{idp.base_url}/authorize"
    assert query["client_id"] == idp.client_id
    assert query["response_type"] == "code"
    assert query["state"]
    # itsdangerous: data.timestamp.signature
    assert session_cookie and session_cookie.count(".") == 2


def _approved_callback(sso: LaunchedInstance) -> tuple[str, str]:
    """The provider's redirect back to Open WebUI, and the session cookie set on the way out."""
    with browser_for(sso) as browser:
        start = browser.get("/oauth/oidc/login")
        approved = browser.get(start.headers["location"])
        return approved.headers["location"], browser.cookies["owui-session"]


def _finish(callback_url: str, session_cookie: str) -> httpx.Response:
    return httpx.get(
        callback_url, headers={"Cookie": f"owui-session={session_cookie}"}, timeout=60.0
    )


def test_a_forged_session_cookie_fails_the_sso_callback_and_an_intact_one_signs_in(sso, idp):
    person = idp.sign_in_as()
    callback_url, session_cookie = _approved_callback(sso)
    forged = _finish(callback_url, _with_flipped_signature(session_cookie))

    assert "token" not in forged.cookies, "a session cookie with a broken signature was trusted"
    assert "error=" in forged.headers["location"]

    callback_url, session_cookie = _approved_callback(sso)
    token = _finish(callback_url, session_cookie).cookies.get("token")
    assert token, "the SSO sign-in with the intact session cookie failed"
    assert session_user(sso, token)["email"] == person["email"]


def _hashes_by_email(data_dir) -> dict[str, str]:
    engine = sqlalchemy.create_engine(f"sqlite:///{data_dir / 'webui.db'}")
    try:
        with engine.connect() as connection:
            rows = connection.execute(sqlalchemy.text("SELECT email, password FROM auth"))
            return {email: password for email, password in rows}
    finally:
        engine.dispose()


def _signs_in(server: RunningBackend, email: str, password: str) -> bool:
    with server.client() as client:
        answer = client.post("/api/v1/auths/signin", json={"email": email, "password": password})
    assert answer.status_code in (200, 400), f"sign-in answered {answer.status_code}: {answer.text}"
    return answer.status_code == 200


def _add_account(server: RunningBackend, admin_token: str, email: str, password: str) -> None:
    with server.client(admin_token) as client:
        added = client.post(
            "/api/v1/auths/add",
            json={"name": "Hashed", "email": email, "password": password, "role": "user"},
        )
    assert added.status_code == 200, added.text


@pytest.mark.slow
def test_accounts_keep_signing_in_when_the_hash_algorithm_switches(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    email = "switch@example.com"
    with serving(data_dir) as bcrypt_server, bcrypt_server.client() as client:
        signed_up = client.post(
            "/api/v1/auths/signup",
            json={"name": "Switch", "email": email, "password": FIRST_PASSWORD},
        )
        assert signed_up.status_code == 200, signed_up.text
    assert _hashes_by_email(data_dir)[email].startswith("$2b$")

    with serving(data_dir, ARGON2) as argon2_server:
        assert _signs_in(argon2_server, email, FIRST_PASSWORD), "the bcrypt account was refused"
        token = sign_in(argon2_server, email, FIRST_PASSWORD)
        with argon2_server.client(token) as client:
            changed = client.post(
                "/api/v1/auths/update/password",
                json={"password": FIRST_PASSWORD, "new_password": LONG_PASSWORD},
            )
        assert changed.status_code == 200 and changed.json() is True, changed.text
        _add_account(argon2_server, token, "twin-a@example.com", LONG_PASSWORD)
        _add_account(argon2_server, token, "twin-b@example.com", LONG_PASSWORD)
    hashes = _hashes_by_email(data_dir)
    assert all(hashes[who].startswith("$argon2") for who in (email, "twin-a@example.com"))
    assert hashes["twin-a@example.com"] != hashes["twin-b@example.com"], "the hash has no salt"

    # bcrypt refuses to set a password this long, so only the argon2 hash can let it in
    with serving(data_dir) as bcrypt_again:
        assert _signs_in(bcrypt_again, email, LONG_PASSWORD), "the argon2 account was refused"
        assert not _signs_in(bcrypt_again, email, LONG_PASSWORD[:-1] + "?")


@pytest.mark.parametrize("damage", DAMAGED_ARGON2_HASHES)
def test_a_damaged_argon2_hash_refuses_the_sign_in(argon2_instance, damage):
    account = create_user(argon2_instance)
    write_rows(
        argon2_instance,
        "UPDATE auth SET password = :password WHERE email = :email",
        [{"password": DAMAGED_ARGON2_HASHES[damage], "email": account.email}],
    )

    refused = _sign_in(argon2_instance, account.email, account.password)

    assert refused.status_code == 400, f"HTTP {refused.status_code}: {refused.text}"


def test_a_refused_code_exchange_logs_the_reason_the_provider_gave(sso, idp):
    idp.sign_in_as()
    callback_url, session_cookie = _approved_callback(sso)
    code = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(callback_url).query))["code"]
    spent = httpx.post(
        f"{idp.base_url}/token",
        data={"grant_type": "authorization_code", "code": code},
        timeout=30.0,
    )
    assert spent.status_code == 200, spent.text
    offset = sso.log_size()

    refused = _finish(callback_url, session_cookie)

    assert "token" not in refused.cookies, "a spent code still signed someone in"
    reported = [
        line
        for line in sso.log_since(offset).splitlines()
        if "authorize_access_token for provider oidc" in line
    ]
    assert reported, "the refused code exchange was not logged"
    assert "invalid_grant" in reported[0], reported[0]
