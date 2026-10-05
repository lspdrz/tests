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
PyJWT signs and checks the session token, whose `iat` and `exp` come from pytz's UTC clock: a
token minted elsewhere with the server's key and HS256 is honoured, one in another HMAC, unsigned,
signed with another key or not a JWT at all is refused. With user info forwarding and a JWT
secret set, PyJWT also signs who is asking into the header the provider gets.
authlib builds the SSO redirect and completes the code exchange, while itsdangerous signs the
`owui-session` cookie that carries its state from one to the other. The completed SSO sign-in
also verifies the provider's RS256 ID token and encrypts the stored OAuth session, which is
what cryptography does on this path (the OAuth twins under integration/security complete many
more): the provider's tokens are kept as a Fernet token under a key derived from the secret key,
decrypted again when a `system_oauth` connection is sent the access token, and a stored token
that no longer decrypts is not forwarded; a session key Fernet cannot use stops the boot. When
the provider refuses the code exchange, authlib's `OAuthError` carries its reason into the
server log. A bump that breaks one of them fails a sign-in or a forwarded token here, not only
an API check in unit/deps. Twin of unit/deps/test_argon2_cffi.py and unit/deps/test_authlib.py.

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
callback error reported by its class name alone drops `invalid_grant` from the log. Storing the
session tokens unencrypted fails the encrypted-at-rest test, a `_decrypt_token` that parses the
stored text without decrypting it fails both stored-token tests and a Fernet that answers a token of
another key with garbage in place of `InvalidToken` fails the undecryptable one; a Fernet that
accepts a malformed key lets the instance with the unusable session key boot. Also on ef67cc3fa, a
session check that takes HS512 too fails the other-HMAC case, one without the signature check fails
every forged case, sessions signed HS384 fail the minted-elsewhere test and a forwarded token signed
HS512 fails the forwarding test. Twin of the behaviour half of unit/deps/test_pyjwt.py, which keeps
its sweep over the source.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import sqlite3
import time
import urllib.parse
import uuid

import httpx
import jwt
import pytest
import sqlalchemy
from cryptography.fernet import Fernet

from harness.actors import create_user, sign_in
from harness.backends import read_rows, write_rows
from harness.instance import ADMIN_EMAIL, ADMIN_PASSWORD, WEBUI_SECRET_KEY, LaunchedInstance
from harness.listener import json_answer
from harness.oidc_provider import (
    browser_for,
    oauth_settings,
    session_user,
    shared_provider,
    sso_env,
)
from harness.oidc_provider import sign_in as sso_sign_in
from harness.prepared_data import RunningBackend, boot_until_settled, serving
from harness.upstream import MOCK_MODEL_ID

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
FORWARDING_SECRET = "forwarding-secret-0123456789abcdef"
FORWARDING_AS_JWT = {
    "ENABLE_FORWARD_USER_INFO_HEADERS": "true",
    "FORWARD_USER_INFO_HEADER_JWT_SECRET": FORWARDING_SECRET,
}
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


def test_a_session_token_signed_elsewhere_with_the_servers_key_is_honoured(instance, make_user):
    account = make_user()
    claims = {**_claims(account.token), "jti": str(uuid.uuid4())}
    minted = jwt.encode(claims, WEBUI_SECRET_KEY, algorithm="HS256")

    assert _session_status(instance, minted) == 200


@pytest.mark.parametrize(
    ("algorithm", "key"),
    [
        pytest.param("HS512", WEBUI_SECRET_KEY, id="another-hmac"),
        pytest.param("none", None, id="unsigned"),
        pytest.param("HS256", "a-key-the-server-never-had-0123456789", id="another-key"),
    ],
)
def test_a_session_token_not_signed_as_the_server_signs_is_refused(
    instance, make_user, algorithm, key
):
    forged = jwt.encode(_claims(make_user().token), key, algorithm=algorithm)

    assert _session_status(instance, forged) == 401


def test_the_provider_is_told_who_asks_in_a_jwt_signed_with_the_forwarding_secret(
    instance_with,
):
    forwarding = instance_with(FORWARDING_AS_JWT)
    account = create_user(forwarding)
    body = {"model": MOCK_MODEL_ID, "messages": [{"role": "user", "content": "hi"}]}
    with account.client() as client:
        answered = client.post("/api/chat/completions", json={**body, "stream": False})
    assert answered.status_code == 200, answered.text

    [sent] = forwarding.upstream.requests_to("/chat/completions")
    headers = {name.lower(): value for name, value in sent.headers.items()}
    token = headers["x-openwebui-user-jwt"]
    claims = jwt.decode(token, FORWARDING_SECRET, algorithms=["HS256"], issuer="open-webui")
    assert (claims["sub"], claims["email"], claims["role"]) == (account.id, account.email, "user")
    assert claims["exp"] - claims["iat"] == 300
    assert abs(claims["iat"] - time.time()) < 60
    with pytest.raises(jwt.InvalidSignatureError):
        jwt.decode(token, "not-the-forwarding-secret-0123456789", algorithms=["HS256"])


@pytest.mark.parametrize("token", ["not-a-token", "not.a.token", "e30.e30."])
def test_a_session_token_that_is_no_jwt_is_refused(instance, token):
    assert _session_status(instance, token) == 401


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
        # the change signs out every session, this one too (24e30d1cb)
        token = sign_in(argon2_server, email, LONG_PASSWORD)
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


TOOL_SPEC = {"openapi": "3.0.0", "info": {"title": "SSO tools", "version": "1"}, "paths": {}}


def _fernet(secret: str) -> Fernet:
    """The session store's key: the secret key hashed into Fernet's 32 url-safe bytes."""
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest()))


def _signed_in_sso_admin(sso, idp) -> tuple[httpx.Client, str]:
    """The SSO admin's browser and the email the provider signed in."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        person = idp.sign_in_as(roles=["admin"])
        result = sso_sign_in(sso)
    assert result.token, f"the sign-in failed: {result.error}"
    return result.browser, person["email"]


def _stored_session(sso, email: str) -> dict:
    [session] = read_rows(
        sso,
        'SELECT s.id, s.token FROM oauth_session s JOIN "user" u ON u.id = s.user_id '
        "WHERE u.email = :email",
        {"email": email},
    )
    return session


def _forwarded_token(browser: httpx.Client, listener) -> str | None:
    """Verify a `system_oauth` tool server; returns the bearer token it was sent."""
    listener.route("GET", "/openapi.json", json_answer(TOOL_SPEC))
    before = len(listener.requests_to("/openapi.json"))
    browser.post(
        "/api/v1/configs/tool_servers/verify",
        json={
            "url": listener.base_url,
            "path": "openapi.json",
            "type": "openapi",
            "auth_type": "system_oauth",
            "key": "",
            "config": {},
        },
    )
    fetches = listener.requests_to("/openapi.json")[before:]
    authorization = fetches[-1].headers.get("Authorization") if fetches else None
    return authorization.removeprefix("Bearer ") if authorization else None


def test_the_provider_token_is_kept_encrypted_and_forwarded_decrypted(sso, idp, listener):
    browser, email = _signed_in_sso_admin(sso, idp)
    access_token = idp.issued[-1]["access_token"]

    stored = _stored_session(sso, email)["token"]
    assert access_token not in stored, "the provider's access token is stored in the clear"
    decrypted = json.loads(_fernet(WEBUI_SECRET_KEY).decrypt(stored.encode()))
    assert decrypted["access_token"] == access_token
    assert _forwarded_token(browser, listener) == access_token


def test_a_stored_token_that_no_longer_decrypts_is_not_forwarded(sso, idp, listener):
    browser, email = _signed_in_sso_admin(sso, idp)
    offset = sso.log_size()
    session = _stored_session(sso, email)
    foreign = _fernet("another-secret").encrypt(json.dumps(idp.issued[-1]).encode()).decode()
    write_rows(
        sso,
        "UPDATE oauth_session SET token = :token WHERE id = :id",
        [{"token": foreign, "id": session["id"]}],
    )

    assert _forwarded_token(browser, listener) is None, (
        "a token encrypted under another key was forwarded"
    )
    assert "Error decrypting tokens: InvalidToken" in sso.log_since(offset)


@pytest.mark.parametrize(
    ("session_key", "boots"),
    [
        pytest.param(Fernet.generate_key().decode(), True, id="fernet-key"),
        pytest.param("!" * 44, False, id="not-base64"),
    ],
)
def test_a_session_key_of_fernet_length_must_be_a_fernet_key(tmp_path, session_key, boots):
    # a 44-character key is taken as a Fernet key as it is; any other length is hashed into one
    outcome = boot_until_settled(
        tmp_path, settings={"OAUTH_SESSION_TOKEN_ENCRYPTION_KEY": session_key}
    )

    assert outcome.healthy is boots, outcome.log[-3000:]
    if not boots:
        assert "Error initializing Fernet with provided key" in outcome.log
